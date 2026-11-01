"""The same real process/HTTP scenarios run against the original and both adapters."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

import lab
import release

PROJECT = Path(__file__).resolve().parents[1]


class IncidentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'workspace'
        lab.initialize(self.root)

    def command(self, *args):
        extra = ['--engine', 'prefect'] if os.getenv('FLOW_ENGINE') == 'prefect' else []
        result = subprocess.run([sys.executable, str(PROJECT / 'lab.py'), '--workspace',
                                 str(self.root), 'run', *args, *extra],
                                capture_output=True, text=True, timeout=120)
        print(json.dumps(dict(command=list(args), exit_code=result.returncode,
                              stdout=result.stdout, stderr=result.stderr), ensure_ascii=False))
        return result

    def pointer_hash(self):
        return hashlib.sha256((self.root / 'accepted.json').read_bytes()).hexdigest()

    @contextlib.contextmanager
    def server(self, warmup=0):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        extra = ['--engine', 'fastapi'] if os.getenv('HTTP_ENGINE') == 'fastapi' else []
        with tempfile.TemporaryFile(mode='w+') as log:
            process = subprocess.Popen([sys.executable, str(PROJECT / 'service.py'),
                '--workspace', str(self.root), '--port', str(port),
                '--warmup-seconds', str(warmup), *extra], stdout=log, stderr=log)
            try:
                def request(path, body=None):
                    data = None if body is None else json.dumps(body).encode()
                    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=data,
                                                 headers={'Content-Type': 'application/json'})
                    try:
                        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                        with opener.open(req, timeout=10) as response:
                            return response.status, json.load(response)
                    except urllib.error.HTTPError as exc:
                        return exc.code, json.load(exc)
                deadline = time.monotonic() + 30
                while True:
                    try:
                        request('/health')
                        break
                    except OSError:
                        if time.monotonic() > deadline or process.poll() is not None:
                            log.seek(0)
                            self.fail('Server failed: ' + log.read())
                        time.sleep(.025)
                yield request
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def test_subprocess_recovery_and_content_cache(self):
        self.assertNotEqual(self.command('--fail-after-prepare').returncode, 0)
        result = self.command()
        self.assertEqual(result.returncode, 0)
        self.assertTrue('CACHE_HIT' in result.stdout+result.stderr or 'Cached' in result.stdout+result.stderr)
        (self.root / 'input.txt').write_text('Changed bytes at the same path\n')
        result = self.command()
        self.assertEqual(result.returncode, 0)
        self.assertIn('PREPARED', result.stdout+result.stderr)
        for key, value in [('template_sha256', 'f'*64), ('max_length', 64),
                           ('tokenizer_revision', 'r2'), ('code_revision', 'prepare-v2')]:
            config = lab.read(self.root / 'config.json')
            config[key] = value
            lab.write(self.root / 'config.json', config)
            result = self.command()
            self.assertIn('PREPARED', result.stdout+result.stderr)

    def test_subprocess_incomplete_foreign_reject_preserve_pointer(self):
        before = self.pointer_hash()
        for name in ['report-reject.json', 'report-incomplete.json', 'report-A.json']:
            with self.subTest(report=name):
                self.assertNotEqual(self.command('--report', name).returncode, 0)
                after = self.pointer_hash()
                print('POINTER', name, before, after)
                self.assertEqual(before, after)

    def test_atomic_disk_failure(self):
        before = self.pointer_hash()
        with patch('os.replace', side_effect=OSError('injected disk failure')):
            with self.assertRaises(OSError):
                release.publish(lab.read(self.root/'manifest-B.json'),
                                lab.read(self.root/'report-B.json'), self.root/'accepted.json')
        self.assertEqual(before, self.pointer_hash())
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_subprocess_repeat_publication(self):
        self.assertEqual(self.command().returncode, 0)
        self.assertEqual(self.command().returncode, 0)
        events = [json.loads(line) for line in (self.root/'publications.jsonl').read_text().splitlines()]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['model_id'], 'fixture-B')

    def test_http_readiness_blocks_backend_until_warmup(self):
        with self.server(warmup=2) as request:
            self.assertEqual(request('/health')[0], 200)
            self.assertEqual(request('/ready')[0], 503)
            calls = request('/debug')[1]['calls']
            self.assertEqual(request('/generate', {'prompt': 'hello'})[0], 503)
            self.assertEqual(request('/debug')[1]['calls'], calls)
            deadline = time.monotonic()+10
            while request('/ready')[0] != 200 and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertEqual(request('/ready')[0], 200)
            self.assertTrue(request('/debug')[1]['warmup_finished'])
            self.assertEqual(request('/generate', {'prompt': 'hello'})[0], 200)

    def test_http_loaded_version_and_rollback(self):
        payload = {'prompt': 'hello', 'max_new_tokens': 32}
        with self.server() as request:
            self.assertEqual(request('/generate', payload)[1]['model_id'], 'fixture-A')
            self.assertEqual(self.command().returncode, 0)
            answer = request('/generate', payload)[1]
            self.assertIn('fixture-A', answer['answer'])
            self.assertEqual(answer['model_id'], 'fixture-A')
        with self.server() as request:
            answer = request('/generate', payload)[1]
            self.assertEqual(answer['model_id'], 'fixture-B')
            self.assertIn('fixture-B', answer['answer'])
        release.publish(lab.read(self.root/'manifest-A.json'), lab.read(self.root/'report-A.json'),
                        self.root/'accepted.json')
        with self.server() as request:
            answer = request('/generate', payload)[1]
            self.assertEqual(answer['model_id'], 'fixture-A')
            self.assertIn('fixture-A', answer['answer'])

    def test_http_invalid_requests_never_call_backend(self):
        with self.server() as request:
            calls = request('/debug')[1]['calls']
            for payload in [{'prompt': ''}, {'prompt': 'x', 'max_new_tokens': 0},
                            {'prompt': 'x', 'max_new_tokens': 100000},
                            {'prompt': 'x', 'max_new_tokens': '32'}]:
                with self.subTest(payload=payload):
                    status, _ = request('/generate', payload)
                    self.assertTrue(400 <= status < 500, status)
                    self.assertEqual(request('/debug')[1]['calls'], calls)
            status, answer = request('/generate', {'prompt': 'valid', 'max_new_tokens': 32})
            self.assertEqual(status, 200)
            self.assertTrue(answer['request_id'])
            self.assertEqual(request('/debug')[1]['calls'], calls+1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
