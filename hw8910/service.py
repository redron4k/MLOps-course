"""Shared serving state; stdlib compatibility adapter and FastAPI production adapter."""
import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from release import canonical, digest, verify_artifact


class StubBackend:
    def __init__(self, manifest, root=None):
        self.loaded_id = manifest['model_id']
        self.calls = 0
        self.runtime_sha256 = manifest['runtime_sha256']

    def warmup(self):
        pass

    def generate(self, prompt, limit):
        self.calls += 1
        time.sleep(.01)
        return {'answer': f'STUB ONLY: {self.loaded_id}', 'output_tokens': 1}


class NotReady(ValueError):
    pass


class ServiceState:
    def __init__(self, root, warmup_seconds=0, backend_factory=StubBackend):
        self.root = Path(root)
        self.backend = None
        self.error = None
        self.warmup_finished = threading.Event()
        self.lock = threading.Lock()
        # Snapshot manifest exactly once. Metadata is never read from a moving pointer.
        self.manifest = json.loads((self.root/'accepted.json').read_text())
        self.max_tokens = self.manifest['runtime']['generation']['max_new_tokens']
        self.max_chars = 4096
        self.real = self.manifest.get('kind') in ('base', 'adapter')
        if self.real and backend_factory is StubBackend:
            raise ValueError('Real manifest requires an explicit real backend')
        self.ready = False

        def load():
            try:
                if digest(canonical(self.manifest['runtime'])) != self.manifest['runtime_sha256']:
                    raise ValueError('Runtime fingerprint mismatch')
                verify_artifact(self.manifest, self.root)
                self.backend = backend_factory(self.manifest, self.root)
                self.backend.warmup()
                time.sleep(warmup_seconds)
                self.warmup_finished.set()
                self.ready = True
            except Exception as exc:
                self.error = f'{type(exc).__name__}: {exc}'
        self.loader = threading.Thread(target=load, daemon=True)
        self.loader.start()

    def generate(self, request):
        started = time.perf_counter()
        if not isinstance(request, dict):
            raise ValueError('JSON object required')
        prompt = request.get('prompt')
        limit = request.get('max_new_tokens', self.max_tokens)
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > self.max_chars:
            raise ValueError('prompt must contain 1..4096 characters')
        if type(limit) is not int or not 1 <= limit <= self.max_tokens:
            raise ValueError(f'max_new_tokens must be an integer in [1, {self.max_tokens}]')
        # Real evaluations use a fixed cap. Shorter caps can change strict JSON accuracy.
        if self.real and limit != self.max_tokens:
            raise ValueError('Real serving requires the exact evaluated token cap')
        if not self.ready:
            raise NotReady('Model is loading or warming up')
        with self.lock:
            generation_started = time.perf_counter()
            result = self.backend.generate(prompt, limit)
            elapsed = time.perf_counter()-generation_started
        result.update(model_id=self.backend.loaded_id, runtime_sha256=self.backend.runtime_sha256,
                      request_id=str(uuid.uuid4()), generation_seconds=elapsed,
                      queue_seconds=generation_started-started, total_seconds=time.perf_counter()-started)
        return result

    def debug(self):
        return {'loaded_id': self.backend.loaded_id if self.backend else None,
                'calls': self.backend.calls if self.backend else 0,
                'warmup_finished': self.warmup_finished.is_set()}


def handler_for(state):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/health':
                self.send_json(200, {'alive': True})
            elif self.path == '/ready':
                self.send_json(200 if state.ready else 503, {'ready': state.ready})
            elif self.path == '/debug':
                self.send_json(200, state.debug())
            else:
                self.send_json(404, {'error': 'not found'})

        def do_POST(self):
            if self.path != '/generate':
                self.send_json(404, {'error': 'not found'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 65536:
                    self.send_json(413, {'error': 'body too large'})
                    return
                self.send_json(200, state.generate(json.loads(self.rfile.read(size))))
            except NotReady as exc:
                self.send_json(503, {'error': str(exc)})
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                self.send_json(422, {'error': str(exc)})
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path('.lab'))
    parser.add_argument('--port', type=int, default=8091)
    parser.add_argument('--warmup-seconds', type=float, default=5)
    parser.add_argument('--engine', choices=['stdlib', 'fastapi'], default='stdlib')
    parser.add_argument('--backend', choices=['stub', 'transformers'], default='stub')
    args = parser.parse_args()
    factory = StubBackend
    if args.backend == 'transformers':
        from backend import TransformersBackend
        factory = TransformersBackend
    if args.engine == 'fastapi':
        import uvicorn
        from api import create_app
        uvicorn.run(create_app(args.workspace, args.warmup_seconds, factory,
                               debug=args.backend=='stub'), host='127.0.0.1', port=args.port)
        return
    state = ServiceState(args.workspace, args.warmup_seconds, factory)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(state))
    print(f'http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
