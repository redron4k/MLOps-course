"""Bounded, observable streaming service, a lifecycle adapter for HW8-10 backend."""
import asyncio
from contextlib import suppress
import json
import os
from pathlib import Path
import time
import uuid
from aiohttp import web
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from backend import SimulatedBackend

BUCKETS = (.01, .025, .05, .1, .2, .4, .8, 1.5, 3, 5, 10, 20, 40, 80)


def env_config():
    return dict(version=os.getenv('MODEL_VERSION', 'A'), warmup=float(os.getenv('WARMUP_SECONDS', '20')),
        broken=os.getenv('BROKEN_PROBES', 'false') == 'true', never=os.getenv('NEVER_READY', 'false') == 'true',
        concurrency=int(os.getenv('CONCURRENCY', '2')), queue_limit=int(os.getenv('QUEUE_LIMIT', '24')),
        queue_timeout=float(os.getenv('QUEUE_TIMEOUT', '15')), cache_capacity=int(os.getenv('CACHE_CAPACITY', '16')),
        hit_seconds=float(os.getenv('HIT_SECONDS', '.04')), miss_seconds=float(os.getenv('MISS_SECONDS', '.25')),
        chunk_seconds=float(os.getenv('CHUNK_SECONDS', '.08')), pod=os.getenv('POD_NAME', 'local'),
        artifacts=os.getenv('ARTIFACT_DIR', '/app/artifacts'))


def create_app(config=None):
    c = config or env_config()
    app = web.Application(client_max_size=65536)
    registry = CollectorRegistry()
    metrics = {}
    for name in ('ttfc', 'duration', 'queue', 'inter_chunk'):
        metrics[name] = Histogram('inference_'+name+'_seconds', name+' seconds; successful streams only', buckets=BUCKETS, registry=registry)
    for name in ('running', 'waiting', 'ready'):
        metrics[name] = Gauge('inference_'+name, name, registry=registry)
    metrics['requests'] = Counter('inference_requests', 'Terminal request outcomes', ['outcome'], registry=registry)
    metrics['cache'] = Counter('inference_cache', 'Simulated cache access', ['result'], registry=registry)
    for label in ('completed', 'rejected', 'interrupted', 'invalid', 'error'):
        metrics['requests'].labels(label)
    state = dict(ready=False, draining=False, running=0, waiting=0, error=None, backend=None)
    sem = asyncio.Semaphore(c['concurrency'])
    app['state'], app['metrics'], app['config'] = state, metrics, c

    def meta(rid=None):
        b = state['backend']
        return dict(request_id=rid, pod=c['pod'], model_id=b.loaded_id if b else None,
                    version=b.loaded_id if b else None, backend_mode='simulated',
                    artifact_sha256=b.manifest['model_sha256'] if b else None)

    def log(event, **kw):
        print(json.dumps({'event':event, 'unix':time.time(), **meta(), **kw}, ensure_ascii=False), flush=True)

    async def load():
        try:
            root = Path(c['artifacts'])
            manifest = json.loads((root/f"manifest-{c['version']}.json").read_text())
            state['backend'] = SimulatedBackend(manifest, root, c)
            state['backend'].warmup()
            await asyncio.sleep(c['warmup'])
            if c['never']:
                raise ValueError('injected unready release')
            state['ready'] = True
            metrics['ready'].set(1)
            log('warmup_complete')
        except Exception as exc:
            state['error'] = str(exc)
            log('load_failed', error=str(exc))

    async def lifecycle(app):
        task = asyncio.create_task(load())
        yield
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    app.cleanup_ctx.append(lifecycle)

    async def probe(request):
        path = request.path
        okay = path == '/health' or (not state['draining'] and (c['broken'] or state['ready']))
        return web.json_response(dict(**meta(), ready=state['ready'], draining=state['draining'], error=state['error']), status=200 if okay else 503)

    async def drain(request):
        # preStop marks unready, allows kube-proxy propagation, then SIGTERM drains HTTP.
        state['draining'] = True
        metrics['ready'].set(0)
        log('draining', running=state['running'])
        await asyncio.sleep(5)
        return web.json_response({'draining': True})

    async def debug(request):
        return web.json_response(dict(**meta(), **{k: v for k, v in state.items() if k != 'backend'}, config=c))

    async def reset(request):
        if state['running'] or state['waiting']:
            return web.json_response({'error': 'busy'}, status=409)
        state['backend'].cache.clear()
        return web.json_response({'cache': 'empty'})

    async def expose(request):
        return web.Response(body=generate_latest(registry), content_type='text/plain')

    async def generate(request):
        started = time.perf_counter()
        rid = request.headers.get('X-Request-ID', str(uuid.uuid4()))[:128]
        session = None
        def reject(status, reason, outcome='rejected'):
            metrics['requests'].labels(outcome).inc()
            log('terminal', request_id=rid, outcome=outcome, reason=reason, status=status)
            return web.json_response(dict(**meta(rid), error=reason), status=status)
        try:
            body = await request.json()
            history = body.get('messages') or [{'role': 'user', 'content': body.get('prompt')}]
            if not isinstance(history, list) or not 1 <= len(history) <= 32:
                raise ValueError('1..32 messages required')
            if any(m.get('role') not in ('system', 'user', 'assistant') or not isinstance(m.get('content'), str) or not m['content'].strip() for m in history):
                raise ValueError('invalid messages')
            if sum(len(m['content']) for m in history) > 8192:
                raise ValueError('history too long')
            chunks = body.get('max_chunks', body.get('max_new_tokens', 4))
            if type(chunks) is not int or not 1 <= chunks <= 128:
                raise ValueError('max_chunks must be 1..128')
            session = str(body.get('session_id', ''))[:128]
        except (ValueError, TypeError, AttributeError, KeyError):
            return reject(422, 'invalid input', 'invalid')
        log('received', request_id=rid, session_id=session, peer=request.remote, warm=state['ready'],
            input_chars=sum(len(m['content']) for m in history), max_chunks=chunks)
        if not state['ready'] or state['draining']:
            return reject(503, 'backend not warm or draining')
        if state['running'] + state['waiting'] >= c['concurrency'] + c['queue_limit']:
            return reject(429, 'queue full')
        state['waiting'] += 1
        metrics['waiting'].inc()
        acquired = False
        try:
            await asyncio.wait_for(sem.acquire(), c['queue_timeout'])
            acquired = True
        except asyncio.TimeoutError:
            return reject(429, 'queue timeout')
        finally:
            state['waiting'] -= 1
            metrics['waiting'].dec()
        state['running'] += 1
        metrics['running'].inc()
        queue = time.perf_counter()-started
        response = web.StreamResponse(headers={'Content-Type': 'application/x-ndjson', 'X-Pod-Name': c['pod'], 'X-Model-Version': state['backend'].loaded_id, 'X-Request-ID': rid})
        outcome = 'interrupted'
        first = previous = None
        intervals = []
        try:
            await response.prepare(request)
            hit = await state['backend'].prepare(history)
            metrics['cache'].labels('hit' if hit else 'miss').inc()
            # Echo caller-owned history as a deterministic isolation/recovery oracle.
            result = state['backend'].generate('|'.join(m['content'] for m in history), chunks)
            for index in range(chunks):
                if index:
                    await asyncio.sleep(c['chunk_seconds'])
                now = time.perf_counter()
                if first is None:
                    first = now
                if previous is not None:
                    intervals.append(now-previous)
                previous = now
                payload = dict(**meta(rid), index=index, content=result['answer'] if index == 0 else f'chunk-{index}', cache_hit=hit, queue_seconds=queue)
                await response.write((json.dumps(payload, ensure_ascii=False)+'\n').encode())
            await response.write((json.dumps(dict(**meta(rid), done=True, chunks=chunks))+'\n').encode())
            await response.write_eof()
            outcome = 'completed'
            metrics['ttfc'].observe(first-started)
            metrics['duration'].observe(time.perf_counter()-started)
            metrics['queue'].observe(queue)
            for interval in intervals:
                metrics['inter_chunk'].observe(interval)
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            log('stream_interrupted', request_id=rid)
        except Exception as exc:
            outcome = 'error'
            log('stream_error', request_id=rid, error=str(exc))
        finally:
            metrics['requests'].labels(outcome).inc()
            metrics['running'].dec()
            state['running'] -= 1
            if acquired:
                sem.release()
            log('terminal', request_id=rid, session_id=session, outcome=outcome, queue_seconds=queue, duration_seconds=time.perf_counter()-started)
        return response

    for path in ('/health', '/ready', '/startup'):
        app.router.add_get(path, probe)
    app.router.add_get('/drain', drain)
    app.router.add_get('/debug', debug)
    app.router.add_post('/reset-cache', reset)
    app.router.add_get('/metrics', expose)
    app.router.add_post('/generate', generate)
    return app


if __name__ == '__main__':
    web.run_app(create_app(), host='0.0.0.0', port=8000, shutdown_timeout=40, access_log=None)
