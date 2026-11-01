"""FastAPI lifecycle owns a single backend; imports never load model weights."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from service import NotReady, ServiceState, StubBackend


def create_app(root, warmup_seconds=0, backend_factory=StubBackend, debug=False):
    @asynccontextmanager
    async def lifespan(app):
        app.state.serving = ServiceState(root, warmup_seconds, backend_factory)
        yield
        app.state.serving.loader.join(timeout=1)

    app = FastAPI(lifespan=lifespan)

    @app.get('/health')
    def health():
        return {'alive': True}

    @app.get('/ready')
    def ready():
        state = app.state.serving
        return JSONResponse(status_code=200 if state.ready else 503, content={'ready': state.ready})

    @app.post('/generate')
    async def generate(request: Request):
        # Reject oversized bodies while reading rather than buffering unbounded input.
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > 65536:
                raise HTTPException(413, 'body too large')
            chunks.append(chunk)
        import json
        from starlette.concurrency import run_in_threadpool
        try:
            payload = json.loads(b''.join(chunks))
            return await run_in_threadpool(app.state.serving.generate, payload)
        except NotReady as exc:
            raise HTTPException(503, str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    if debug:
        @app.get('/debug')
        def inspect():
            return app.state.serving.debug()
    return app
