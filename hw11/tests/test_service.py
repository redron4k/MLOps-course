import asyncio
from pathlib import Path
import aiohttp
from aiohttp import web
import pytest
from server import create_app,env_config
from client import attempt,schedule,summarize


@pytest.fixture
async def service():
    c=env_config();c.update(artifacts=str(Path(__file__).resolve().parents[1]/'artifacts'),warmup=.15,hit_seconds=.005,miss_seconds=.01,chunk_seconds=.025,concurrency=1,queue_limit=1,queue_timeout=.08)
    app=create_app(c)
    runner=web.AppRunner(app);await runner.setup()
    site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
    url=f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
    yield app,url
    await runner.cleanup()


async def test_warmup_readiness_and_version(service):
    app,url=service
    async with aiohttp.ClientSession() as s:
        for path,expected in [('/health',200),('/ready',503),('/startup',503)]:
            async with s.get(url+path) as r:assert r.status==expected
        await asyncio.sleep(.2)
        async with s.get(url+'/ready') as r:
            assert r.status==200;assert (await r.json())['version']=='A'


async def test_caller_owned_dialogue_isolation(service):
    app,url=service;await asyncio.sleep(.2)
    async with aiohttp.ClientSession() as s:
        for marker,other in [('ALPHA','BETA'),('BETA','ALPHA')]:
            async with s.post(url+'/generate',json={'messages':[{'role':'user','content':marker}],'max_chunks':1}) as r:
                body=await r.text();assert marker in body and other not in body and '"done": true' in body


async def test_bound_queue_and_terminal_accounting(service):
    app,url=service;await asyncio.sleep(.2)
    async with aiohttp.ClientSession() as s:
        async def request(n):
            async with s.post(url+'/generate',json={'prompt':str(n),'max_chunks':20}) as r:
                await r.read();return r.status
        statuses=await asyncio.gather(*(request(i) for i in range(6)))
    assert statuses.count(200)==1 and statuses.count(429)==5
    assert app['state']['running']==app['state']['waiting']==0
    assert app['metrics']['requests'].labels('completed')._value.get()==1
    assert app['metrics']['requests'].labels('rejected')._value.get()==5


async def test_aborted_stream_releases_capacity(service):
    app,url=service;await asyncio.sleep(.2)
    async with aiohttp.ClientSession() as s:
        async with s.post(url+'/generate',json={'prompt':'abort','max_chunks':40}) as r:
            await r.content.readline();r.close()
        await asyncio.sleep(.1)
    assert app['state']['running']==0
    assert app['metrics']['requests'].labels('interrupted')._value.get()==1


async def test_bad_artifact_never_ready(tmp_path):
    from shutil import copytree
    copytree(Path(__file__).resolve().parents[1]/'artifacts',tmp_path/'artifacts')
    (tmp_path/'artifacts/model-A.json').write_text('corrupted')
    c=env_config();c.update(artifacts=str(tmp_path/'artifacts'),warmup=.01)
    app=create_app(c);runner=web.AppRunner(app);await runner.setup()
    await asyncio.sleep(.03)
    assert not app['state']['ready'] and 'Artifact changed' in app['state']['error']
    await runner.cleanup()


def test_open_loop_schedule_identical_and_skew_exact():
    a=schedule(120,8,True);b=schedule(120,8,True)
    assert a==b and len(a)==960
    assert sum(e['client']==0 for e in a)==768
    assert len({e['client'] for e in a})==4


async def test_broken_probes_reproduce_original_defect():
    c=env_config();c.update(artifacts=str(Path(__file__).resolve().parents[1]/'artifacts'),warmup=.2,broken=True)
    app=create_app(c);runner=web.AppRunner(app);await runner.setup()
    site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
    url=f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url+'/ready') as r:
                assert r.status==200 and not (await r.json())['ready']
            async with s.post(url+'/generate',json={'prompt':'cold','max_chunks':1}) as r:
                assert r.status==503
    finally:await runner.cleanup()


async def test_plain_text_http_failure_is_not_a_connection_error():
    app=web.Application()
    async def fail(request):return web.Response(status=500,text='upstream unavailable')
    app.router.add_post('/generate',fail)
    runner=web.AppRunner(app);await runner.setup()
    site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
    url=f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
    try:
        async with aiohttp.ClientSession() as session:
            import time
            row=await attempt(session,url,{'id':0,'at':0,'client':0,'payload':{'prompt':'fail','max_chunks':1}},time.perf_counter(),'test-500')
        assert row['status']==500 and row['outcome']=='http_error'
        assert row['error']=='upstream unavailable'
        assert summarize([row])['error_fraction']==1
    finally:await runner.cleanup()


async def test_drain_unready_but_serves_stale_network_connections(service):
    app,url=service;app['config']['drain_seconds']=.1
    await asyncio.sleep(.2)
    async with aiohttp.ClientSession() as session:
        draining=asyncio.create_task(session.get(url+'/drain'))
        await asyncio.sleep(.02)
        async with session.get(url+'/ready') as response:assert response.status==503
        async with session.post(url+'/generate',json={'prompt':'stale proxy connection','max_chunks':1}) as response:
            assert response.status==200 and '"done": true' in await response.text()
        response=await draining;response.release()
    assert app['state']['draining']
