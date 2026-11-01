"""Лёгкий HTTP-прототип с дефектами. Только localhost, STUB ONLY.

Для итогового проекта перенесите исправленный контракт в FastAPI и подключите
загрузчик ДЗ 6+7. Прототип нужен для опытов без установки зависимостей и весов.
"""
import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class StubBackend:
    def __init__(self,manifest):
        self.loaded_id=manifest['model_id'];self.calls=0;self.runtime_sha256=manifest['runtime_sha256']

    def generate(self,prompt,limit):
        self.calls+=1;time.sleep(.01)
        return {'answer':f'STUB ONLY: {self.loaded_id}', 'output_tokens':1}


class ServiceState:
    def __init__(self,root,warmup_seconds):
        self.root=root;self.ready=True
        manifest=json.loads((root/'accepted.json').read_text())
        self.backend=StubBackend(manifest)
        self.warmup_finished=threading.Event()
        def warmup():
            time.sleep(warmup_seconds);self.warmup_finished.set()
        threading.Thread(target=warmup,daemon=True).start()

    def generate(self,request):
        started=time.perf_counter()
        result=self.backend.generate(request.get('prompt',''),request.get('max_new_tokens',32))
        advertised=json.loads((self.root/'accepted.json').read_text())
        result.update(model_id=advertised['model_id'],runtime_sha256=advertised['runtime_sha256'],
                      request_id=str(uuid.uuid4()),generation_seconds=time.perf_counter()-started)
        return result


def handler_for(state):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self,status,payload):
            data=json.dumps(payload,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)

        def do_GET(self):
            if self.path=='/health':self.send_json(200,{'alive':True,'backend':'STUB ONLY'})
            elif self.path=='/ready':self.send_json(200 if state.ready else 503,{'ready':state.ready})
            elif self.path=='/debug':
                self.send_json(200,{'loaded_id':state.backend.loaded_id,'calls':state.backend.calls,'warmup_finished':state.warmup_finished.is_set()})
            else:self.send_json(404,{'error':'not found'})

        def do_POST(self):
            if self.path!='/generate':self.send_json(404,{'error':'not found'});return
            size=int(self.headers.get('Content-Length','0'))
            if size<0 or size>65536:self.send_json(413,{'error':'body too large'});return
            try:
                payload=json.loads(self.rfile.read(size))
                self.send_json(200,state.generate(payload))
            except (ValueError,TypeError,AttributeError,KeyError) as e:
                self.send_json(400,{'error':str(e)})
    return Handler


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path('.lab'))
    p.add_argument('--port',type=int,default=8091);p.add_argument('--warmup-seconds',type=float,default=5)
    args=p.parse_args();state=ServiceState(args.workspace,args.warmup_seconds)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(state))
    print(f'STUB ONLY: http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
