"""Independent guest receipts with bounded local-only management service."""
import json,threading,time
from http.server import BaseHTTPRequestHandler,HTTPServer,ThreadingHTTPServer
from e2e5g_guest import Handler as ReceiptHandler,PUBLIC,BOOT

WAIT_MS=0.
LOCK=threading.Lock()


class Application(ReceiptHandler):
    def do_POST(self):
        with LOCK:wait=WAIT_MS
        time.sleep(wait/1000)
        super().do_POST()


class Management(BaseHTTPRequestHandler):
    def do_POST(self):
        global WAIT_MS
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<128:raise ValueError('size')
            data=json.loads(self.rfile.read(n))
            if set(data)!={'wait_ms'}:raise ValueError('schema')
            value=float(data['wait_ms'])
            if not 0<=value<=150:raise ValueError('bound')
            with LOCK:WAIT_MS=value
            self.send_response(200);self.end_headers();self.wfile.write(b'{"applied":true}')
        except Exception:
            self.send_response(400);self.end_headers();self.wfile.write(b'{"error":"invalid bounded test control"}')
    def log_message(self,*args):pass


if __name__=='__main__':
    # Only guest port8081 is reachable through the user-data forwarding path.
    # Guest8082 is forwarded solely to core loopback127.0.0.1:18082.
    threading.Thread(target=HTTPServer(('0.0.0.0',8082),Management).serve_forever,daemon=True).start()
    ThreadingHTTPServer(('0.0.0.0',8081),Application).serve_forever()
