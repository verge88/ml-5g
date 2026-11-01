"""Independent-kernel application receipt witness, never an NF identity oracle."""
import base64, hashlib, json, os, platform, secrets, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

KEY = Ed25519PrivateKey.generate()
PUBLIC = base64.b64encode(KEY.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
BOOT = open('/proc/sys/kernel/random/boot_id').read().strip()
SEEN = set()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps(dict(public_key=PUBLIC,boot_id=BOOT,kernel=platform.release(),schema='e2e-guest-public-v1')).encode())
    def do_POST(self):
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<4096: raise ValueError('size')
            request=json.loads(self.rfile.read(n)); nonce=request['nonce']
            if not isinstance(nonce,str) or len(nonce)!=32 or nonce in SEEN: raise ValueError('nonce')
            if set(request)!={'nonce','body_sha256'} or len(request['body_sha256'])!=64: raise ValueError('scope')
            SEEN.add(nonce)
            payload=dict(schema='e2e-application-receipt-v1',nonce=nonce,body_sha256=request['body_sha256'],boot_id=BOOT,kernel=platform.release(),received_ns=time.time_ns())
            canonical=json.dumps(payload,sort_keys=True,separators=(',',':')).encode()
            result=dict(payload=payload,signature=base64.b64encode(KEY.sign(canonical)).decode())
            self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(result).encode())
        except Exception:
            self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"invalid receipt request"}')
    def log_message(self,*args): pass

print(json.dumps(dict(witness_public_key=PUBLIC,guest_boot_id=BOOT,guest_kernel=platform.release())),flush=True)
HTTPServer(('0.0.0.0',8081),Handler).serve_forever()
