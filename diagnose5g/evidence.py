import base64,json
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


def verify_probe(row,public,route,now_ns):
    if row.get('returncode')!=0:return False
    if route not in ['uesimtun0','core-bypass'] or row.get('interface')!=route:return False
    if abs(now_ns-row['unix_ns'])>3e9:return False
    p=row['response']['payload'];q=row['request']
    if p.get('schema')!='e2e-application-receipt-v1' or p['boot_id']!=public['boot_id'] or p['nonce']!=q['nonce'] or p['body_sha256']!=q['body_sha256']:return False
    key=Ed25519PublicKey.from_public_bytes(base64.b64decode(public['public_key']))
    key.verify(base64.b64decode(row['response']['signature']),json.dumps(p,sort_keys=True,separators=(',',':')).encode())
    return True
