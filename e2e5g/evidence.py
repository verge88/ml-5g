"""Public application evidence and a small offline authorization contract."""
import base64,json,time
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

def dnn_result(log):
    return dict(registered='Registration accept received' in log,established='PDU Session establishment is successful' in log,
        denied='DNN_NOT_SUPPORTED_OR_NOT_SUBSCRIBED' in log or 'PDU Session Establishment Reject received' in log,
        denial_cause='DNN_NOT_SUPPORTED_OR_NOT_SUBSCRIBED' if 'DNN_NOT_SUPPORTED_OR_NOT_SUBSCRIBED' in log else ('PDU_SESSION_ESTABLISHMENT_REJECT' if 'PDU Session Establishment Reject received' in log else None))

def receipt(row,public,now_ns=None):
    if row is None or row.get('returncode')!=0:return False
    if row.get('interface')!='uesimtun0':return False
    if abs((time.time_ns() if now_ns is None else now_ns)-row['unix_ns'])>3e9:return False
    bundle=row['response'];p=bundle['payload'];request=row['request']
    if p.get('schema')!='e2e-application-receipt-v1' or p['nonce']!=request['nonce'] or p['body_sha256']!=request['body_sha256'] or p['boot_id']!=public['boot_id']:return False
    key=Ed25519PublicKey.from_public_bytes(base64.b64decode(public['public_key']))
    key.verify(base64.b64decode(bundle['signature']),json.dumps(p,sort_keys=True,separators=(',',':')).encode())
    return True

def authorization_contract(event,grant):
    required={'actor','subscriber','context','dnn','slice','operation'}
    if not required<=set(event) or not {'actors','subscriber','context','dnns','slices','operations'}<=set(grant):return 'UNKNOWN'
    ok=(event['actor'] in grant['actors'] and event['subscriber']==grant['subscriber'] and event['context']==grant['context'] and event['dnn'] in grant['dnns'] and event['slice'] in grant['slices'] and event['operation'] in grant['operations'])
    return 'PERMITTED_IN_DECLARED_CONTRACT' if ok else 'VIOLATION'
