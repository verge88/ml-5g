"""Verify measurement origin and recompute pilot summaries before using data."""
import argparse,base64,hashlib,json,re
from pathlib import Path
import numpy as np
from cryptography.hazmat.primitives import serialization,hashes
from cryptography.hazmat.primitives.asymmetric import ec
from .data import PLAN
from .evidence import verify_probe


def canonical(p):return json.dumps(p,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


def run(folder):
    folder=Path(folder);capture=json.loads((folder/'diagnostic-capture.json').read_text())
    complete=json.loads((folder/'complete.json').read_text())
    assert len(complete['roles'])==10 and len(list((folder/'config').glob('*.yaml')))==10
    assert capture['source_commit']==complete['source_commit']
    assert capture['plan_sha256']==hashlib.sha256(Path(__file__).with_name('plan.json').read_bytes()).hexdigest()
    pub=json.loads((folder/'guest-public.json').read_text());serial=(folder/'guest-serial.log').read_text(errors='replace')
    assert pub['boot_id']!=pub['host_boot_id'] and pub['public_key'] in serial and pub['boot_id'] in serial
    packet_nonces=set(re.findall(rb'"nonce"\s*:\s*"([0-9a-f]{32})"',(folder/'ue-user-plane.pcap').read_bytes()))
    assert '0 packets dropped by kernel' in (folder/'tcpdump.log').read_text()
    rows=[json.loads(s) for s in (folder/'warmup/observations.jsonl').read_text().splitlines()]
    for trial in capture['trials']:rows += [json.loads(s) for s in (folder/'trials'/trial['name']/'observations.jsonl').read_text().splitlines()]
    assert len(rows)==PLAN['warmup']+8*PLAN['ticks'] and [r['id'] for r in rows]==list(range(len(rows)))
    counts=dict(rows=len(rows),end_signatures=0,bypass_signatures=0,pcap_backed_end_signatures=0,nf_agent_signatures=0,native_received_signatures=0,runtime_witnesses=0)
    nonces=set()
    for row in rows:
        assert row['id']==row['batch']['command_id']
        for route,probes,key in [('uesimtun0',row['batch']['probes'],'end'),('core-bypass',row['bypass'],'bypass')]:
            values=[]
            for p in probes:
                nonce=p['request']['nonce'];assert nonce not in nonces;nonces.add(nonce)
                if p['returncode']==0:
                    assert verify_probe(p,pub,route,row['batch']['unix_ns'] if key=='end' else row['unix_ns'])
                    assert p['rtt_ms']>=0;values.append(p['rtt_ms']);counts[key+'_signatures']+=1
                    if key=='end':assert nonce.encode() in packet_nonces;counts['pcap_backed_end_signatures']+=1
            assert len(values)==row['measured'][key+'_signed']
            expected=float(np.median(values)) if values else None
            assert row['measured'][key+'_ms']==expected
    keys={s:serialization.load_pem_public_key((folder/f'{s}.pub').read_bytes()) for s in ['ausf','nssf']}
    def verify(bundle,key):
        key.verify(base64.b64decode(bundle['signature']),canonical(bundle['payload']),ec.ECDSA(hashes.SHA256()))
        return bundle['payload']
    for slot,key in keys.items():
        last=-1
        for line in (folder/f'nf-public-{slot}.jsonl').read_text().splitlines():
            p=verify(json.loads(line),key)
            assert p['mac_checked_before_signing'] and p['request']['subject_id']==slot and p['request']['cycle']>last
            assert p['request']['nonce']==p['response']['request_nonce'];last=p['request']['cycle'];counts['nf_agent_signatures']+=1
    for c in json.loads((folder/'cu-completions.json').read_text()):verify(c['public_bundle'],keys[c['slot']]);counts['native_received_signatures']+=1
    bounds=json.loads((folder/'bound-observations.json').read_text());assert len(bounds)==len(rows)
    witness=serialization.load_pem_public_key((folder/'runtime-witness.pub').read_bytes())
    for row,b in zip(rows,bounds):
        assert row['cu']['hard_alarm']==b['hard_alarm']
        if b['witness']:
            p=verify(b['witness'],witness);assert p['challenge']==b['challenge'] and abs(p['issued_ns']-b['observed_unix_ns'])<5e9
            counts['runtime_witnesses']+=1
    result=dict(verified=True,replica=capture['replica'],source_commit=capture['source_commit'],guest_boot_id=pub['boot_id'],counts=counts,
        same_signer_for_both_paths=True,physical_separate_host=False,collector_clock_trusted=True)
    (folder/'audit.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();print(json.dumps(run(a.folder)))
