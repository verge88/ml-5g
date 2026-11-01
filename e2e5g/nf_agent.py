"""Retain live KIE MAC verification and add publicly verifiable NF transcripts."""
from dataclasses import asdict
from pathlib import Path
import json,sys,time
from hybrid.vendor.stage39_kie import FreshnessAnchor,KIERequest,KIEResponse,sha256_json
from cu5g.common import POLICY,atomic,runtime,sign

def run(slot):
    folder=Path('/nf')/slot;key=Path(f'/nfkeys/{slot}.key').read_bytes();last=None
    anchor=FreshnessAnchor(folder/'request-anchor.json')
    while not Path('/control/done').exists():
        try:
            raw=(folder/'request.json').read_text()
            if raw!=last:
                req=KIERequest(**json.loads(raw))
                if req.subject_id!=slot or req.controller_id!='native-controller': raise ValueError('scope')
                rec=runtime(slot);digest=sha256_json(POLICY.expected_profile(slot,rec['nf_instance_id']))
                req.verify(key,anchor,POLICY.digest,digest)
                report=dict(slot=slot,policy_sha256=POLICY.digest,telemetry_digest=digest,runtime=rec,report_source='NF-side agent; context export remains NF-produced')
                response=asdict(KIEResponse.create(req,report,key))
                public=sign(dict(schema='public-live-kie-transcript-v1',request=asdict(req),response=response,mac_checked_before_signing=True),f'/private/{slot}.key')
                atomic(folder/'custom_info.json',dict(labKIE=dict(response=response,public_bundle=public)))
                with Path(f'/out/nf-public-{slot}.jsonl').open('a') as stream:stream.write(json.dumps(public)+'\n')
                last=raw
        except Exception: pass
        time.sleep(.05)

if __name__=='__main__':run(sys.argv[1])
