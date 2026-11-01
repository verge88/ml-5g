"""Longer independent benign lab captures with registered local fault variety."""
import asyncio,os,json,hashlib
from pathlib import Path
from live5g import capture
from fullsuite.native_hook import run

def settings(replica):
 if replica not in list(range(801,807))+[809,810,811,812,813]:raise ValueError('unregistered laboratory deployment')
 periods={801:.3,802:.45,803:.6,804:.3,805:.45,806:.6,809:.35,810:.55,811:.3,812:.45,813:.6}
 training=[['ausf','stop'],['nssf','stop'],['ausf','pause'],['nssf','pause']];shift=replica%4;training=training[shift:]+training[:shift]
 return dict(period_seconds=periods[replica],fit_ticks=600,calibration_ticks=150,normal_test_ticks=300,baseline_ticks=16,recovery_ticks=16,
  fault_ticks=([8,12,20][(replica-801)%3] if 801<=replica<=806 else 12),training_faults=training,
  test_faults=[['ausf','stop'],['nssf','stop'],['ausf','pause'],['nssf','pause']])

async def main():
 replica=int(os.environ['REPLICA']);active={**capture.P,**settings(replica)};capture.P.clear();capture.P.update(active)
 raw=json.dumps(active,sort_keys=True,indent=2).encode();Path('/out/broad-collection-settings.json').write_bytes(raw)
 await run()
 p=Path('/out/native/capture-manifest.json');m=json.loads(p.read_text());m['active_capture_settings_sha256']=hashlib.sha256(raw).hexdigest();p.write_text(json.dumps(m,indent=2))
if __name__=='__main__':asyncio.run(main())
