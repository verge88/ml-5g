"""Collect causal measurements on a live isolated full core; no predictor here."""
import asyncio,hashlib,json,os,secrets,shutil,subprocess,sys,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from pymongo import MongoClient
from e2e5g import core as base
from e2e5g.configure import generate,subscriber,ROLES
from e2e5g.evidence import receipt
from cu5g import driver
from cu5g.common import POLICY,atomic,native_profile,verified
from fullsuite.native_hook import Sampler
from hybrid.vendor.stage39_kie import FreshnessAnchor,KIERequest,StreamRelationalVerifier,sha256_json
from live5g.capture import observe
from .profiles import configuration,command,case_order
OUT=Path('/out');PLAN=json.loads(Path(__file__).with_name('plan.json').read_text())

async def capture(public):
    replica=int(os.environ['REPLICA']);smoke=os.environ.get('FORECAST_SMOKE')=='1'
    sampler=Sampler(StreamRelationalVerifier(POLICY),FreshnessAnchor(OUT/'controller-anchor.json'))
    sampler.last_issue=1e30;sampler.cycle=40000
    last_signed={s:None for s in POLICY.slots};seen=set();completions=[];requests=[];index=0
    async def sample(c,case,tick,folder):
        nonlocal index
        began=time.monotonic();identifier=index;index+=1
        atomic('/control/probe-command.json',dict(id=identifier,**c))
        raw=await observe();obs=sampler.observe(raw);monitor=obs['ausf']['integrity_monitor']
        for s in POLICY.slots:
            if monitor['states'][s]=='AUTHENTICATED_CURRENT_CU':
                profile=obs[s]['nrf']['body']['nfInstances'][0];envelope=profile['customInfo']['labKIE']['public_bundle'];p=verified(envelope,OUT/f'{s}.pub')
                assert p['request']==asdict(sampler.latest[s]);last_signed[s]=p['response']['issued_at_ms']/1000
                token=(s,p['request']['cycle'])
                if token not in seen:completions.append(dict(slot=s,public_bundle=envelope));seen.add(token)
        if identifier%PLAN['cu_period_ticks']==0:
            sampler.cycle+=1
            for s in POLICY.slots:
                body=raw[s]['nrf']['body'];profiles=body.get('nfInstances',[]) if isinstance(body,dict) else []
                if len(profiles)==1:
                    normalized=native_profile(s,profiles[0])
                    if POLICY.matches(normalized):
                        q=KIERequest.create('native-controller',s,sampler.cycle,POLICY.digest,sha256_json(normalized),sampler.keys[s]);sampler.latest[s]=q;atomic(f'/nf/{s}/request.json',asdict(q));requests.append(dict(id=identifier,slot=s,request=asdict(q)))
        batch=None
        for _ in range(400):
            try:
                candidate=json.loads((OUT/'ue/batch.json').read_text())
                if candidate['command_id']==identifier:batch=candidate;break
            except (FileNotFoundError,ValueError):pass
            await asyncio.sleep(.02)
        if batch is None:raise RuntimeError('test UE batch unavailable; inspect UE logs')
        valid=[]
        for p in batch['probes']:
            if p['returncode']==0:
                assert receipt(p,public,now_ns=batch['unix_ns']);valid.append(p['rtt_ms'])
        summary=dict(signed=len(valid),requested=batch['requested'],median_ms=float(np.median(valid)) if valid else 0.,p90_ms=float(np.percentile(valid,90)) if valid else 0.,std_ms=float(np.std(valid)) if valid else 0.,
            success_fraction=len(valid)/batch['requested'],session_established=batch['session_established'],registered=batch['registered'],tun=batch['tun'],seconds_since_session=batch['seconds_since_session'])
        now=time.time();ages={s:now-last_signed[s] if last_signed[s] is not None else 1e6 for s in POLICY.slots}
        row=dict(id=identifier,trial=case,tick=tick,monotonic_ns=time.monotonic_ns(),unix_ns=time.time_ns(),batch=batch,probe_summary=summary,
            cu=dict(ages=ages,hard_alarm=monitor['hard_alarm'],current_confirmed=monitor['confirmed'],witness_available=monitor['witness_available']),observations=obs)
        with (folder/'observations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        await asyncio.sleep(max(0,PLAN['period_seconds']-(time.monotonic()-began)))
        return row
    warm=OUT/'warmup';warm.mkdir()
    for t in range(PLAN['warmup_ticks']):last=await sample(dict(delay_ms=0.,count=3,restart=False),'warmup',t,warm)
    assert last['probe_summary']['signed']>=2 and max(last['cu']['ages'].values())<=PLAN['maximum_cu_age_seconds']
    cases=['quiet','benign_restart','gradual_restart'] if smoke else PLAN['cases']
    order=case_order(replica,cases);trials=[]
    for name in order:
        folder=OUT/'trials'/name;folder.mkdir(parents=True);config=configuration(replica,name,smoke)
        atomic(folder/'actuation-plan.json',config)
        for t in range(config['ticks']):await sample(command(config,t),name,t,folder)
        trials.append(dict(name=name,rows=config['ticks']))
    atomic(OUT/'cu-completions.json',completions);atomic(OUT/'cu-requests.json',requests);atomic(OUT/'bound-observations.json',sampler.bound_evidence)
    atomic(OUT/'capture.json',dict(replica=replica,smoke=smoke,trials=trials,order=order,source_commit=os.environ['SOURCE_COMMIT'],plan_sha256=hashlib.sha256(Path(__file__).with_name('plan.json').read_bytes()).hexdigest(),
        source='measured UE RTT and signed application responses through live UE/AMF/SMF/UPF',predictor_used=False,actuator_settings_used_as_features=False))

async def run():
    OUT.mkdir(exist_ok=True);agents=[];vm=None;tcp=None
    for p in ['/runtime','/nf','/nfkeys','/private','/witnesskey','/public','/witness','/config','/control']:Path(p).mkdir(exist_ok=True)
    generate();shutil.copytree('/config',OUT/'config',dirs_exist_ok=True);driver.ROOT=Path('/config');driver.OUT=OUT
    for s in POLICY.slots:
        Path(f'/nf/{s}').mkdir(exist_ok=True);Path(f'/nfkeys/{s}.key').write_bytes(secrets.token_bytes(32));(OUT/f'{s}.pub').write_bytes(base.key(Path(f'/private/{s}.key')))
    pub=base.key(Path('/witnesskey/witness.key'));Path('/public/witness.pub').write_bytes(pub);(OUT/'runtime-witness.pub').write_bytes(pub)
    db=MongoClient('mongodb://database',serverSelectionTimeoutMS=1000)
    for _ in range(60):
        try:db.admin.command('ping');break
        except Exception:await asyncio.sleep(.5)
    else:raise RuntimeError('subscriber database unavailable')
    db.open5gs.subscribers.replace_one({'imsi':'999700000000001'},subscriber(),upsert=True)
    for args in [['ip','tuntap','add','name','ogstun','mode','tun'],['ip','addr','add','10.45.0.1/16','dev','ogstun'],['ip','link','set','ogstun','up']]:subprocess.run(args,check=True)
    try:
        driver.start('nrf');await asyncio.sleep(.6)
        for s in ROLES[1:]:driver.start(s)
        for s in POLICY.slots:agents.append(subprocess.Popen([sys.executable,'-m','e2e5g.nf_agent',s],stdout=(OUT/f'{s}-agent.log').open('w'),stderr=subprocess.STDOUT))
        agents.append(subprocess.Popen([sys.executable,'-m','cu5g.witness'],stdout=(OUT/'runtime-witness.log').open('w'),stderr=subprocess.STDOUT))
        await driver.ready();vm,public=await base.vm_start()
        tcp=subprocess.Popen(['tcpdump','--immediate-mode','-B','4096','-i','ogstun','-nn','-U','-w',str(OUT/'ue-user-plane.pcap'),'tcp','port','18081'],stdout=(OUT/'tcpdump.log').open('w'),stderr=subprocess.STDOUT)
        atomic('/control/core-ready.json',dict(ready=True));await capture(public)
        Path('/control/stop-probes').touch()
        for _ in range(100):
            if (OUT/'ue/done.json').exists():break
            await asyncio.sleep(.1)
        else:raise RuntimeError('test UE did not stop before capture drain')
        await asyncio.sleep(1.2)
        atomic(OUT/'complete.json',dict(replica=int(os.environ['REPLICA']),source_commit=os.environ['SOURCE_COMMIT'],roles=ROLES,production=False,semantic_attacks=False,physical_separate_host=False))
    finally:
        Path('/control/done').touch()
        if tcp is not None:
            tcp.send_signal(__import__('signal').SIGINT);tcp.wait(timeout=5)
        for p in agents+[vm]:
            if p is not None:
                p.terminate()
                try:p.wait(timeout=3)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        for s in reversed(list(driver.PROCS)):driver.stop(s)

if __name__=='__main__':asyncio.run(run())
