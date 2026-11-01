"""Full private core, live bounded-budget CU acquisition and guest receipts."""
import asyncio,base64,hashlib,json,os,platform,secrets,signal,subprocess,sys,time,urllib.request
from dataclasses import asdict
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec,ed25519
from pymongo import MongoClient
from cu5g import driver
from cu5g.common import POLICY,atomic,native_profile,verified
from hybrid.vendor.stage39_kie import FreshnessAnchor,KIERequest,StreamRelationalVerifier,sha256_json
from fullsuite.native_hook import Sampler
from live5g.capture import observe
from .configure import generate,subscriber,ROLES
from .schedule import Scheduler,SLOTS
from .evidence import receipt,dnn_result

OUT=Path('/out');PLAN=json.loads(Path(__file__).with_name('plan.json').read_text())

def latest(public):
    try:
        r=json.loads((OUT/'ue/latest.json').read_text());return r,receipt(r,public)
    except Exception:return None,False

async def wait_app(public,timeout=75):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        r,ok=latest(public)
        if ok:return r
        await asyncio.sleep(.25)
    raise RuntimeError('UE registration/session/application receipt did not converge')

def key(path):
    k=ec.generate_private_key(ec.SECP256R1())
    path.write_bytes(k.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()));os.chmod(path,0o600)
    return k.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo)

async def vm_start():
    choices=[('tcg,thread=multi','max')]
    if os.path.exists('/dev/kvm') and os.access('/dev/kvm',os.W_OK):choices.insert(0,('kvm','host'))
    public=None
    for acceleration,cpu in choices:
        logpath=OUT/f"guest-{acceleration.split(',')[0]}-serial.log"
        cmd=['qemu-system-x86_64','-accel',acceleration,'-cpu',cpu,'-m','768','-smp','1','-nographic','-no-reboot','-kernel','/vm/vmlinuz-virt','-initrd','/vm/rootfs.cpio.gz','-append','console=ttyS0,115200 rdinit=/init loglevel=7 earlyprintk=serial,ttyS0,115200 nokaslr','-object','rng-random,filename=/dev/urandom,id=rng0','-device','virtio-rng-pci,rng=rng0','-nic','user,model=virtio-net-pci,restrict=on,hostfwd=tcp:10.45.0.1:18081-:8081']
        log=logpath.open('w');p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        for _ in range(100):
            if p.poll() is not None:break
            try:
                with urllib.request.urlopen('http://10.45.0.1:18081/public',timeout=.3) as r:candidate=json.load(r)
                serial=logpath.read_text(errors='replace')
                if candidate['public_key'] in serial and candidate['boot_id'] in serial:public=candidate;break
            except Exception:pass
            await asyncio.sleep(.5)
        if public is not None:
            (OUT/'guest-serial.log').write_text(logpath.read_text(errors='replace'));break
        p.terminate()
        try:p.wait(timeout=3)
        except subprocess.TimeoutExpired:p.kill();p.wait()
        log.close()
    if public is None:raise RuntimeError('guest receipt witness did not boot; see saved KVM/TCG serial diagnostics')
    host_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert public['boot_id']!=host_boot
    atomic(OUT/'guest-public.json',dict(**public,host_boot_id=host_boot,host_kernel=platform.release(),acceleration=acceleration,source='key pinned against guest serial, not NF export',kernel_sha256=hashlib.sha256(Path('/vm/vmlinuz-virt').read_bytes()).hexdigest(),initrd_sha256=hashlib.sha256(Path('/vm/rootfs.cpio.gz').read_bytes()).hexdigest()))
    return p,public

async def subscription_controls(public):
    # Ordinary UE subscription conformance, never forged SBI traffic.
    Path('/control/ue-dnn').write_text('not-subscribed');Path('/control/ue-restart').write_text(secrets.token_hex(8))
    result=None
    for _ in range(120):
        row,_=latest(public)
        if row and row.get('dnn')=='not-subscribed':
            path=OUT/f"ue/ue-{row['attempt']}.log"
            if path.exists():
                state=dnn_result(path.read_text(errors='replace'))
                if state['denied']:
                    result=dict(**state,dnn='not-subscribed',attempt=row['attempt'],application_probe_returncode=row['returncode'],source='ordinary registered UE, DNN absent from its test subscription');break
        await asyncio.sleep(.25)
    if result is None:raise RuntimeError('subscription conformance did not produce an explicit DNN denial')
    assert result['registered'] and not result['established']
    Path('/control/ue-dnn').write_text('internet');Path('/control/ue-restart').write_text(secrets.token_hex(8))
    await asyncio.sleep(1.1);restored=await wait_app(public)
    assert restored['dnn']=='internet'
    atomic(OUT/'live-subscription-controls.json',dict(denial=result,restored_application=restored,scope='live standard subscription denial and restored permitted service; not semantic attack detection'))

async def collect(mode,order,replica,public,models):
    folder=OUT/mode;folder.mkdir(exist_ok=True);rows=[];events=[];requests=[];completions=[]
    sampler=Sampler(StreamRelationalVerifier(POLICY),FreshnessAnchor(folder/'controller-anchor.json'))
    sampler.cycle=10000*(order+1);sampler.last_issue=1e30
    scheduler=Scheduler(mode,models,PLAN['max_cu_age_ticks']);tick=0
    last_auth={s:None for s in SLOTS};done=set()
    async def sample(stage,count,episode=None):
        nonlocal tick
        for _ in range(count):
            began=time.monotonic();raw=await observe();selected,allocation=scheduler.choose(tick,raw)
            # Assess already issued requests before beginning the next acquisition.
            obs=sampler.observe(raw);m=obs['ausf']['integrity_monitor']
            for s in SLOTS:
                if m['states'][s]=='AUTHENTICATED_CURRENT_CU':
                    last_auth[s]=tick
                    req=sampler.latest[s];token=(s,req.cycle)
                    profiles=obs[s]['nrf']['body']['nfInstances']
                    envelope=profiles[0]['customInfo']['labKIE']['public_bundle']
                    p=verified(envelope,OUT/f'{s}.pub')
                    assert p['request']==asdict(req) and p['response']==profiles[0]['customInfo']['labKIE']['response']
                    if token not in done:
                        completions.append(dict(slot=s,cycle=req.cycle,tick=tick,observed_ns=time.time_ns(),public_bundle=envelope));done.add(token)
            app,app_ok=latest(public)
            cue={s:last_auth[s] is not None and tick-last_auth[s]>PLAN['max_cu_age_ticks'] for s in SLOTS}
            issue=None;profiles=raw[selected]['nrf']['body'];profiles=profiles.get('nfInstances',[]) if isinstance(profiles,dict) else []
            if len(profiles)==1:
                normalized=native_profile(selected,profiles[0])
                if POLICY.matches(normalized):
                    sampler.cycle+=1;req=KIERequest.create('native-controller',selected,sampler.cycle,POLICY.digest,sha256_json(normalized),sampler.keys[selected])
                    sampler.latest[selected]=req;atomic(f'/nf/{selected}/request.json',asdict(req));issue=dict(slot=selected,tick=tick,issued_ns=time.time_ns(),request=asdict(req));requests.append(issue)
            row=dict(tick=tick,monotonic_ns=time.monotonic_ns(),unix_ns=time.time_ns(),stage=stage,episode=episode,observations=obs,selected=selected,allocation=allocation,cu_request=issue,cu_unavailable=cue,application_ok=app_ok,application=app,
                received_bytes=sum(o[k]['received_bytes'] for o in obs.values() for k in ['direct','nrf']),collection_elapsed_ms=(time.monotonic()-began)*1000)
            rows.append(row)
            with (folder/'observations.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            tick+=1;await asyncio.sleep(max(0,PLAN['period_seconds']-(time.monotonic()-began)))
    await wait_app(public)
    await sample('test_normal',PLAN['normal_ticks'])
    faults=list(PLAN['local_faults']);shift=(replica-901)%len(faults);faults=faults[shift:]+faults[:shift]
    for i,s in enumerate(faults):
        await wait_app(public);ep=f'{mode}-{s}'
        await sample('test_baseline',PLAN['baseline_ticks'],ep)
        pid=driver.PROCS[s].pid;os.kill(pid,signal.SIGSTOP)
        start_tick,start_ns=tick,time.monotonic_ns()
        # A new benign registration/session is needed to exercise AUSF and SMF.
        if s in ['ausf','smf']:(Path('/control')/'ue-restart').write_text(secrets.token_hex(8))
        try:await sample('test_fault',PLAN['fault_ticks'],ep)
        finally:os.kill(pid,signal.SIGCONT)
        events.append(dict(episode=ep,slot=s,kind='authorized_local_pause',split='test',start_tick=start_tick,end_tick=tick,start_ns=start_ns,end_ns=time.monotonic_ns(),ue_restart=s in ['ausf','smf']))
        if s in ['ausf','smf']:(Path('/control')/'ue-restart').write_text(secrets.token_hex(8))
        await sample('test_recovery',PLAN['recovery_ticks'],ep);await wait_app(public)
    atomic(folder/'events.json',events);atomic(folder/'cu-requests.json',requests);atomic(folder/'cu-completions.json',completions)
    atomic(folder/'bound-observations.json',sampler.bound_evidence)
    atomic(folder/'manifest.json',dict(replica=replica,mode=mode,order=order,rows=len(rows),cu_requests=len(requests),cu_completions=len(completions),plan_sha256=hashlib.sha256(Path(__file__).with_name('plan.json').read_bytes()).hexdigest(),risk_models_sha256=hashlib.sha256(Path('/models/risk-models.json').read_bytes()).hexdigest(),source_commit=os.environ['SOURCE_COMMIT'],event_labels_available_to_scheduler=False,period_seconds=PLAN['period_seconds'],scheduled_budget_per_tick=1))

async def run():
    OUT.mkdir(exist_ok=True);agents=[];vm=None;tcp=None
    for p in ['/runtime','/nf','/nfkeys','/private','/witnesskey','/public','/witness','/config','/control']:Path(p).mkdir(exist_ok=True)
    generate();driver.ROOT=Path('/config');driver.OUT=OUT
    for s in SLOTS:
        Path(f'/nf/{s}').mkdir(exist_ok=True);Path(f'/nfkeys/{s}.key').write_bytes(secrets.token_bytes(32))
        (OUT/f'{s}.pub').write_bytes(key(Path(f'/private/{s}.key')))
    pub=key(Path('/witnesskey/witness.key'));Path('/public/witness.pub').write_bytes(pub);(OUT/'runtime-witness.pub').write_bytes(pub)
    db=MongoClient('mongodb://database',serverSelectionTimeoutMS=1000)
    for _ in range(60):
        try:db.admin.command('ping');break
        except Exception:await asyncio.sleep(.5)
    else:raise RuntimeError('private subscriber database unavailable')
    db.open5gs.subscribers.replace_one({'imsi':'999700000000001'},subscriber(),upsert=True)
    subprocess.run(['ip','tuntap','add','name','ogstun','mode','tun'],check=True)
    subprocess.run(['ip','addr','add','10.45.0.1/16','dev','ogstun'],check=True);subprocess.run(['ip','link','set','ogstun','up'],check=True)
    assert Path('/proc/sys/net/ipv4/ip_forward').read_text().strip()=='1'
    try:
        driver.start('nrf');await asyncio.sleep(.6)
        for s in ROLES[1:]:driver.start(s)
        for s in SLOTS:
            log=(OUT/f'{s}-agent.log').open('w');agents.append(subprocess.Popen([sys.executable,'-m','e2e5g.nf_agent',s],stdout=log,stderr=subprocess.STDOUT))
        log=(OUT/'runtime-witness.log').open('w');agents.append(subprocess.Popen([sys.executable,'-m','cu5g.witness'],stdout=log,stderr=subprocess.STDOUT))
        await driver.ready();vm,public=await vm_start()
        tcp=subprocess.Popen(['tcpdump','-i','ogstun','-nn','-U','-w',str(OUT/'ue-user-plane.pcap'),'tcp','port','18081'],stdout=(OUT/'tcpdump.log').open('w'),stderr=subprocess.STDOUT)
        atomic(Path('/control/core-ready.json'),dict(ready=True,guest_boot_id=public['boot_id']))
        await wait_app(public)
        replica=int(os.environ['REPLICA']);modes=list(PLAN['schedules']);shift=(replica-901)%3;modes=modes[shift:]+modes[:shift]
        models=json.loads(Path('/models/risk-models.json').read_text())['models']
        for order,mode in enumerate(modes):await collect(mode,order,replica,public,models)
        await subscription_controls(public)
        atomic(OUT/'complete.json',dict(replica=replica,modes=modes,source_commit=os.environ['SOURCE_COMMIT'],roles=ROLES,guest_distinct_kernel=True,physical_separate_host=False,production=False,semantic_attacks=False))
    finally:
        Path('/control/done').touch()
        for s,p in driver.PROCS.items():
            try:os.kill(p.pid,signal.SIGCONT)
            except ProcessLookupError:pass
        for p in agents+[vm,tcp]:
            if p is not None:
                p.terminate()
                try:p.wait(timeout=3)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        for s in reversed(list(driver.PROCS)):driver.stop(s)

if __name__=='__main__':asyncio.run(run())
