"""All observations collected on live isolated5GC. No predictor or scoring here."""
import asyncio,hashlib,json,os,platform,secrets,subprocess,time,urllib.request
from dataclasses import asdict
from pathlib import Path
import numpy as np
from forecast5g import core as base
from cu5g.common import POLICY,atomic,native_profile,verified
from fullsuite.native_hook import Sampler
from hybrid.vendor.stage39_kie import FreshnessAnchor,KIERequest,StreamRelationalVerifier,sha256_json
from live5g.capture import observe
from e2e5g.evidence import receipt
from .profiles import configuration,command,trials
from .evidence import verify_probe

OUT=Path('/out')
PLAN=json.loads(Path(__file__).with_name('plan.json').read_text())


async def vm_start():
    choices=[('tcg,thread=multi','max')]
    if os.path.exists('/dev/kvm') and os.access('/dev/kvm',os.W_OK):choices.insert(0,('kvm','host'))
    for acceleration,cpu in choices:
        logpath=OUT/f"guest-{acceleration.split(',')[0]}-serial.log"
        args=['qemu-system-x86_64','-accel',acceleration,'-cpu',cpu,'-m','768','-smp','1','-nographic','-no-reboot',
              '-kernel','/vm/vmlinuz-virt','-initrd','/vm/rootfs.cpio.gz','-append','console=ttyS0,115200 rdinit=/init loglevel=7 nokaslr',
              '-object','rng-random,filename=/dev/urandom,id=rng0','-device','virtio-rng-pci,rng=rng0',
              '-nic','user,model=virtio-net-pci,restrict=on,hostfwd=tcp:10.45.0.1:18081-:8081,hostfwd=tcp:127.0.0.1:18082-:8082']
        p=subprocess.Popen(args,stdout=logpath.open('w'),stderr=subprocess.STDOUT)
        public=None
        for _ in range(100):
            if p.poll() is not None:break
            try:
                with urllib.request.urlopen('http://10.45.0.1:18081/public',timeout=.3) as r:candidate=json.load(r)
                serial=logpath.read_text(errors='replace')
                if candidate['public_key'] in serial and candidate['boot_id'] in serial:public=candidate;break
            except Exception:pass
            await asyncio.sleep(.5)
        if public:
            host_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();assert public['boot_id']!=host_boot
            (OUT/'guest-serial.log').write_text(logpath.read_text(errors='replace'))
            atomic(OUT/'guest-public.json',dict(**public,host_boot_id=host_boot,host_kernel=platform.release(),acceleration=acceleration,
                source='serial-pinned guest key; measurement paths share this signer',
                kernel_sha256=hashlib.sha256(Path('/vm/vmlinuz-virt').read_bytes()).hexdigest(),
                initrd_sha256=hashlib.sha256(Path('/vm/rootfs.cpio.gz').read_bytes()).hexdigest()))
            return p,public
        p.terminate()
        try:p.wait(timeout=3)
        except subprocess.TimeoutExpired:p.kill();p.wait()
    raise RuntimeError('diagnostic guest failed to start')


def manage(wait_ms):
    assert 0<=wait_ms<=150
    req=urllib.request.Request('http://127.0.0.1:18082/',json.dumps(dict(wait_ms=wait_ms)).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=2) as r:assert json.load(r)=={'applied':True}


def bypass_probe(sequence):
    request=dict(nonce=secrets.token_hex(16),body_sha256=hashlib.sha256(f'bypass:{sequence}'.encode()).hexdigest())
    began=time.monotonic_ns();unix_ns=time.time_ns()
    row=dict(sequence=sequence,request=request,unix_ns=unix_ns,response=None,returncode=1,rtt_ms=None,interface='core-bypass')
    req=urllib.request.Request('http://10.45.0.1:18081/receipt',json.dumps(request).encode(),{'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=1.5) as r:body=json.load(r)
        row.update(returncode=0,response=body,rtt_ms=(time.monotonic_ns()-began)/1e6)
    except Exception as exc:row['error']=type(exc).__name__
    return row


async def capture(public):
    replica=int(os.environ['REPLICA'])
    sampler=Sampler(StreamRelationalVerifier(POLICY),FreshnessAnchor(OUT/'controller-anchor.json'))
    sampler.last_issue=1e30;sampler.cycle=60000
    last_signed={s:None for s in POLICY.slots};seen=set();completions=[];requests=[];index=0

    async def sample(c,name,tick,folder):
        nonlocal index
        began=time.monotonic();identifier=index;index+=1
        await asyncio.to_thread(manage,c['app_wait_ms'])
        atomic('/control/probe-command.json',dict(id=identifier,**{k:v for k,v in c.items() if k!='app_wait_ms'}))
        direct_task=asyncio.create_task(asyncio.to_thread(lambda:[bypass_probe(identifier*2+j) for j in range(2)]))
        raw=await observe();obs=sampler.observe(raw);monitor=obs['ausf']['integrity_monitor']
        for s in POLICY.slots:
            if monitor['states'][s]=='AUTHENTICATED_CURRENT_CU':
                profile=obs[s]['nrf']['body']['nfInstances'][0];envelope=profile['customInfo']['labKIE']['public_bundle']
                payload=verified(envelope,OUT/f'{s}.pub');assert payload['request']==asdict(sampler.latest[s])
                last_signed[s]=payload['response']['issued_at_ms']/1000;token=(s,payload['request']['cycle'])
                if token not in seen:completions.append(dict(slot=s,public_bundle=envelope));seen.add(token)
        if identifier%4==0:
            sampler.cycle+=1
            for s in POLICY.slots:
                body=raw[s]['nrf']['body'];profiles=body.get('nfInstances',[]) if isinstance(body,dict) else []
                if len(profiles)==1:
                    normalized=native_profile(s,profiles[0])
                    if POLICY.matches(normalized):
                        q=KIERequest.create('native-controller',s,sampler.cycle,POLICY.digest,sha256_json(normalized),sampler.keys[s])
                        sampler.latest[s]=q;atomic(f'/nf/{s}/request.json',asdict(q));requests.append(dict(id=identifier,slot=s,request=asdict(q)))
        batch=None
        for _ in range(400):
            try:
                candidate=json.loads((OUT/'ue/batch.json').read_text())
                if candidate['command_id']==identifier:batch=candidate;break
            except (FileNotFoundError,ValueError):pass
            await asyncio.sleep(.02)
        if batch is None:raise RuntimeError('diagnostic UE batch unavailable')
        bypass=await direct_task
        end_values=[];direct_values=[]
        for p in batch['probes']:
            if p['returncode']==0:
                assert receipt(p,public,now_ns=batch['unix_ns']);end_values.append(p['rtt_ms'])
        now_ns=time.time_ns()
        for p in bypass:
            if p['returncode']==0:
                assert verify_probe(p,public,'core-bypass',now_ns);direct_values.append(p['rtt_ms'])
        ages={s:time.time()-last_signed[s] if last_signed[s] is not None else 1e6 for s in POLICY.slots}
        row=dict(id=identifier,trial=name,tick=tick,monotonic_ns=time.monotonic_ns(),unix_ns=now_ns,batch=batch,bypass=bypass,
            measured=dict(end_signed=len(end_values),bypass_signed=len(direct_values),
                end_ms=float(np.median(end_values)) if end_values else None,
                bypass_ms=float(np.median(direct_values)) if direct_values else None),
            cu=dict(ages=ages,hard_alarm=monitor['hard_alarm'],witness_available=monitor['witness_available']),observations=obs)
        with (folder/'observations.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        await asyncio.sleep(max(0,PLAN['period_seconds']-(time.monotonic()-began)))
        return row

    warm=OUT/'warmup';warm.mkdir()
    zero=dict(delay_ms=0.,app_wait_ms=0.,count=3,restart=False)
    for t in range(PLAN['warmup']):last=await sample(zero,'warmup',t,warm)
    assert last['measured']['end_signed']>=2 and last['measured']['bypass_signed']==2
    entries=[]
    for family,repeat in trials(replica):
        name=f'{family}-{repeat}';folder=OUT/'trials'/name;folder.mkdir(parents=True)
        config=configuration(replica,family,repeat);atomic(folder/'actuation-plan.json',config)
        for tick in range(PLAN['ticks']):await sample(command(config,tick),name,tick,folder)
        entries.append(dict(name=name,family=family,repeat=repeat,rows=PLAN['ticks']))
    atomic(OUT/'cu-completions.json',completions);atomic(OUT/'cu-requests.json',requests)
    atomic(OUT/'bound-observations.json',sampler.bound_evidence)
    atomic(OUT/'diagnostic-capture.json',dict(replica=replica,trials=entries,source_commit=os.environ['SOURCE_COMMIT'],
        plan_sha256=hashlib.sha256(Path(__file__).with_name('plan.json').read_bytes()).hexdigest(),
        predictor_used=False,all_bypass_probes_physically_collected=True,adaptive_physical_savings_measured=False))


if __name__=='__main__':
    base.capture=capture
    base.base.vm_start=vm_start
    asyncio.run(base.run())
