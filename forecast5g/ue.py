"""Ordinary test UE traffic and bounded delay on its own user-data TUN only."""
import concurrent.futures,hashlib,json,os,secrets,subprocess,time
from pathlib import Path
from e2e5g.ue import atomic
OUT=Path('/out/ue');CONTROL=Path('/control');OUT.mkdir(parents=True,exist_ok=True)
SRC=Path(os.environ.get('FORECAST_UER_SRC','/src/UERANSIM/config'))
BIN=Path(os.environ.get('FORECAST_UER_BIN','/src/UERANSIM/build'))

def probe(sequence):
    request=dict(nonce=secrets.token_hex(16),body_sha256=hashlib.sha256(f'ordinary-probe:{sequence}'.encode()).hexdigest())
    began=time.monotonic();row=dict(sequence=sequence,request=request,unix_ns=time.time_ns(),interface='uesimtun0')
    cmd=['curl','--interface','uesimtun0','--noproxy','*','-sS','--connect-timeout','.7','--max-time','1.5',
        '-w','\n%{time_total}','-H','Content-Type: application/json','--data-binary',json.dumps(request),'http://10.45.0.1:18081/receipt']
    r=subprocess.run(cmd,capture_output=True,text=True)
    row.update(returncode=r.returncode,error=r.stderr,elapsed_ms=(time.monotonic()-began)*1000,response=None,rtt_ms=None)
    try:
        body,elapsed=r.stdout.rsplit('\n',1);row['rtt_ms']=1000*float(elapsed);row['response']=json.loads(body)
    except (ValueError,TypeError):pass
    return row

def run():
    cfg=Path('/config');cfg.mkdir(exist_ok=True)
    (cfg/'gnb.yaml').write_text((SRC/'open5gs-gnb.yaml').read_text().replace('127.0.0.1','172.28.0.20').replace('127.0.0.5','172.28.0.10'))
    (cfg/'ue.yaml').write_text((SRC/'open5gs-ue.yaml').read_text().replace('127.0.0.1','172.28.0.20'))
    gnb=subprocess.Popen([str(BIN/'nr-gnb'),'-c',str(cfg/'gnb.yaml')],stdout=(OUT/'gnb.log').open('w'),stderr=subprocess.STDOUT)
    ue=None;attempt=0;last_command=-1;sequence=0;session_since=None;qdisc_ready=False
    def restart():
        nonlocal ue,attempt,session_since,qdisc_ready
        if ue is not None:
            ue.terminate()
            try:ue.wait(timeout=3)
            except subprocess.TimeoutExpired:ue.kill();ue.wait()
        subprocess.run(['ip','link','delete','uesimtun0'],capture_output=True)
        attempt+=1;session_since=None;qdisc_ready=False
        ue=subprocess.Popen([str(BIN/'nr-ue'),'-c',str(cfg/'ue.yaml')],stdout=(OUT/f'ue-{attempt}.log').open('w'),stderr=subprocess.STDOUT)
    try:
        time.sleep(2);restart()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            while not (CONTROL/'done').exists() and not (CONTROL/'stop-probes').exists():
                try:c=json.loads((CONTROL/'probe-command.json').read_text())
                except (FileNotFoundError,ValueError):time.sleep(.02);continue
                if c['id']==last_command:time.sleep(.02);continue
                assert c['id']>last_command and 0<=c['delay_ms']<=180 and c['count'] in [3,6]
                if c['restart']:restart()
                tun=Path('/sys/class/net/uesimtun0').exists()
                if tun:
                    action='change' if qdisc_ready else 'replace'
                    r=subprocess.run(['tc','qdisc',action,'dev','uesimtun0','root','netem','limit','1000','delay',str(c['delay_ms'])+'ms'],capture_output=True,text=True)
                    if r.returncode:raise RuntimeError('local UE netem failed: '+r.stderr)
                    qdisc_ready=True
                text=(OUT/f'ue-{attempt}.log').read_text(errors='replace');established='PDU Session establishment is successful' in text
                if established and session_since is None:session_since=time.monotonic_ns()
                begin=time.monotonic_ns();probes=list(pool.map(probe,range(sequence,sequence+c['count'])));sequence+=c['count']
                for p in probes:
                    p.update(attempt=attempt,command_id=c['id'],dnn='internet')
                    with (OUT/'probes.jsonl').open('a') as f:f.write(json.dumps(p)+'\n')
                atomic(OUT/'latest.json',probes[-1])
                batch=dict(command_id=c['id'],started_ns=begin,completed_ns=time.monotonic_ns(),unix_ns=time.time_ns(),attempt=attempt,
                    requested=c['count'],tun=tun,registered='Registration accept received' in text,session_established=established,
                    seconds_since_session=(time.monotonic_ns()-session_since)/1e9 if session_since else 0,probes=probes)
                atomic(OUT/'batch.json',batch)
                with (OUT/'actuator.jsonl').open('a') as f:f.write(json.dumps(dict(**c,attempt=attempt,tun=tun))+'\n')
                last_command=c['id']
    finally:
        for p in [ue,gnb]:
            if p is not None:
                p.terminate()
                try:p.wait(timeout=3)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        atomic(OUT/'done.json',dict(last_command=last_command,attempts=attempt,probes=sequence))

if __name__=='__main__':run()
