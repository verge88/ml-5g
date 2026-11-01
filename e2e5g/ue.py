"""Benign test UE/gNB and nonce receipts strictly over the UE TUN interface."""
import hashlib,json,os,secrets,subprocess,time
from pathlib import Path

OUT=Path('/out/ue'); CONTROL=Path('/control'); OUT.mkdir(parents=True,exist_ok=True)
def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data));os.replace(tmp,path)

def run():
    src=Path('/src/UERANSIM/config');cfg=Path('/config');cfg.mkdir(exist_ok=True)
    (cfg/'gnb.yaml').write_text((src/'open5gs-gnb.yaml').read_text().replace('127.0.0.1','172.28.0.20').replace('127.0.0.5','172.28.0.10'))
    (cfg/'ue.yaml').write_text((src/'open5gs-ue.yaml').read_text().replace('127.0.0.1','172.28.0.20'))
    gnb_log=(OUT/'gnb.log').open('w');gnb=subprocess.Popen(['/src/UERANSIM/build/nr-gnb','-c',str(cfg/'gnb.yaml')],stdout=gnb_log,stderr=subprocess.STDOUT)
    time.sleep(3);ue=None;token=None;attempt=0;seq=0
    try:
        while not (CONTROL/'done').exists():
            new=(CONTROL/'ue-restart').read_text() if (CONTROL/'ue-restart').exists() else 'initial'
            if new!=token:
                if ue is not None:
                    ue.terminate()
                    try:ue.wait(timeout=3)
                    except subprocess.TimeoutExpired:ue.kill();ue.wait()
                for name in ['uesimtun0']:
                    subprocess.run(['ip','link','delete',name],capture_output=True)
                attempt+=1;log=(OUT/f'ue-{attempt}.log').open('w')
                ue=subprocess.Popen(['/src/UERANSIM/build/nr-ue','-c',str(cfg/'ue.yaml')],stdout=log,stderr=subprocess.STDOUT)
                token=new
            started=time.monotonic();nonce=secrets.token_hex(16);body=hashlib.sha256(f'benign-test:{seq}'.encode()).hexdigest();request=dict(nonce=nonce,body_sha256=body)
            interfaces=subprocess.run(['ip','-j','addr','show'],capture_output=True,text=True).stdout
            row=dict(sequence=seq,attempt=attempt,unix_ns=time.time_ns(),request=request,interface='uesimtun0',interfaces=json.loads(interfaces),command_scope='test application through UE PDU session')
            cmd=['curl','--interface','uesimtun0','--noproxy','*','-sS','--connect-timeout','0.3','--max-time','0.7','-H','Content-Type: application/json','--data-binary',json.dumps(request),'http://10.45.0.1:18081/receipt']
            r=subprocess.run(cmd,capture_output=True,text=True)
            row.update(returncode=r.returncode,error=r.stderr,elapsed_ms=(time.monotonic()-started)*1000)
            try:row['response']=json.loads(r.stdout)
            except ValueError:row['response']=None
            with (OUT/'receipts.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            atomic(OUT/'latest.json',row);seq+=1
            time.sleep(max(0,1-(time.monotonic()-started)))
    finally:
        for p in [ue,gnb]:
            if p is not None:
                p.terminate()
                try:p.wait(timeout=3)
                except subprocess.TimeoutExpired:p.kill();p.wait()

if __name__=='__main__':run()
