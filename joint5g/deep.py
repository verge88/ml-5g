"""Causal sequence learning with frozen development selection and fresh deployment tests."""
import argparse,json,hashlib,os,itertools
from pathlib import Path
import numpy as np,torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from .evaluate import read_run,metrics,combine
SEEDS=[11,29,47];ARCH=['mlp','lstm','transformer'];COMPOSITIONS=['raw','context']

def windows(x,length):
    n=len(x);out=np.zeros((n,length,x.shape[1]),dtype='float32');counts=np.minimum(np.arange(n)+1,length)
    for i,c in enumerate(counts):out[i,:c]=x[i-c+1:i+1]
    return out,counts.astype('int64')

class Net(nn.Module):
    def __init__(self,width,arch,length):
        super().__init__();self.arch=arch
        if arch=='mlp':self.network=nn.Sequential(nn.Linear(width,48),nn.ReLU(),nn.Linear(48,24),nn.ReLU(),nn.Linear(24,1))
        else:
            self.project=nn.Linear(width,32);self.position=nn.Parameter(torch.zeros(length,32))
            self.network=nn.LSTM(32,32,batch_first=True) if arch=='lstm' else nn.TransformerEncoder(nn.TransformerEncoderLayer(32,2,64,dropout=0.,batch_first=True),1,enable_nested_tensor=False)
            self.head=nn.Sequential(nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x,count):
        ii=torch.arange(len(x));last=count-1
        if self.arch=='mlp':return self.network(x[ii,last]).flatten()
        z=torch.tanh(self.project(x))+self.position[None,:,:]
        if self.arch=='lstm':h,_=self.network(z)
        else:h=self.network(z,src_key_padding_mask=torch.arange(x.shape[1])[None,:]>=count[:,None])
        return self.head(h[ii,last]).flatten()

@torch.no_grad()
def score(model,x,count):
    model.eval();return np.concatenate([model(torch.from_numpy(x[i:i+256]),torch.from_numpy(count[i:i+256])).sigmoid().numpy() for i in range(0,len(x),256)])

def fit(inputs,out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);out.mkdir(parents=True,exist_ok=True)
    training=[read_run(inputs/f'joint-native-{i}',i) for i in [701,702]]
    validation=[read_run(inputs/f'joint-native-{i}',i) for i in [704,705,706]]
    calibration=read_run(inputs/'joint-native-703',703);details=[]
    for arch,composition in itertools.product(ARCH,COMPOSITIONS):
        scaler=StandardScaler().fit(np.concatenate([r[composition][r['train']&r['normal']] for r in training]))
        for length in ([1] if arch=='mlp' else [16,32]):
            xx=[];cc=[];yy=[]
            for r in training:
                x,c=windows(scaler.transform(r[composition]).astype('float32'),length);xx.append(x[r['train']]);cc.append(c[r['train']]);yy.append(r['truth'][r['train']])
            x=np.concatenate(xx);count=np.concatenate(cc);y=np.concatenate(yy).astype('float32')
            for seed in SEEDS:
                torch.manual_seed(seed);model=Net(x.shape[2],arch,length);before={k:p.detach().clone() for k,p in model.named_parameters()};opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
                criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y==0).sum()/(y==1).sum())));losses=[]
                for epoch in range(24):
                    model.train();total=0.
                    for i in range(0,len(y),64):
                        opt.zero_grad(set_to_none=True);loss=criterion(model(torch.from_numpy(x[i:i+64]),torch.from_numpy(count[i:i+64])),torch.from_numpy(y[i:i+64]));assert torch.isfinite(loss)
                        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss.detach())*len(y[i:i+64])
                    losses.append(total/len(y))
                changed=sum(bool(torch.any(p.detach()!=before[k])) for k,p in model.named_parameters());assert changed
                dev=[]
                for r in validation:
                    v,c=windows(scaler.transform(r[composition]).astype('float32'),length);s=score(model,v,c);mask=r['normal']|r['truth'];q=np.clip(s[mask],1e-7,1-1e-7);target=r['truth'][mask]
                    target=target.astype('float32');dev.append(float(np.mean(-target*np.log(q)-(1-target)*np.log(1-q))))
                v,c=windows(scaler.transform(calibration[composition]).astype('float32'),length);cal=score(model,v,c)[calibration['normal']]
                name=f'{arch}-{composition}-{length}-{seed}';torch.save(model.state_dict(),out/(name+'.pt'));np.save(out/(name+'-calibration.npy'),cal)
                d=dict(name=name,arch=arch,composition=composition,length=length,seed=seed,validation_loss=float(np.mean(dev)),threshold=float(cal.max()),losses=losses,changed_tensors=changed,
                    scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),parameters=sum(p.numel() for p in model.parameters()),weights_sha256=hashlib.sha256((out/(name+'.pt')).read_bytes()).hexdigest())
                details.append(d);print(name,d['validation_loss'],flush=True)
    selected=[]
    for arch,composition in itertools.product(ARCH,COMPOSITIONS):
        candidates=[d for d in details if (d['arch'],d['composition'])==(arch,composition)];lengths=sorted(set(d['length'] for d in candidates))
        best=min(lengths,key=lambda n:(np.mean([d['validation_loss'] for d in candidates if d['length']==n]),n));selected.extend(d['name'] for d in candidates if d['length']==best)
    result=dict(commit=os.environ.get('GITHUB_SHA','local'),training_deployments=[701,702],development_deployments=[704,705,706],calibration_deployment=703,test_deployments=[707,708,709],models=details,selected=selected)
    assert len(details)==30 and len(selected)==18
    (out/'frozen.json').write_text(json.dumps(result,indent=2));print('FROZEN',selected)

def evaluate(inputs,models,out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);frozen=json.loads((models/'frozen.json').read_text());out.mkdir(parents=True,exist_ok=True);results=[]
    for n in [707,708,709]:
        r=read_run(inputs/f'joint-native-{n}',n);arrays=dict(truth=r['truth'],normal=r['normal'],hard=r['hard'],unknown=r['unknown'],recovery=r['recovery'],tick=np.array([x['tick'] for x in r['rows']]),ns=np.array([x['ns'] for x in r['rows']]))
        results.append(dict(replica=n,model='integrity_only',composition='none',seed=None,method='integrity_only',**metrics(r,r['hard'])))
        for d in frozen['models']:
            if d['name'] not in frozen['selected']:continue
            path=models/(d['name']+'.pt');assert hashlib.sha256(path.read_bytes()).hexdigest()==d['weights_sha256']
            model=Net(len(d['scaler_mean']),d['arch'],d['length']);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            x=(r[d['composition']]-np.array(d['scaler_mean']))/np.array(d['scaler_scale']);v,c=windows(x.astype('float32'),d['length']);s=score(model,v,c);ml=s>d['threshold'];full=combine(r['hard'],ml)
            arrays['score_'+d['name']]=s;arrays['ml_'+d['name']]=ml;arrays['full_'+d['name']]=full
            for method,alarm in [('ml',ml),('full',full)]:results.append(dict(replica=n,model=d['arch'],composition=d['composition'],seed=d['seed'],length=d['length'],method=method,**metrics(r,alarm)))
        np.savez_compressed(out/f'test-{n}.npz',**arrays)
    (out/'results.json').write_text(json.dumps(dict(frozen=frozen,results=results),indent=2))
    lines=['# Глубокие модели на новых связанных наблюдениях','','Обучение701/702, разработка704–706, калибровка703, новые тесты707–709. Длина окна выбрана по средней ошибке на разработке до проверки.','',
        '| Модель | Признаки | Решение | Эпизоды из12 (11 / 29 / 47) | Средний FPR, % |','|---|---|---|---|---:|']
    for arch,composition,method in itertools.product(ARCH,COMPOSITIONS,['ml','full']):
        v=[r for r in results if (r['model'],r['composition'],r['method'])==(arch,composition,method)]
        counts=[sum(r['detected_episodes'] for r in v if r['seed']==seed) for seed in SEEDS]
        lines.append(f"| {arch} | {composition} | {method} | {' / '.join(map(str,counts))} | {100*np.mean([r['fpr'] for r in v]):.4f} |")
    lines+=['','Проверены локальные отказы, не семантические кибератаки. Три инициализации не увеличивают12 физических проверочных событий. UNKNOWN сохраняется; ML не подтверждает целостность и не отменяет тревоги. Целевой FPR0,1% проверяется по фактическим показателям, положительный результат не требуется.']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--models',type=Path);a=p.parse_args()
    if a.models:evaluate(a.input,a.models,a.output)
    else:fit(a.input,a.output)
