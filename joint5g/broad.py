"""Data expansion with retained old weights and an update-budget control."""
import argparse,hashlib,json,itertools,os
from pathlib import Path
import numpy as np,torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from .deep import Net,windows,score
from .evaluate import read_run,metrics,combine
SEEDS=[11,29,47];TRAIN=[701,702,801,802,803,804,805,806];CAL=[809,810];TEST=[811,812,813]
def fit(inputs,previous,out,arch,comp):
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);out.mkdir(parents=True,exist_ok=True)
 old=json.loads((previous/'frozen.json').read_text());candidates=[d for d in old['models'] if d['name'] in old['selected'] and (d['arch'],d['composition'])==(arch,comp)];assert len(candidates)==3
 length=candidates[0]['length'];assert all(d['length']==length for d in candidates)
 runs=[read_run(inputs/f'joint-native-{n}',n) for n in TRAIN];calruns=[read_run(inputs/f'joint-native-{n}',n) for n in CAL]
 scaler=StandardScaler().fit(np.concatenate([r[comp][r['train']&r['normal']] for r in runs]));xx=[];cc=[];yy=[]
 for r in runs:
  x,c=windows(scaler.transform(r[comp]).astype('float32'),length);xx.append(x[r['train']]);cc.append(c[r['train']]);yy.append(r['truth'][r['train']])
 x=np.concatenate(xx);count=np.concatenate(cc);y=np.concatenate(yy).astype('float32');details=[]
 for seed in SEEDS:
  reference=next(d for d in candidates if d['seed']==seed)
  for arm in ['old_weights','broad_full','broad_equal_updates']:
   torch.manual_seed(seed);model=Net(x.shape[2],arch,length);losses=[];steps=0;changed=0
   if arm=='old_weights':
    path=previous/(reference['name']+'.pt');assert hashlib.sha256(path.read_bytes()).hexdigest()==reference['weights_sha256'];model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));center=reference['scaler_mean'];scale=reference['scaler_scale']
   else:
    center=scaler.mean_.tolist();scale=scaler.scale_.tolist();before={k:p.detach().clone() for k,p in model.named_parameters()};opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y==0).sum()/(y==1).sum())));rng=np.random.default_rng(seed)
    for epoch in range(24):
     model.train();total=0.;seen=0
     for ids in np.array_split(rng.permutation(len(y)),np.ceil(len(y)/64).astype(int)):
      if arm=='broad_equal_updates' and steps>=192:break
      opt.zero_grad(set_to_none=True);loss=criterion(model(torch.from_numpy(x[ids]),torch.from_numpy(count[ids])),torch.from_numpy(y[ids]));assert torch.isfinite(loss)
      loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();steps+=1;total+=float(loss.detach())*len(ids);seen+=len(ids)
     if seen:losses.append(total/seen)
     if arm=='broad_equal_updates' and steps>=192:break
    changed=sum(bool(torch.any(p.detach()!=before[k])) for k,p in model.named_parameters());assert changed>0
   cal=[]
   for r in calruns:
    v,c=windows(((r[comp]-np.asarray(center))/np.asarray(scale)).astype('float32'),length);cal.append(score(model,v,c)[r['normal']])
   cal=np.concatenate(cal);name=f'{arm}-{arch}-{comp}-{seed}';torch.save(model.state_dict(),out/(name+'.pt'));np.save(out/(name+'-calibration.npy'),cal)
   d=dict(name=name,arm=arm,arch=arch,composition=comp,length=length,seed=seed,threshold=float(cal.max()),scaler_mean=center,scaler_scale=scale,source_weights_sha256=reference['weights_sha256'] if arm=='old_weights' else None,source_name=reference['name'] if arm=='old_weights' else None,
    training_rows=496 if arm=='old_weights' else len(y),training_positive=48 if arm=='old_weights' else int(y.sum()),training_deployments=[701,702] if arm=='old_weights' else TRAIN,
    calibration_deployments=CAL,calibration_rows=len(cal),updates=192 if arm=='old_weights' else steps,changed_tensors=changed,losses=losses,weights_sha256=hashlib.sha256((out/(name+'.pt')).read_bytes()).hexdigest())
   details.append(d);print(name,d['training_rows'],d['updates'],flush=True)
 (out/'frozen.json').write_text(json.dumps(dict(commit=os.environ.get('GITHUB_SHA','local'),models=details,test_deployments=TEST),indent=2))

def evaluate(inputs,models,out):
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);details=[]
 for p in models.rglob('frozen.json'):details.extend(json.loads(p.read_text())['models'])
 assert len(details)==54;out.mkdir(parents=True,exist_ok=True);(out/'frozen-all.json').write_text(json.dumps(details,indent=2));results=[]
 for n in TEST:
  r=read_run(inputs/f'joint-native-{n}',n);arrays=dict(truth=r['truth'],normal=r['normal'],hard=r['hard'],unknown=r['unknown'],recovery=r['recovery'],tick=np.array([x['tick'] for x in r['rows']]),ns=np.array([x['ns'] for x in r['rows']]))
  for d in details:
   paths=list(models.rglob(d['name']+'.pt'));assert len(paths)==1 and hashlib.sha256(paths[0].read_bytes()).hexdigest()==d['weights_sha256']
   model=Net(len(d['scaler_mean']),d['arch'],d['length']);model.load_state_dict(torch.load(paths[0],map_location='cpu',weights_only=True))
   v,c=windows(((r[d['composition']]-np.asarray(d['scaler_mean']))/np.asarray(d['scaler_scale'])).astype('float32'),d['length']);s=score(model,v,c);ml=s>d['threshold'];full=combine(r['hard'],ml)
   arrays['score_'+d['name']]=s;arrays['ml_'+d['name']]=ml;arrays['full_'+d['name']]=full
   for method,alarm in [('ml',ml),('full',full)]:results.append(dict(replica=n,arm=d['arm'],arch=d['arch'],composition=d['composition'],seed=d['seed'],method=method,**metrics(r,alarm)))
  np.savez_compressed(out/f'test-{n}.npz',**arrays)
 (out/'results.json').write_text(json.dumps(dict(models=details,results=results),indent=2))
 lines=['# Обучение на расширенных связанных наблюдениях','','Прежние веса сохранены; все три варианта используют общую новую калибровку809/810 и новые тесты811–813. Расширенное обучение использует восемь развёртываний, включая паузы процессов. Это не проверка неизвестных семантических атак.','',
  '| Архитектура | Признаки | Обучение | Эпизоды из12 (11 / 29 / 47) | Средний FPR, % |','|---|---|---|---|---:|']
 for arch,comp,arm in itertools.product(['mlp','lstm','transformer'],['raw','context'],['old_weights','broad_equal_updates','broad_full']):
  rows=[r for r in results if (r['arch'],r['composition'],r['arm'],r['method'])==(arch,comp,arm,'full')];counts=[sum(r['detected_episodes'] for r in rows if r['seed']==seed) for seed in SEEDS]
  lines.append(f"| {arch} | {comp} | {arm} | {' / '.join(map(str,counts))} | {100*np.mean([r['fpr'] for r in rows]):.4f} |")
 (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path);p.add_argument('--models',type=Path);p.add_argument('--arch');p.add_argument('--composition');a=p.parse_args()
 if a.models:evaluate(a.input,a.models,a.output)
 else:fit(a.input,a.previous,a.output,a.arch,a.composition)
