"""Common normal-only fitting protocol and bounded causal sequence batches."""
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from neural5g.models import make_model
from .prepare import P
from . import references

def causal_batch(x,start,end,length):
    ids=np.arange(start,end)[:,None]-np.arange(length-1,-1,-1)[None,:]
    valid=ids>=0
    values=x[np.maximum(ids,0)].copy();values[~valid]=0
    # Existing sequence models consume right-padded windows.
    lengths=valid.sum(1)
    for i,n in enumerate(lengths):values[i,:n]=values[i,-n:].copy();values[i,n:]=0
    return torch.from_numpy(values),torch.from_numpy(lengths.astype(np.int64))

def negative_batch(x,s,d,info,rng):
    ns=s[rng.permutation(len(s))];nd=d[rng.permutation(len(d))]
    out=x.copy();n=info['n_nodes'];identity=2*n
    raw=np.zeros((len(out),identity),dtype=np.float32)
    raw[np.arange(len(raw)),ns]=1;raw[np.arange(len(raw)),n+nd]=1
    center=np.asarray(info['center'][:identity]);scale=np.asarray(info['scale'][:identity])
    out[:,:identity]=np.clip((raw-center)/scale,-20,20)
    return out,ns,nd

class Existing:
    def __init__(self,name,width,info):
        self.name=name;self.sequential=name in ('lstm','transformer')
        self.model=make_model(name,width,info['n_nodes'],P)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=P['learning_rate'],weight_decay=P['weight_decay'])
    def train(self,yes):self.model.train(yes)
    def parameters(self):return list(self.model.named_parameters())
    def reset(self,capacity):
        if not self.sequential:self.model.reset()
    def score(self,stream,start,end,negative=None):
        if self.sequential:
            history,lengths=causal_batch(stream['x'],start,end,P['sequence_length'])
            if negative is not None:history[torch.arange(len(history)),lengths-1]=torch.from_numpy(negative[0])
            return self.model(history,lengths)
        x,s,d=(stream['x'][start:end],stream['source'][start:end],stream['destination'][start:end]) if negative is None else negative
        return self.model(torch.from_numpy(x),torch.from_numpy(s),torch.from_numpy(d),torch.arange(start,end,dtype=torch.float64)+stream.get('offset',0))
    def observe(self,stream,start,end):
        if not self.sequential:
            self.model.observe(torch.from_numpy(stream['x'][start:end]),torch.from_numpy(stream['source'][start:end]),
                torch.from_numpy(stream['destination'][start:end]),torch.arange(start,end,dtype=torch.float64)+stream.get('offset',0))
    def detach(self):pass
    def save(self,path):torch.save(self.model.state_dict(),path)

class PublishedTGN:
    def __init__(self,width,info,root):
        from torch_geometric.nn.models.tgn import LastNeighborLoader
        self.loader_class=LastNeighborLoader
        self.model=references.granomaly(root)(SimpleNamespace(num_nodes=info['n_nodes'],msg=torch.empty(0,width)),torch.device('cpu'),P['learning_rate'])
        self.n_nodes=info['n_nodes'];self.width=width;self.optimizer=self.model.optimizer
    def train(self,yes):
        for m in (self.model.memory,self.model.gnn,self.model.link_pred):m.train(yes)
    def parameters(self):
        seen=set();result=[]
        for prefix in ('memory','gnn','link_pred'):
            for name,p in getattr(self.model,prefix).named_parameters():
                if id(p) not in seen:seen.add(id(p));result.append((prefix+'.'+name,p))
        return result
    def reset(self,capacity):
        self.model.memory.reset_state();self.loader=self.loader_class(self.n_nodes,size=50,device='cpu')
        self.ht=torch.empty(capacity,dtype=torch.long);self.hx=torch.empty(capacity,self.width);self.count=0
    def score(self,stream,start,end,negative=None):
        _,s,d=(stream['x'][start:end],stream['source'][start:end],stream['destination'][start:end]) if negative is None else negative
        s=torch.from_numpy(s);d=torch.from_numpy(d)
        nodes,edges,ids=self.loader(torch.cat((s,d)).unique())
        self.model.assoc[nodes]=torch.arange(len(nodes))
        z,updated=self.model.memory(nodes)
        z=self.model.gnn(z,updated,edges,self.ht[ids],self.hx[ids])
        return self.model.link_pred(z[self.model.assoc[s]],z[self.model.assoc[d]]).squeeze(-1)
    def observe(self,stream,start,end):
        src=torch.from_numpy(stream['source'][start:end]);dst=torch.from_numpy(stream['destination'][start:end])
        t=torch.arange(start,end,dtype=torch.long)+stream.get('offset',0);x=torch.from_numpy(stream['x'][start:end])
        self.ht[self.count:self.count+len(t)]=t;self.hx[self.count:self.count+len(t)]=x
        self.model.memory.update_state(src,dst,t,x);self.loader.insert(src,dst);self.count+=len(t)
    def detach(self):self.model.memory.detach()
    def save(self,path):
        torch.save({prefix:getattr(self.model,prefix).state_dict() for prefix in ('memory','gnn','link_pred')},path)

class SVDD(nn.Module):
    def __init__(self,width):
        super().__init__();self.net=nn.Sequential(nn.Linear(width,64,bias=False),nn.ReLU(),nn.Linear(64,32,bias=False),nn.ReLU(),nn.Linear(32,16,bias=False))
    def forward(self,x):return self.net(x)

def fit_links(engine,stream,info,seed):
    engine.train(True);params=engine.parameters();initial={n:p.detach().clone() for n,p in params}
    losses=[];grads={};rng=np.random.default_rng(seed);collisions=0;examples=0
    for epoch in range(P['epochs']):
        engine.reset(len(stream['x'])+P['batch_size']);total=0.;points=0
        for start in range(0,len(stream['x']),P['batch_size']):
            end=min(start+P['batch_size'],len(stream['x']))
            negative=negative_batch(stream['x'][start:end],stream['source'][start:end],stream['destination'][start:end],info,rng)
            collisions+=int(((negative[1]==stream['source'][start:end])&(negative[2]==stream['destination'][start:end])).sum());examples+=end-start
            engine.optimizer.zero_grad(set_to_none=True)
            positive=engine.score(stream,start,end);neg=engine.score(stream,start,end,negative)
            loss=(nn.functional.binary_cross_entropy_with_logits(positive,torch.zeros_like(positive))+
                  nn.functional.binary_cross_entropy_with_logits(neg,torch.ones_like(neg)))/2
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            # PyG's training memory update follows its published pre-backward convention.
            if isinstance(engine,PublishedTGN):engine.observe(stream,start,end)
            loss.backward()
            for name,p in params:
                if p.grad is not None:grads[name.split('.')[0]]=max(grads.get(name.split('.')[0],0.),float(p.grad.abs().sum()))
            nn.utils.clip_grad_norm_([p for _,p in params],P['gradient_clip']);engine.optimizer.step()
            if not isinstance(engine,PublishedTGN):engine.observe(stream,start,end)
            engine.detach();total+=float(loss.detach())*(end-start);points+=end-start
        losses.append(total/points)
        print(f'epoch {epoch+1}/{P["epochs"]} loss={losses[-1]:.6f}',flush=True)
    changed=sum(bool(torch.any(p.detach()!=initial[n])) for n,p in params)
    if not changed:raise ValueError('no learned parameter changed')
    if isinstance(engine,Existing) and engine.name=='tgn':
        if not all(grads.get(k,0)>0 for k in ('message','updater','attention','time_encoder')):raise ValueError('untrained TGN component')
    return dict(epoch_loss=losses,changed_parameter_tensors=changed,parameter_tensors=len(params),parameters=sum(p.numel() for _,p in params),
        gradient_max_l1=grads,synthetic_negative_examples=examples,same_event_endpoint_collisions=collisions,
        negative_scope='permuted training endpoints only; no real attack labels and no negatives in history')

@torch.no_grad()
def predict_links(engine,train,stream):
    engine.train(False);engine.reset(len(train['x'])+len(stream['x'])+P['batch_size'])
    if not (isinstance(engine,Existing) and engine.sequential):
        for start in range(0,len(train['x']),P['batch_size']):
            end=min(start+P['batch_size'],len(train['x']));engine.score(train,start,end);engine.observe(train,start,end)
    target={**stream,'offset':len(train['x'])};scores=[]
    for start in range(0,len(stream['x']),P['batch_size']):
        end=min(start+P['batch_size'],len(stream['x']));scores.append(engine.score(target,start,end).sigmoid().numpy())
        engine.observe(target,start,end)
    result=np.concatenate(scores)
    if not np.isfinite(result).all():raise ValueError('nonfinite prediction')
    return result
