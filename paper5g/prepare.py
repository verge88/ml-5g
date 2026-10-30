"""Deidentified, full-size SBI matrices; labels never enter the feature builder."""
import argparse
from collections import Counter,defaultdict,deque
import hashlib,json
from pathlib import Path
import numpy as np
from coverage5g.audit import download,prepare_captures,requests
from semantic5g.analyze import read_csv
from hybrid.fetch_data import fetch as fetch_granomaly
from zipfile import ZipFile

P=json.loads(Path(__file__).with_name('protocol.json').read_text())
METHODS=('GET','POST','PUT','PATCH','DELETE','OPTIONS','HEAD')

def bin_id(text,n):return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')%n

def partition(normal):
    times=np.asarray([r['time'] for r in normal])
    a=np.searchsorted(times,times[int(len(times)*P['normal_train_fraction'])-1],side='right')
    b=np.searchsorted(times,times[int(len(times)*P['normal_calibration_end_fraction'])-1],side='right')
    assert 0<a<b<len(normal)
    return dict(train=normal[:a],calibration=normal[a:b],clean_holdout=normal[b:])

def features(rows,nodes):
    n=len(nodes)+1;identity=2*n
    x=np.zeros((len(rows),identity+49+8),dtype=np.float32)
    context=np.zeros((len(rows),40),dtype=np.float32)
    sources=[];destinations=[];history=defaultdict(lambda:deque(maxlen=32))
    last_c={};last_e={};last_d={};recent=deque(maxlen=32);last_time=None
    for i,r in enumerate(rows):
        src,dst=r['source'],r['destination'];s=nodes.get(src,0);d=nodes.get(dst,0)
        sources.append(s);destinations.append(d);x[i,s]=1;x[i,n+d]=1
        k=identity
        if r['method'] in METHODS:x[i,k+METHODS.index(r['method'])]=1
        service=r['path'].split('/')[1] if '/' in r['path'] else ''
        x[i,k+7+bin_id(service,8)]=1;x[i,k+15+bin_id(r['path'],32)]=1
        x[i,k+47+(r.get('wire','HTTP2')=='HTTP2')]=1
        t=r['time'];edge=(src,dst)
        gaps=[0 if v is None else max(0.,t-v) for v in (last_time,last_c.get(src),last_d.get(dst),last_e.get(edge))]
        past=history[src];op=(dst,r['method'],r['path'])
        vals=[*np.log1p(gaps),np.log1p(sum(a==src for a,b in recent)),
              np.log1p(sum((a,b)==edge for a,b in recent)),np.log1p(len({b for a,b in recent if a==src})),np.log1p(len(recent))]
        x[i,k+49:]=vals
        for lag in (1,2):
            if len(past)>=lag:context[i,(lag-1)*16+bin_id(repr(past[-lag]),16)]=1
        context[i,32:]=[len(past)/32,len(set(past))/32,len({v[0] for v in past})/32,
            len({v[1] for v in past})/7,len({v[2] for v in past})/32,
            sum(v==op for v in past)/max(1,len(past)),sum(v[1]==r['method'] for v in past)/max(1,len(past)),
            sum(v[2]==r['path'] for v in past)/max(1,len(past))]
        past.append(op);recent.append(edge);last_c[src]=t;last_d[dst]=t;last_e[edge]=t;last_time=t
    return x,context,np.asarray(sources,dtype=np.int64),np.asarray(destinations,dtype=np.int64)

def build_dataset(name,normal,targets,out):
    streams={**partition(normal),**targets}
    nodes={v:i+1 for i,v in enumerate(sorted({r[k] for r in streams['train'] for k in ('source','destination')}))}
    values={k:features(rows,nodes) for k,rows in streams.items()}
    train=np.concatenate(values['train'][:2],axis=1)
    center=np.median(train,axis=0);scale=np.maximum(1.4826*np.median(abs(train-center),axis=0),.1)
    base_dim=values['train'][0].shape[1]
    folder=out/name;folder.mkdir(parents=True,exist_ok=True)
    categories={v:i for i,v in enumerate(sorted({r.get('kind','normal') for rr in streams.values() for r in rr}))}
    episode_ids={};manifest={}
    for key,rows in streams.items():
        x,context,source,destination=values[key]
        full=np.concatenate((x,context),axis=1)
        full=np.clip((full-center)/scale,-20,20).astype(np.float32)
        y=np.array([r.get('label',0) for r in rows],dtype=np.int8)
        kinds=np.array([categories[r.get('kind','normal')] for r in rows],dtype=np.int16)
        episodes=[]
        for r in rows:
            group=(key,r.get('episode',''))
            if group not in episode_ids:episode_ids[group]=len(episode_ids)
            episodes.append(episode_ids[group])
        stamps=np.array([r['time']-rows[0]['time'] for r in rows],dtype=np.float64)
        # Integer monotone event indices are the native Granomaly time convention.
        np.savez_compressed(folder/(key+'.npz'),x=full,source=source,destination=destination,
            time=stamps,y=y,kind=kinds,episode=np.asarray(episodes,dtype=np.int64))
        manifest[key]=dict(rows=len(rows),normal=int((y==0).sum()),positive=int((y==1).sum()),unknown=int((y<0).sum()),
            unknown_source=int((source==0).sum()),unknown_destination=int((destination==0).sum()),
            sha256=hashlib.sha256((folder/(key+'.npz')).read_bytes()).hexdigest())
    info=dict(dataset=name,n_nodes=len(nodes)+1,base_dim=base_dim,context_dim=40,
        identity_dim=2*(len(nodes)+1),center=center.tolist(),scale=scale.tolist(),categories=categories,streams=manifest,
        node_vocabulary_sha256=hashlib.sha256(json.dumps(nodes,sort_keys=True).encode()).hexdigest(),
        identity_policy='training-only observed endpoint IDs, unknown=0; vocabulary not released',
        test_size_cap=None,training_only_normal=True,protocol=P)
    (folder/'manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(dict(dataset=name,streams=manifest)),flush=True)
    return info

def ctd(scratch,out):
    scratch.mkdir(parents=True,exist_ok=True);archive=scratch/'CTD5G.zip'
    print('CTD archive download/checksum',flush=True);download(archive)
    with ZipFile(archive) as z:db,audit,ports=prepare_captures(z,scratch)
    print('CTD full offline request decoding',flush=True)
    normal,an=requests(scratch/'benign.pcap',ports,scratch)
    mixed,am=requests(scratch/'mixed.pcap',ports,scratch)
    for r in normal:r.update(label=0,kind='normal',episode='clean')
    for r in mixed:
        label=db.execute('SELECT label,kind,episode FROM labels WHERE frame=?',(r['frame'],)).fetchone()
        r.update(label=label[0] if label else -1,kind=label[1] if label else 'UNKNOWN',episode=label[2] if label else '')
    db.close()
    assert len(normal)==227159 and len(mixed)==77695,'Pinned extraction counts changed'
    assert sum(r['label']==1 for r in mixed)==7878
    result=build_dataset('ctd5g',normal,{'mixed':mixed},out)
    result.update(packet_audit=audit,ambiguous_excluded=[an,am],archive_sha256=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest())
    (out/'ctd5g'/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

def granomaly(data,out):
    fetch_granomaly(data)
    rows={}
    for filename,key in [('bridge_normal.csv','normal'),('bridge_attack_nrf.csv','nrf'),('bridge_attack_smf.csv','smf')]:
        rr,_=read_csv(data/filename)
        for r in rr:
            r.update(label=int(r['label']!='normal'),kind=r['label'],episode=r['label'],wire='HTTP2')
        rows[key]=sorted(rr,key=lambda r:r['time'])
    build_dataset('granomaly',rows['normal'],{'nrf':rows['nrf'],'smf':rows['smf']},out)

def main(scratch,out,granomaly_data):
    out.mkdir(parents=True,exist_ok=True)
    granomaly(granomaly_data,out);ctd(scratch,out)
    (out/'protocol.json').write_text(json.dumps(P,indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scratch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--granomaly-data',type=Path,default=Path('data/granomaly'));a=p.parse_args();main(a.scratch,a.output,a.granomaly_data)
