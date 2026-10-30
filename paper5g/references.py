"""Fetch only reviewed ML modules at pinned hashes, never upstream attack tools."""
import hashlib,importlib.util,json,sys,urllib.request
from pathlib import Path
from .prepare import P

def fetch(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);records=[]
    g=P['granomaly_reference'];k=P['kitnet_reference']
    specs=[('granomaly',g['repository'],g['commit'],{'models/tgn.py':g['blob']}),
           ('kitnet',k['repository'],k['commit'],k['files'])]
    for name,repo,commit,files in specs:
        folder=root/name;folder.mkdir(exist_ok=True)
        for path,expected in files.items():
            target=folder/Path(path).name
            url=f'https://raw.githubusercontent.com/{repo}/{commit}/{path}'
            if not target.exists():
                with urllib.request.urlopen(url,timeout=60) as response:data=response.read(128*1024+1)
                if len(data)>128*1024:raise ValueError('reference module too large')
                target.write_bytes(data)
            data=target.read_bytes()
            actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if actual!=expected:raise ValueError('reference hash mismatch: '+path)
            records.append(dict(reference=name,path=path,url=url,git_blob=actual,sha256=hashlib.sha256(data).hexdigest()))
    return records

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def granomaly(root):return module(Path(root)/'granomaly/tgn.py','published_granomaly').TGN

def kitnet(root):
    import numpy as np
    if not hasattr(np,'Inf'):np.Inf=np.inf
    folder=str(Path(root)/'kitnet');sys.path.insert(0,folder)
    try:return module(Path(folder)/'KitNET.py','published_kitnet').KitNET
    finally:sys.path.remove(folder)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('.references'));p.add_argument('--manifest',type=Path)
    a=p.parse_args();records=fetch(a.output)
    if a.manifest:a.manifest.parent.mkdir(parents=True,exist_ok=True);a.manifest.write_text(json.dumps(records,indent=2))
    print(json.dumps(dict(reviewed_modules=len(records),hashes_verified=True)))
