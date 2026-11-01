"""Offline invariant cases; not traffic sent to a core and not attack results."""
import argparse,json
from pathlib import Path
from .evidence import authorization_contract

def cases():
    grant=dict(actors=['authorized-function'],subscriber='test-subscriber',context='test-context',dnns=['internet'],slices=[1],operations=['create','release'])
    good=dict(actor='authorized-function',subscriber='test-subscriber',context='test-context',dnn='internet',slice=1,operation='create')
    rows=[]
    for operation in ['create','release']:
        e=dict(good,operation=operation);rows.append(dict(case='permitted-'+operation,event=e,expected='PERMITTED_IN_DECLARED_CONTRACT',actual=authorization_contract(e,grant)))
    for field,bad in [('actor','other-function'),('subscriber','other-subscriber'),('context','other-context'),('dnn','other-network'),('slice',2),('operation','other-operation')]:
        e=dict(good);e[field]=bad;rows.append(dict(case='inconsistent-'+field,event=e,expected='VIOLATION',actual=authorization_contract(e,grant)))
    for field in good:
        e=dict(good);del e[field];rows.append(dict(case='missing-'+field,event=e,expected='UNKNOWN',actual=authorization_contract(e,grant)))
    assert all(r['actual']==r['expected'] for r in rows)
    return dict(scope='14 offline authorization-contract cases, no requests sent to a core; not evidence of semantic attack detection',grant=grant,cases=rows,passed=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);(a.output/'offline-controls.json').write_text(json.dumps(cases(),indent=2))
