"""Explicit feature allowlist: no plan, actuator, phase or deployment identifiers."""
import hashlib,json
from pathlib import Path
import numpy as np

PLAN=json.loads(Path(__file__).with_name('plan.json').read_text())
CLASSES=['healthy','user_path','application','combined']


def features(row):
    m=row['measured']
    return np.array([float(m['end_ms']),float(m['bypass_ms'])],dtype=float)


def eligible(row,require_bypass=False):
    m=row['measured'];cu=row['cu']
    ok=(m['end_signed']>=2 and m['end_ms'] is not None and np.isfinite(m['end_ms']) and
        not cu['hard_alarm'] and cu['witness_available'] and max(cu['ages'].values())<=PLAN['maximum_cu_age_seconds'])
    if require_bypass:ok=ok and m['bypass_signed']>=2 and m['bypass_ms'] is not None and np.isfinite(m['bypass_ms'])
    return bool(ok)


def load(root,ids):
    records=[];provenance=[]
    expected=hashlib.sha256(Path(__file__).with_name('plan.json').read_bytes()).hexdigest()
    for replica in ids:
        folder=Path(root)/f'diagnostic-native-{replica}'
        capture=json.loads((folder/'diagnostic-capture.json').read_text())
        assert capture['replica']==replica and capture['plan_sha256']==expected
        for e in capture['trials']:
            p=folder/'trials'/e['name']/'observations.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
            assert len(rows)==PLAN['ticks']
            assert [r['tick'] for r in rows]==list(range(PLAN['ticks']))
            records.append(dict(replica=replica,name=e['name'],label=CLASSES.index(e['family']),rows=rows))
            provenance.append(dict(replica=replica,trial=e['name'],sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    return records,provenance


def steady(records):
    xx=[];yy=[]
    for trial in records:
        for row in trial['rows'][PLAN['steady_start']:]:
            if eligible(row,True):xx.append(features(row));yy.append(trial['label'])
    return np.asarray(xx),np.asarray(yy)
