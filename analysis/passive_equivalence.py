"""Post-registration structural audit; no test outcomes or fitted models used."""
import hashlib
import json
from pathlib import Path

import numpy as np

from probe_calibrated.simulator import PLAN,FEATURES,cells,simulate

assert hashlib.sha256(Path('probe_calibrated/simulator.py').read_bytes()).hexdigest()=='a9cde44ea356911eb38d99afd24654ba8813332a96f82a61ebb1a872bbd39c11'
records=[]
for seed in range(88301,88317):
    for cell in cells():
        if cell['policy']!='passive':continue
        a,x,y,_=simulate(seed,cell,1)
        b,z,w,_=simulate(seed,cell,2)
        assert a['queries']==b['queries']==0
        assert np.array_equal(x,z)
        assert a['user_p95']==b['user_p95']
        assert int((y!=w).sum())==120
        records.append(dict(seed=seed,**cell,identical_features=True,different_labels=120,
                            user_p95=a['user_p95']))
out=Path('out/passive-equivalence');out.mkdir(parents=True,exist_ok=True)
result=dict(scope='Structural counterexample in the registered synthetic simulator; not a claim about all passive 5G telemetry',
    simulator_sha256=hashlib.sha256(Path('probe_calibrated/simulator.py').read_bytes()).hexdigest(),
    compared_pairs=len(records),simulated_trajectories=2*len(records),development_seeds=16,
    feature_names=FEATURES,all_feature_matrices_identical=True,records=records,
    conditional_fact='If mechanisms path/application have equal prior probabilities and these identical observations are the only inputs, no classifier can exceed 1/2 average per-decision identification accuracy on that pair.',
    explanation='No probes are sent. Both mechanisms add the same constant to each user completion; all arrival, completion and query counters and all optional masks are then identical. Detection of a deviation and identification of its cause remain distinct tasks.',
    novelty='Elementary indistinguishability argument, not a new theorem or real-network discovery.')
(out/'equivalence.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['records','feature_names']},indent=2))
