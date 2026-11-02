import argparse,json,os
from pathlib import Path
from .data import PLAN,load
from .models import fit
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    train,pa=load(a.input,PLAN['train']);cal,pb=load(a.input,PLAN['calibration']);frozen=fit(train,cal,a.output,pa+pb,os.environ['SOURCE_COMMIT'])
    print(json.dumps(dict(training_rows=frozen['training_eligible_rows'],positive=frozen['training_positives'],models=len(frozen['models']),selected_statistical=frozen['selected_statistical'],test_access=False)))
