import copy
import numpy as np
from diagnose5g.data import PLAN,eligible,features
from diagnose5g.profiles import trials,configuration,command
from diagnose5g.policy import GaussianDiagnostic,ProbePolicy
from diagnose5g.evaluate import replay


def fixture_model():
    rng=np.random.default_rng(7);means=[[10,5],[130,5],[130,125],[130,65]]
    x=np.vstack([rng.normal(m,2,(20,2)) for m in means]);y=np.repeat(np.arange(4),20)
    return GaussianDiagnostic().fit(x,y)


def row(t,end=130.,bypass=125.):
    return dict(tick=t,monotonic_ns=t*1_000_000_000,measured=dict(end_ms=end,bypass_ms=bypass,end_signed=3,bypass_signed=2),
        cu=dict(hard_alarm=False,witness_available=True,ages=dict(ausf=1.,nssf=1.)))


def test_feature_allowlist_and_missingness():
    a=row(0);b=copy.deepcopy(a);b.update(family='healthy',actuator=999,future=0,replica=99)
    assert np.array_equal(features(a),features(b))
    b['measured']['bypass_ms']=None;b['measured']['bypass_signed']=0
    assert eligible(b) and not eligible(b,True)
    b['cu']['ages']['ausf']=10;assert not eligible(b)


def test_bounds_and_disjoint_splits():
    groups=[set(PLAN[k]) for k in ['train','calibration','test']]
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    for n in sum([PLAN[k] for k in ['train','calibration','test']],[]):
        assert len(set(trials(n)))==8
        for family,repeat in trials(n):
            cfg=configuration(n,family,repeat)
            for t in range(40):
                q=command(cfg,t);assert 0<=q['delay_ms']<=80 and 0<=q['app_wait_ms']<=150
            assert command(cfg,22)==command(cfg,39)


def test_distinguishable_only_after_optional_measurement():
    m=fixture_model()
    assert m.posterior(130).max()<.5
    assert m.posterior(130,125).argmax()==2 and m.posterior(130,5).argmax()==1
    assert m.expected_risk_reduction(m.posterior(130))>.4
    assert m.expected_risk_reduction(m.posterior(10))<.05


def test_budget_and_failed_replies_do_not_refresh():
    p=ProbePolicy('expected_risk',budget=2)
    assert p.choose(0,0,.5)[0]
    assert p.last is None
    assert p.choose(1,1,.5)[0]
    assert not p.choose(30,30,.5)[0] and p.used==2


def test_unselected_optional_values_are_invisible():
    m=fixture_model();a=dict(replica=1,name='unit',rows=[row(t) for t in range(40)])
    baseline=replay(a,m,'fixed',.8);b=copy.deepcopy(a)
    for i,r in enumerate(b['rows']):
        if not baseline['records'][i]['queried']:
            r['measured'].update(bypass_ms=1e6,bypass_signed=0)
    assert replay(b,m,'fixed',.8)==baseline


def test_future_changes_do_not_change_past_decisions():
    m=fixture_model();a=dict(replica=1,name='unit',rows=[row(t) for t in range(40)]);b=copy.deepcopy(a)
    for r in b['rows'][25:]:r['measured'].update(end_ms=1.,bypass_ms=2.)
    for mode in ['fixed','oldest','expected_risk']:
        aa=replay(a,m,mode,.8);bb=replay(b,m,mode,.8)
        assert aa['records'][:25]==bb['records'][:25] and aa['queries']<=10 and bb['queries']<=10


def test_stale_optional_is_not_treated_as_current():
    m=fixture_model();a=dict(replica=1,name='unit',rows=[row(t) for t in range(40)])
    for r in a['rows'][1:]:r['measured'].update(bypass_signed=0,bypass_ms=None)
    result=replay(a,m,'fixed',.8)
    assert not result['records'][8]['optional_fresh']
    assert result['records'][8]['confidence']<.5


def test_fit_freeze_and_evaluate_pipeline_with_unit_fixture(tmp_path):
    # Synthetic fixture verifies plumbing only, never scientific evidence.
    import hashlib,json
    from pathlib import Path
    from diagnose5g.train import run as train
    from diagnose5g.evaluate import run as evaluate
    root=tmp_path/'inputs';models=tmp_path/'models';out=tmp_path/'results'
    digest=hashlib.sha256(Path(__file__).parents[1].joinpath('diagnose5g/plan.json').read_bytes()).hexdigest()
    means=[(10,5),(130,5),(130,125),(130,65)]
    for replica in sum([PLAN[k] for k in ['train','calibration','test']],[]):
        folder=root/f'diagnostic-native-{replica}';folder.mkdir(parents=True)
        entries=[]
        for family,repeat in trials(replica):
            label=PLAN['families'].index(family);name=f'{family}-{repeat}';p=folder/'trials'/name;p.mkdir(parents=True)
            rows=[row(t,*(means[label] if t>=22 else means[0])) for t in range(40)]
            (p/'observations.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            entries.append(dict(name=name,family=family,repeat=repeat))
        (folder/'diagnostic-capture.json').write_text(json.dumps(dict(replica=replica,trials=entries,plan_sha256=digest)))
    train(root,models);meta=json.loads((models/'frozen.json').read_text())
    assert {p['replica'] for p in meta['train_source']}==set(PLAN['train'])
    assert not set(PLAN['test'])&{p['replica'] for p in meta['train_source']+meta['calibration_source']}
    evaluate(root,models,out);result=json.loads((out/'results.json').read_text())
    assert result['models']['gaussian_both']['accuracy']==1 and result['test_model_fits']==0
    assert all(v['queries']<=160 for v in result['replay'].values())
