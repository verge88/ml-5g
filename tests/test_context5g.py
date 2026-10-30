import copy
import json
import numpy as np
import pytest
from neural5g.data import Stream
from context5g.data import ContextBank,NUMERIC,FEATURES,P,prepare
from context5g.benchmark import evaluate,operating_points,record_points,main


def rows(n=40,positive=False):
    result=[]
    for i in range(n):
        label="udm_subscriber_data_curl" if positive and i%4==0 else "normal"
        result.append(dict(source="observed-peer",destination="observed-service",method="GET",
            path="/service/v1/{subscriber}" if i%3 else "/other/v1/{uuid}",time=float(i),
            label=label,family="other",packet_row=i))
    return result


def stream(rr,name):
    rng=np.random.default_rng(3)
    return Stream(rng.normal(size=(len(rr),20)).astype(np.float32),np.zeros(len(rr),dtype=np.int64),
        np.ones(len(rr),dtype=np.int64),np.array([r["time"] for r in rr]),
        np.array([r["label"]!="normal" for r in rr],dtype=np.float32),rr,name)


def fake_prepare(*args):
    train=stream(rows(120),"normal_train");source=stream(rows(80,True),"source")
    cal=stream(rows(100),"cal");target=stream(rows(60,True),args[2]);holdout=stream(rows(60),"holdout")
    edge={(r["source"],r["destination"]) for r in train.rows}
    return (train,source),cal,{args[2]:target,"normal_holdout":holdout},dict(n_nodes=2,input_dim=20),(edge,set())


def test_no_future_event_changes_current_or_previous_context():
    rr=rows();bank=ContextBank(rr[:20]);original=bank.score(rr)
    changed=copy.deepcopy(rr);changed[25]["destination"]="different-peer"
    other=bank.score(changed)
    np.testing.assert_array_equal(original[:25],other[:25])
    assert not np.array_equal(original[25:],other[25:])


def test_context_ignores_labels_response_ids_and_absolute_epoch():
    rr=rows();bank=ContextBank(rr[:20]);original=bank.score(rr)
    changed=copy.deepcopy(rr)
    for r in changed:
        r.update(label="made-up",status=500,packet_row=100000,frame=900,token="ignored")
        r["time"]+=2000000000
    np.testing.assert_allclose(original,bank.score(changed),rtol=0,atol=0)
    assert original.shape==(len(rr),len(FEATURES))


def test_test_scoring_cannot_update_frozen_transition_counts():
    bank=ContextBank(rows(20));before=copy.deepcopy((bank.counts,bank.totals,bank.edge_op))
    new=rows(10)
    for r in new:r["path"]="/never-observed"
    bank.score(new)
    assert before==(bank.counts,bank.totals,bank.edge_op)


def test_absent_primary_examples_remain_unidentifiable():
    s=stream(rows(8,True),"target")
    result=evaluate(s,np.ones(len(s)),.5,(set(),set()))
    primary=result["scopes"]["data_access_known_edge"]
    assert primary["requests"]==0 and primary["recall"] is None
    assert primary["status"]=="NOT_IDENTIFIABLE_NO_EXAMPLES"


def test_policy_budget_failure_is_not_hidden_by_quantile_ties():
    cal=np.ones(100)
    points=operating_points(cal,policy=True)
    assert points["0.001"]["threshold"]==.5 and points["0.001"]["calibration_fpr"]==1
    s=stream(rows(100),"normal")
    outputs=record_points(cal,{"target":cal,"normal_holdout":cal},
        {"target":s,"normal_holdout":s},(set(),set()),points)
    assert not outputs[0]["test_budget_met"] and not outputs[0]["calibration_budget_met"]


def test_target_values_cannot_change_normal_context_scaler(monkeypatch):
    import context5g.data as module
    monkeypatch.setattr(module,"base_prepare",fake_prepare)
    _,_,before,_=prepare(None,"source","target")
    def changed(*args):
        training,cal,tests,info,banks=fake_prepare(*args)
        for r in tests[args[2]].rows:r["path"]="/entirely-new"
        return training,cal,tests,info,banks
    monkeypatch.setattr(module,"base_prepare",changed)
    views,_,after,_=prepare(None,"source","target")
    assert before["context_center"]==after["context_center"] and before["context_scale"]==after["context_scale"]
    assert all(s.time[0]==0 for s in views["metadata_api_context"].values())


def test_hgb_rules_and_report_end_to_end(monkeypatch,tmp_path):
    import context5g.data as module
    from context5g.report import run
    monkeypatch.setattr(module,"base_prepare",fake_prepare)
    monkeypatch.setitem(P,"hgb_iterations",3)
    (tmp_path/"manifest.json").write_text("[]")
    main(tmp_path,tmp_path/"context-hgb","hgb")
    main(tmp_path,tmp_path/"context-rules","rules")
    # The report's family count guard is intentionally independent of fit internals.
    summary=json.loads((tmp_path/"context-hgb/summary.json").read_text())
    assert len(summary["records"])==12
    for r in summary["records"]:
        folder=tmp_path/"context-hgb"/r["fold"]/r["input"]/str(r["seed"])
        frozen=json.loads((folder/"frozen-before-evaluation.json").read_text())
        assert "operating_points" not in frozen
        for alpha,point in frozen["points"].items():
            scores=np.load(folder/"calibration.npy")
            assert point["threshold"]==np.quantile(scores,1-float(alpha),method="higher")
    for model in ("tgn","gnn","lstm","transformer"):
        target=tmp_path/f"context-{model}";target.mkdir()
        surrogate=copy.deepcopy(summary);surrogate["model"]=model
        for r in surrogate["records"]:r["model"]=model
        (target/"summary.json").write_text(json.dumps(surrogate))
    # report.py reads the file protocol, so the smoke override must be restored there too.
    import context5g.report as reporter
    monkeypatch.setattr(reporter,"P",P)
    final=run(tmp_path,tmp_path/"report")
    assert final["primary_status"]=="NOT_IDENTIFIABLE_NO_EXAMPLES"
    assert all(h["status"]=="NOT_IDENTIFIABLE_NO_EXAMPLES" for h in final["hypotheses"] if h["scope"]==P["primary_scope"])
