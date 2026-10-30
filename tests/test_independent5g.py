"""Independence, feature-leakage and measured-exposure regression controls."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from independent5g.collect import P, assignment
from independent5g.data import features, prepare, read_deployment
from independent5g.benchmark import evaluate, fit_masked
from neural5g.data import Stream
from neural5g.models import make_model
from neural5g.benchmark import seed_all, predict


def observation(value=1.):
    return {"nrf":dict(ok=True,status=200,elapsed_ms=value,received_bytes=500,body={}),
        "direct":dict(ok=True,status=400,elapsed_ms=value,received_bytes=160,body={}),
        "integrity_monitor":dict(states={"ausf":"PENDING_CURRENT_CU","nssf":"PENDING_CURRENT_CU"},
            decision={"verdict":"UNKNOWN"},witness_available=False)}


@pytest.fixture
def captures(tmp_path,monkeypatch):
    for k,v in dict(normal_ticks=4,baseline_ticks=1,fault_ticks=1,recovery_ticks=1).items():
        monkeypatch.setitem(P,k,v)
    for split,replicas in P["splits"].items():
        for replica in replicas:
            folder=tmp_path/f"independent-native-{replica}"/"native";folder.mkdir(parents=True)
            rows,events=[],[]
            phases=[("normal",None)]*P["normal_ticks"]
            if split != "calibration":
                for i,(slot,kind) in enumerate(P["fault_plan"]):
                    phases.append(("normal",None))
                    tick=len(phases)
                    events.append(dict(episode=f"fault-{i}",slot=slot,kind=kind,start_tick=tick,
                        end_tick=tick+1,start_ns=tick*1000000000,end_ns=(tick+1)*1000000000))
                    phases.extend([("fault",f"fault-{i}"),("recovery",None)])
            for tick,(phase,episode) in enumerate(phases):
                obs={s:observation(1.+replica/1000.) for s in ("ausf","nssf")}
                rows.append(dict(tick=tick,phase=phase,episode=episode,started_ns=tick*1000000000,
                    monotonic_ns=tick*1000000000+1000000,interval_end_ns=(tick+1)*1000000000,observations=obs))
            manifest=dict(replica=replica,split=split,period_seconds=assignment(replica)[1],observed_ticks=len(rows),
                protocol_sha256=hashlib.sha256(Path("independent5g/protocol.json").read_bytes()).hexdigest(),
                fresh_deployment=True,commit="fixture-only",initial_discovery={s:{"nrf":{"body":{"nfInstances":[
                    {"nfInstanceId":f"fixture-{replica}-{s}"}]}}} for s in ("ausf","nssf")})
            (folder/"capture-manifest.json").write_text(json.dumps(manifest))
            (folder/"observations.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            (folder/"events.json").write_text(json.dumps(events))
    return tmp_path


def test_disjoint_assignment_and_unknown_refused():
    ids=[x for group in P["splits"].values() for x in group]
    assert len(ids)==len(set(ids))==9
    with pytest.raises(ValueError):assignment(401)


def test_feature_whitelist_excludes_truth_identity_and_payload():
    original=observation()
    changed=copy.deepcopy(original)
    changed.update(label=1,phase="fault",replica=609,tick=9000,episode="other",unix_ns=123)
    changed["nrf"]["body"]={"nfInstances":[{"nfInstanceId":"different","customInfo":{"secret":"ignored"}}]}
    changed["integrity_monitor"]["current_request_cycle"]=9999
    for name in P["inputs"]:
        np.testing.assert_array_equal(features(original,"ausf",name),features(changed,"ausf",name))
    changed["integrity_monitor"]["witness_available"]=True
    np.testing.assert_array_equal(features(original,"ausf","metadata_api"),features(changed,"ausf","metadata_api"))
    assert not np.array_equal(features(original,"ausf","metadata_api_witness"),features(changed,"ausf","metadata_api_witness"))


def test_test_changes_cannot_change_training_scaler(captures):
    train,_,_,before=prepare(captures,"metadata")
    path=captures/"independent-native-607/native/observations.jsonl"
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    rows[-1]["observations"]["ausf"]["nrf"]["elapsed_ms"]=1e8
    path.write_text("\n".join(json.dumps(r) for r in rows))
    train2,_,_,after=prepare(captures,"metadata")
    assert before["center"]==after["center"] and before["scale"]==after["scale"]
    np.testing.assert_array_equal(train[0].x,train2[0].x)


def test_calibration_contamination_refused(captures):
    path=captures/"independent-native-604/native/events.json"
    path.write_text(json.dumps([dict(start_tick=0,end_tick=1,slot="ausf",kind="stop",episode="bad")]))
    with pytest.raises(ValueError,match="calibration"):
        read_deployment(path.parent,604,"metadata")


def test_duplicate_uuid_refused(captures):
    a=captures/"independent-native-601/native/capture-manifest.json"
    b=captures/"independent-native-607/native/capture-manifest.json"
    changed=json.loads(b.read_text());changed["initial_discovery"]=json.loads(a.read_text())["initial_discovery"]
    b.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="UUID reused"):
        prepare(captures,"metadata")


def test_metrics_exposure_not_doubled_for_two_functions(captures):
    stream=read_deployment(captures/"independent-native-607/native",607,"metadata")
    scores=np.array([1. if r["normal"] and r["slot"]=="ausf" else float(r["label"]) for r in stream.rows])
    m=evaluate(stream,scores,.5)
    assert m["recall"]==1 and m["episode_recall"]==1 and m["fpr"]==.5
    assert m["normal_seconds"]==8 and m["false_alerts_per_hour"]==3600
    assert m["false_alarm_episodes"]==4


@pytest.mark.parametrize("architecture",["tgn","gnn","lstm","transformer"])
def test_masked_training_all_modules_and_history(architecture):
    seed_all(6101);torch.set_num_threads(1)
    rng=np.random.default_rng(5);n=280
    x=rng.normal(size=(n,6)).astype(np.float32)
    y=(np.arange(n)%40>=20).astype(np.float32)
    rows=[dict(selected=i%7!=0) for i in range(n)]
    stream=Stream(x,np.zeros(n,dtype=np.int64),1+np.arange(n,dtype=np.int64)%2,
        np.arange(n,dtype=np.float64)*.1,y,rows,"fixture")
    config=dict(P,epochs=1,hidden=8,batch_size=32,sequence_length=4,attention_heads=2,temporal_neighbors=3)
    model=make_model(architecture,6,3,config)
    fitted=fit_masked(model,[stream],architecture,config)
    assert fitted["selected_training_rows"]==sum(r["selected"] for r in rows)
    score=predict(model,stream,architecture,config)
    assert score.shape==(n,) and np.isfinite(score).all()
    # A later row cannot alter earlier scores, including within a minibatch.
    changed=copy.deepcopy(stream);changed.x[-1]*=1000
    other=predict(model,changed,architecture,config)
    np.testing.assert_allclose(score[:-1],other[:-1],rtol=1e-5,atol=1e-6)


def test_all_five_models_three_inputs_and_report(captures,monkeypatch,tmp_path):
    from independent5g.benchmark import run
    from independent5g.report import run as report
    for k,v in dict(epochs=1,hidden=8,batch_size=8,sequence_length=4,attention_heads=2,
        temporal_neighbors=3,seeds=[6101],hgb_iterations=3,torch_threads=1,bootstrap_draws=30).items():
        monkeypatch.setitem(P,k,v)
    output=tmp_path/"model-smoke"
    for architecture in P["models"]:
        summary=run(captures,output/f"independent-model-{architecture}",architecture)
        assert len(summary["runs"])==3
        for composition in P["inputs"]:
            folder=output/f"independent-model-{architecture}"/composition/"6101"
            frozen=json.loads((folder/"frozen-before-test.json").read_text())
            assert "results" not in frozen
            assert frozen["calibration_observed_fpr"]<=.01
            assert len(list(folder.glob("test-*.npz")))==3
    rows=report(output,tmp_path/"fixture-report")
    assert len(rows)==15 and all(r["test_deployments"]==3 for r in rows)


def test_collector_schedule_and_fail_safe_cleanup(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from live5g import capture
    from independent5g.collect import capture_run
    for k,v in dict(normal_ticks=2,baseline_ticks=1,fault_ticks=2,recovery_ticks=1,
        period_seconds=[0.,0.,0.]).items():
        monkeypatch.setitem(P,k,v)
    calls=[]
    async def observe():return {s:observation() for s in ("ausf","nssf")}
    monkeypatch.setattr(capture,"observe",observe)
    monkeypatch.setattr(capture,"wait_ready",observe)
    monkeypatch.setattr(capture,"command",lambda *args:calls.append(args))
    def drain(coroutine):
        # Fake observations and zero-period sleep need no OS event loop or sockets.
        try:
            while True:
                assert coroutine.send(None) is None
        except StopIteration:
            pass
    folder=tmp_path/"collector"
    drain(capture_run(SimpleNamespace(output=folder,replica=601)))
    parsed=read_deployment(folder,601,"metadata")
    assert int(parsed.y.sum())==8
    assert calls[-2:]==[("unpause","ausf"),("unpause","nssf")]
    async def fail():raise RuntimeError("controlled observer failure")
    monkeypatch.setattr(capture,"observe",fail)
    with pytest.raises(RuntimeError,match="controlled observer failure"):
        drain(capture_run(SimpleNamespace(output=tmp_path/"failed",replica=601)))
    assert calls[-2:]==[("unpause","ausf"),("unpause","nssf")]
