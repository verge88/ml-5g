"""Require every registered domain and preserve different evidence units."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .common import P,MASTER_SHA,provenance,write_csv,write_json


def interval(values):
    a=np.asarray(values,dtype=float)
    rng=np.random.default_rng(20261029)
    draws=a[rng.integers(len(a),size=(2000,len(a)))].mean(axis=1)
    return {"mean":float(a.mean()),"lo":float(np.quantile(draws,.025)),"hi":float(np.quantile(draws,.975))}


def simulation(rows,metrics):
    names=set(rows[0]["models"])
    assert all(set(r["models"])==names for r in rows)
    result={name:{k:interval([r["models"][name][k] for r in rows]) for k in metrics} for name in sorted(names)}
    if rows[0]["suite_domain"]=="complex":
        for name in names:
            result[name]["per_kind"]={kind:{k:interval([r["models"][name]["per_kind"][kind][k] for r in rows]) for k in ("recall","paired_detection_gain","restricted_delay")} for kind in rows[0]["models"][name]["per_kind"]}
            assert result[name]["per_kind"]["invisible"]["paired_detection_gain"]["mean"]==0
    return result


def run(root,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    domains={k:[] for k in ("basic","complex","external","native")}
    for path in Path(root).rglob("summary.json"):
        r=json.loads(path.read_text())
        if r.get("suite_domain") in domains:
            assert r["master_protocol_sha256"]==MASTER_SHA
            assert r["code_commit"]==os.environ["GITHUB_SHA"]
            domains[r["suite_domain"]].append(r)
    assert sorted(r["seed"] for r in domains["basic"])==P["basic_seeds"]
    assert sorted(r["seed"] for r in domains["complex"])==P["complex_seeds"]
    assert sorted(r["replica"] for r in domains["native"])==P["native_replicas"]
    assert len(domains["external"])==1
    base=simulation(domains["basic"],["observable_event_recall","profile_valid_observable_recall","fpr","restart_fpr","drift_fpr","restricted_delay_ticks"])
    rich=simulation(domains["complex"],["observable_event_recall","behavior_event_recall","heldout_recall","point_fpr","benign_episode_alarm_rate","behavior_paired_gain","restricted_delay"])
    sched={kind:{mode:{k:interval([r["scheduler"][mode]["per_kind"][kind][k] for r in domains["complex"]]) for k in ("recall","restricted_delay")} for mode in P["scheduler_modes"]} for kind in ("integrity_only","combined")}
    for r in domains["complex"]:assert len({s["checks"] for s in r["scheduler"].values()})==1
    native={};normal_coverage={}
    for name in domains["native"][0]["models"]:
        rr=[r["models"][name] for r in domains["native"]]
        events=[e for r in rr for e in r["episodes"]]
        points=sum(r["normal_points"] for r in rr);fp=sum(r["false_points"] for r in rr)
        native[name]={"events":len(events),"detected":sum(e["detected"] for e in events),"normal_points":points,"false_points":fp,"fpr":fp/points,
                      "stop_detected":sum(e["detected"] for e in events if e["kind"]=="stop"),"pause_detected":sum(e["detected"] for e in events if e["kind"]=="pause"),
                      "restricted_delay_seconds":float(np.mean([e["restricted_delay_seconds"] for e in events]))}
    for r in domains["native"]:
        for name,count in r["normal_monitor_states"].items():normal_coverage[name]=normal_coverage.get(name,0)+count
        assert r["unknown_is_not_clear"]
    cu=[json.loads(p.read_text()) for p in Path(root).rglob("evaluation.json")]
    cu=[r for r in cu if "verifier_git_blob" in r]
    assert sorted(r["replica"] for r in cu)==P["native_replicas"]
    assert all(r["commit"]==os.environ["GITHUB_SHA"] and all(r["controls"].values()) for r in cu)
    sampler=[]
    for path in Path(root).rglob("sampler-accepted-cus.json"):sampler.extend(json.loads(path.read_text()))
    registration=list(Path(root).rglob("registration.log"));assert len(registration)==1 and "All tests passed." in registration[0].read_text(errors="replace")
    protocol_counts=(registration[0].parent/"protocol-counts.txt").read_text()
    counts={"basic_test_runs":sum(len(r["run_ids"]["test"]) for r in domains["basic"]),"basic_episodes":sum(r["models"]["integrity"]["events"] for r in domains["basic"]),
            "complex_test_runs":sum(r["test_runs"] for r in domains["complex"]),"complex_episodes":sum(r["test_episodes"] for r in domains["complex"]),"complex_negative_controls":len(domains["complex"])*4,
            "native_availability_test_episodes":12,"native_deployments":3,"cu_base_verified_responses":sum(r["native_cu_responses_verified"] for r in cu),
            "cu_continuous_unique_authenticated_responses":len(sampler),"cu_authorized_restarts":sum(len(r["real_restarts"]) for r in cu),
            "cu_control_conditions":sum(len(r["controls"]) for r in cu),"basic_models_and_hybrids":len(base),"complex_models_calibrations_and_hybrids":len(rich),"native_models_and_hybrids":len(native)}
    comparisons={}
    for suffix in ("","@episode"):
        for model in ("hgb_rich","mlp_rich","if_rich"):
            for reference in P["paired_statistical_references"]:
                comparisons[model+suffix+"_minus_"+reference]=interval([r["models"][model+suffix]["behavior_paired_gain"]-r["models"][reference+suffix]["behavior_paired_gain"] for r in domains["complex"]])
    result={**provenance("complete_report"),"counts":counts,"basic":base,"complex":rich,"complex_scheduler":sched,"native":native,
            "native_normal_integrity_states":normal_coverage,"external":domains["external"][0],"paired":comparisons,
            "native_registration_protocol_counts":protocol_counts,"component_records":domains,"cu_records":cu,
            "limits":["Domains have different truth, features and units; no pooled rank","Live faults are availability, offline copy controls are not attacks","Shared-host and NF-origin witness trust assumptions remain","No unimplemented architecture results","Empirical calibration, no guaranteed test FPR","Bootstrap intervals resample ten simulator seeds only"]}
    write_json(out/"summary.json",result)
    write_csv(out/"basic-comparison.csv",[{"model":m,**{k:v["mean"] for k,v in s.items()}} for m,s in base.items()])
    write_csv(out/"complex-comparison.csv",[{"model":m,**{k:v["mean"] for k,v in s.items() if k!="per_kind"}} for m,s in rich.items()])
    write_csv(out/"native-comparison.csv",[{"model":m,**s} for m,s in native.items()])
    write_csv(out/"scheduler-comparison.csv",[{"kind":kind,"mode":mode,**{k:v["mean"] for k,v in s.items()}} for kind,modes in sched.items() for mode,s in modes.items()])
    lines=["# Complete registered method comparison","",f"Code: {os.environ['GITHUB_SHA']}",f"Master protocol SHA256: {MASTER_SHA}","",json.dumps(counts),"",
           "No real/synthetic pooled ranking. Different domains have different labels, features and evidence scope.","","## Basic behavior simulation","",
           "|Method|Profile-valid observable recall|Point FPR|","|---|---:|---:|"]
    for name in P["basic_methods"]:lines.append(f"|{name}|{base[name]['profile_valid_observable_recall']['mean']:.2%}|{base[name]['fpr']['mean']:.2%}|")
    lines += ["","## Rich behavior simulation: block-calibrated operating point","","|Method|Behavior recall|Held-out recall|Benign matched interval alarms|Paired gain|","|---|---:|---:|---:|---:|"]
    for name in P["complex_methods"]:
        key=name if name=="integrity" else name+"@episode";s=rich[key]
        lines.append(f"|{name}|{s['behavior_event_recall']['mean']:.2%}|{s['heldout_recall']['mean']:.2%}|{s['benign_episode_alarm_rate']['mean']:.2%}|{s['behavior_paired_gain']['mean']:.2%}|")
    lines += ["","## Native availability on real responses","","|Method|Detected episodes|False normal observations|","|---|---:|---:|"]
    for name in P["native_methods"]:
        s=native[name];lines.append(f"|{name}|{s['detected']}/{s['events']}|{s['false_points']}/{s['normal_points']} ({s['fpr']:.2%})|")
    lines += ["","Native CU integrity states are reported separately; availability is not the same label as integrity.",json.dumps(normal_coverage),"",
              "Full tables include all hybrids, both rich calibration modes, per-family results, sequential models, transfer folds and scheduler costs.",
              "Native capture was online, model evaluation retrospective. Continuous CU requests and independent-signer witnesses were collected during capture.",
              "Witness runtime UUID export originates in the NF and all components share a host. No hardware attestation or hostile NF compromise claim.",
              "TGN/GNN/LSTM/Transformer are not implemented and have no attributed performance.",
              "Unmodified official registration suite passed separately; no attack effects are inferred from that suite.",
              "Model superiority is not required for CI success. Negative results and convergence warnings remain in summary.json."]
    report="\n".join(lines)+"\n";(out/"REPORT.md").write_text(report);print(report)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output",required=True);a=p.parse_args();run(a.input,a.output)
