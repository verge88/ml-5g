"""Audit every saved prediction against original requests, then aggregate seeds."""
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from semantic5g.analyze import candidate, read_csv, signatures

P = json.loads(Path(__file__).with_name("protocol.json").read_text())


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def normal_partitions(data):
    with (data/"bridge_normal.csv").open(encoding="utf-8-sig", newline="") as f:
        packets=sorted(enumerate(csv.DictReader(f)), key=lambda r: float(r[1]["frame.time_epoch"]))
    te,ce=int(.6*len(packets)),int(.8*len(packets))
    train={i for i,r in packets[:te]}
    calibration={i for i,r in packets[te+P["embargo_packets"]:ce]}
    holdout={i for i,r in packets[ce+P["embargo_packets"]:]}
    requests,_=read_csv(data/"bridge_normal.csv");requests.sort(key=lambda r:r["time"])
    bank={signatures(r)[0] for r in requests if r["packet_row"] in train}
    return [r for r in requests if r["packet_row"] in calibration], [r for r in requests if r["packet_row"] in holdout],bank


def audit_prediction(rows, scores, threshold, result, bank):
    flags=scores>threshold
    normal=np.array([r["label"]=="normal" for r in rows])
    positive={"all_request_labels":~normal,
        "semantic_label_requests":np.array([r["label"] not in ("normal","portscan") for r in rows]),
        "producer_candidates":np.array([candidate(r) for r in rows])}
    positive["candidate_on_known_edge"]=positive["producer_candidates"] & np.array([signatures(r)[0] in bank for r in rows])
    for scope, selected in positive.items():
        expected=result[scope]
        tp,fp,p,n=int(flags[selected].sum()),int(flags[normal].sum()),int(selected.sum()),int(normal.sum())
        if (tp,fp,p,n)!=(expected["detected_requests"],expected["false_alerts"],expected["positive_requests"],expected["normal_requests"]):
            raise ValueError("Prediction audit failed: "+scope)
        if p and not np.isclose(expected["recall"],tp/p):raise ValueError("Recall audit failed")
        if n and not np.isclose(expected["false_alert_rate"],fp/n):raise ValueError("FPR audit failed")


def run(root, output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    expected_sha=hashlib.sha256(Path(__file__).with_name("protocol.json").read_bytes()).hexdigest()
    data=root/"neural-pinned-data"
    manifest=json.loads((data/"manifest.json").read_text())
    for item in manifest:
        if hashlib.sha256((data/item["name"]).read_bytes()).hexdigest()!=item["sha256"]:
            raise ValueError("Source file digest mismatch")
    calibration,holdout,bank=normal_partitions(data)
    captures={"normal_holdout":holdout}
    for name in ("bridge_attack_nrf.csv","bridge_attack_smf.csv"):
        requests,_=read_csv(data/name);requests.sort(key=lambda r:r["time"]);captures[name]=requests
    flat,runs,verified_predictions,verified_thresholds=[],[],0,0
    for architecture in P["models"]:
        model_root=root/("neural-"+architecture)
        summary=json.loads((model_root/"summary.json").read_text())
        if summary["protocol_sha256"]!=expected_sha:raise ValueError("Mixed protocol")
        expected=set(itertools.product([f[0] for f in P["folds"]],P["inputs"],P["seeds"]))
        observed={(r["fold"],r["input"],r["seed"]) for r in summary["runs"]}
        if observed!=expected or len(summary["runs"])!=len(expected):raise ValueError("Missing/duplicate fits")
        for record in summary["runs"]:
            directory=model_root/record["fold"]/record["input"]/str(record["seed"])
            threshold=record["threshold"]
            with np.load(directory/"calibration-scores.npz",allow_pickle=False) as saved:
                if not np.array_equal(saved["raw_row"],np.array([r["packet_row"] for r in calibration])):
                    raise ValueError("Calibration alignment mismatch")
                actual=float(np.quantile(saved["score"],.99,method="higher"))
                if actual!=threshold:raise ValueError("Unreproducible threshold")
                if float((saved["score"]>actual).mean())>.01:raise ValueError("Calibration budget violated")
            verified_thresholds+=1
            for capture,result in record["results"].items():
                rows=captures[capture]
                with np.load(directory/(capture.replace(".csv","")+"-scores.npz"),allow_pickle=False) as saved:
                    if not np.isfinite(saved["score"]).all():raise ValueError("Non-finite scores")
                    if not np.array_equal(saved["raw_row"],np.array([r["packet_row"] for r in rows])):raise ValueError("Row mismatch")
                    if not np.array_equal(saved["frame"],np.array([r["frame"] for r in rows])):raise ValueError("Frame mismatch")
                    if not np.array_equal(saved["label"],np.array([r["label"]!="normal" for r in rows])):raise ValueError("Label mismatch")
                    audit_prediction(rows,saved["score"],threshold,result,bank)
                    verified_predictions+=len(rows)
                for scope in ("all_request_labels","semantic_label_requests","producer_candidates","candidate_on_known_edge"):
                    flat.append(dict(architecture=architecture,fold=record["fold"],input=record["input"],seed=record["seed"],
                        capture=capture,scope=scope,threshold=threshold,parameters=record["parameters"],
                        train_seconds=record["train_seconds"],inference_seconds_per_request=record["inference_seconds_per_request"][capture],**result[scope]))
            if architecture=="tgn" and not all(record["training"]["max_gradient_l1_by_module"].get(k,0)>0 for k in ("updater","message","attention","time_encoder")):
                raise ValueError("TGN updater or attention not trained")
            runs.append(record)
    grouped=defaultdict(list)
    for row in flat:
        grouped[tuple(row[k] for k in ("architecture","fold","input","capture","scope"))].append(row)
    aggregates=[]
    for keys,rows in sorted(grouped.items()):
        result=dict(zip(("architecture","fold","input","capture","scope"),keys),initializations=len(rows),
            positive_requests=rows[0]["positive_requests"],normal_requests=rows[0]["normal_requests"])
        for field in ("detected_requests","false_alerts","recall","false_alert_rate","pr_auc","roc_auc","train_seconds","inference_seconds_per_request"):
            values=[r[field] for r in rows if r[field] is not None]
            result[field+"_mean"]=float(np.mean(values)) if values else None
            result[field+"_std"]=float(np.std(values,ddof=1)) if len(values)>1 else None
        aggregates.append(result)
    write_csv(output/"per-initialization.csv",flat)
    write_csv(output/"aggregate.csv",aggregates)
    audit=dict(model_fits=len(runs),verified_thresholds=verified_thresholds,verified_prediction_requests=verified_predictions,
        verified_metric_rows=len(flat),source_files_verified=len(manifest),protocol_sha256=expected_sha,
        seed_scope="Initialization variability only; two capture sessions are not independent population replications")
    (output/"audit.json").write_text(json.dumps(audit,indent=2))
    (output/"summary.json").write_text(json.dumps(dict(audit=audit,runs=runs,aggregates=aggregates),indent=2))
    lines=["# Causal temporal and graph models on recorded 5G SBI requests","",
        "Five model families, two feature sets, two cross-capture folds and three initializations: 60 fits.",
        "All request-label and producer-candidate denominators are retained separately. No attack execution or authorization proof.",
        "Standard deviations quantify initialization variability, not confidence across independent attacks.","",
        "| Model | Features | Transfer | Candidate recall mean | Normal false-alert rate mean | Known-edge candidate recall mean |",
        "|---|---|---|---:|---:|---:|"]
    for row in aggregates:
        if row["scope"]!="producer_candidates" or row["capture"]=="normal_holdout":continue
        hard=next(r for r in aggregates if all(r[k]==row[k] for k in ("architecture","fold","input","capture")) and r["scope"]=="candidate_on_known_edge")
        fmt=lambda x:"n/a" if x is None else f"{100*x:.2f}%"
        lines.append(f"| {row['architecture']} | {row['input']} | {row['fold']} | {fmt(row['recall_mean'])} | {fmt(row['false_alert_rate_mean'])} | {fmt(hard['recall_mean'])} |")
    (output/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(audit),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();run(a.input,a.output)
