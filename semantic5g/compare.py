"""Align frozen packet-level ML outputs with recorded request-bearing frames.

No model fitting or threshold adjustment occurs here. OR fusion is exploratory.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .analyze import candidate, digest, rate


def run(data, frozen, output):
    summary = json.loads((frozen / "summary.json").read_text())
    records = []
    for name, fold in (("bridge_attack_nrf", "train_smf_test_nrf"), ("bridge_attack_smf", "train_nrf_test_smf")):
        with (data / (name + ".csv")).open(encoding="utf-8-sig", newline="") as f:
            packets = sorted(csv.DictReader(f), key=lambda r: float(r["frame.time_epoch"]))
        with (frozen / (name + "-predictions.csv")).open(newline="") as f:
            predictions = list(csv.DictReader(f))
        assert len(packets) == len(predictions)
        assert all(p["Label"] == q["label"] and int(q["row"]) == i for i, (p, q) in enumerate(zip(packets, predictions)))
        frame_index = {int(row["frame.number"]): i for i, row in enumerate(packets)}
        assert len(frame_index) == len(packets), "Nonunique frame identifiers"
        with (output / (name + "-requests.csv")).open(newline="") as f:
            requests = list(csv.DictReader(f))
        models = {method: (np.array([float(r[method]) for r in predictions]), threshold)
            for method, threshold in summary["thresholds"].items()}
        for method, model in summary["supervised"][fold].items():
            npz = np.load(frozen / (fold + "-" + method + "-" + name + ".npz"))
            assert np.array_equal(npz["label"], np.array([p["Label"] != "normal" for p in packets]))
            models[method] = (npz["score"], model["threshold"])
        aligned = {}
        for method, (score, threshold) in models.items():
            aligned[method] = [bool(score[frame_index[int(r["frame"])] ] > threshold) for r in requests]
        aligned["operation_policy"] = [r["edge_operation_novelty"] == "True" for r in requests]
        aligned["hgb_OR_operation_policy"] = [a or b for a, b in zip(aligned["hgb_cross_capture"], aligned["operation_policy"])]
        aligned["mlp_OR_operation_policy"] = [a or b for a, b in zip(aligned["mlp_cross_capture"], aligned["operation_policy"])]
        for scope, select in (("all_attack_label_requests", lambda r: r["label"] not in ("normal", "portscan")),
                              ("producer_candidates", candidate)):
            positive = [i for i, r in enumerate(requests) if select(r)]
            normal = [i for i, r in enumerate(requests) if r["label"] == "normal"]
            for method, flags in aligned.items():
                tp, fp = sum(flags[i] for i in positive), sum(flags[i] for i in normal)
                records.append(dict(capture=name, scope=scope, method=method, attack_requests=len(positive),
                    detected_requests=tp, recall=rate(tp, len(positive)), normal_requests=len(normal),
                    false_alerts=fp, false_alert_rate=rate(fp, len(normal))))
        with (output / (name + "-aligned-alerts.csv")).open("w", newline="", encoding="utf-8") as f:
            fields = ["frame", "label", "producer_candidate", *aligned]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i, r in enumerate(requests):
                w.writerow(dict(frame=r["frame"], label=r["label"], producer_candidate=candidate(r),
                    **{method: flags[i] for method, flags in aligned.items()}))
    with (output / "request-level-comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    result = dict(frozen_ml_commit=summary["code_commit"], frozen_summary_sha256=digest(frozen / "summary.json"),
        unit="observed request-bearing frame/hop; a packet score is assigned to its request; not packet-level original metrics",
        training="same first 60% of normal packets for operation banks and frozen ML training; thresholds unchanged",
        caveat="retrospective exploratory analysis; no confidence intervals across dependent requests or claims of new independent replications",
        results=records)
    (output / "ml-comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps([r for r in records if r["scope"] == "producer_candidates" and r["method"] in
        ("edgebank", "hgb_cross_capture", "mlp_cross_capture", "operation_policy", "hgb_OR_operation_policy")], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.data, a.frozen, a.output)
