"""Behavior-only evaluation on pinned Open5GS packet captures.

No synthetic NFProfile/witness is joined to these traces. Never use Label,
textual info, capture ID, frame number or absolute epoch as a model input.
"""
import argparse
from collections import Counter, deque
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from .benchmark import dump_csv, threshold
from .fetch_data import git_blob_sha
from .simulation import PROTOCOL

FEATURES = ["log_packet_length", "log_http_header_length", "log_http_header_count",
            "is_tcp", "is_udp", "is_sctp", "is_http2", "is_ngap", "well_known_src_port", "well_known_dst_port",
            "log_past32_mean_length", "log_interarrival", "source_surprise", "destination_surprise", "pair_surprise", "new_pair"]


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.


def read(path):
    expected = PROTOCOL["external_dataset"]["files"][Path(path).name]
    assert git_blob_sha(Path(path).read_bytes()) == expected
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    # Order each capture independently; captures come from different sessions.
    return sorted(rows, key=lambda r: number(r["frame.time_epoch"]))


def memory(rows):
    return (Counter(r["ip.src"] for r in rows), Counter(r["ip.dst"] for r in rows),
            Counter((r["ip.src"], r["ip.dst"]) for r in rows))


def features(rows, reference):
    src, dst, pair = reference
    total = sum(src.values())
    prior_lengths = deque(maxlen=32)
    previous = None
    result = []
    for row in rows:
        a, b = row["ip.src"], row["ip.dst"]
        length = max(0, number(row["frame.len"]))
        stamp = number(row["frame.time_epoch"])
        gap = max(0, stamp - previous) if previous is not None else 0
        protocols = set(row["frame.protocols"].split(":"))
        vals = [np.log1p(length), np.log1p(max(0, number(row["http2.header.length"]))),
                np.log1p(max(0, number(row["http2.header.count"])))]
        vals += [float(p in protocols) for p in ("tcp", "udp", "sctp", "http2", "ngap")]
        vals += [float(0 < number(row[k]) <= 1024) for k in ("srcport", "dstport")]
        vals += [np.log1p(np.mean(prior_lengths) if prior_lengths else 0), np.log1p(gap)]
        vals += [-np.log((counter[key] + 1) / (total + len(counter) + 1))
                 for counter, key in ((src, a), (dst, b), (pair, (a, b)))]
        vals += [float((a, b) not in pair)]
        result.append(vals)
        prior_lengths.append(length)
        previous = stamp
    return np.asarray(result)


def labels(rows):
    return np.array([r["Label"].strip().lower() != "normal" for r in rows])


def metrics(y, score, limit, rows):
    pred = score > limit
    normal, attack = ~y, y
    fp, tp = int((pred & normal).sum()), int((pred & attack).sum())
    return {"points": len(y), "attacked_points": int(attack.sum()), "normal_points": int(normal.sum()),
            "fp": fp, "tp": tp, "fpr": float(pred[normal].mean()),
            "recall": float(pred[attack].mean()) if attack.any() else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "pr_auc": float(average_precision_score(y, score)) if attack.any() and normal.any() else None,
            "roc_auc": float(roc_auc_score(y, score)) if attack.any() and normal.any() else None,
            "per_label": {label: {"count": sum(r["Label"] == label for r in rows),
                           "alarm_rate": float(pred[np.array([r["Label"] == label for r in rows])].mean())}
                          for label in sorted({r["Label"] for r in rows})}}


def experiment(data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    normal = read(Path(data) / "bridge_normal.csv")
    assert not labels(normal).any()
    train_end, cal_end = int(.6 * len(normal)), int(.8 * len(normal))
    reference = memory(normal[:train_end])
    normal_x = features(normal, reference)
    train_x = normal_x[:train_end]
    cal_x = normal_x[train_end + 32:cal_end]
    test_x = normal_x[cal_end + 32:]
    median = np.median(train_x, axis=0)
    mad = np.maximum(np.median(np.abs(train_x - median), axis=0) * 1.4826, .1)
    fit_time = time.perf_counter()
    # Raw packet model versus packet plus frozen graph-frequency context.
    models = {"if_packet": IsolationForest(n_estimators=100, max_samples=512, random_state=41, n_jobs=1).fit(train_x[:, :12]),
              "if_packet_graph": IsolationForest(n_estimators=100, max_samples=512, random_state=41, n_jobs=1).fit(train_x)}
    fit_time = time.perf_counter() - fit_time

    def scores(x):
        return {"edgebank": x[:, -1], "max_mad": np.max(np.abs((x - median) / mad), axis=1),
                "if_packet": -models["if_packet"].score_samples(x[:, :12]),
                "if_packet_graph": -models["if_packet_graph"].score_samples(x)}

    calibration = scores(cal_x)
    limits = {name: threshold(s, .01) for name, s in calibration.items()}
    evaluations = {"normal_holdout": (normal[cal_end + 32:], test_x)}
    for capture in ("bridge_attack_nrf.csv", "bridge_attack_smf.csv"):
        rows = read(Path(data) / capture)
        evaluations[capture] = (rows, features(rows, reference))
    result = {}
    for capture, (rows, x) in evaluations.items():
        t0 = time.perf_counter()
        ss = scores(x)
        inference = time.perf_counter() - t0
        y = labels(rows)
        result[capture] = {name: metrics(y, s, limits[name], rows) for name, s in ss.items()}
        result[capture]["inference_seconds_all_models"] = inference
        # Source CSVs are not redistributed; row scores and label audit are enough.
        dump_csv(output / (capture.replace(".csv", "") + "-predictions.csv"),
                 [{"row": i, "label": row["Label"], **{m: float(s[i]) for m, s in ss.items()}}
                  for i, row in enumerate(rows)])
    manifest = json.loads((Path(data) / "manifest.json").read_text())
    summary = {"scope": "External packet behavior only; no joint NFProfile/witness evidence is available.",
               "limitations": ["Only one normal capture and two attack captures; no independent-run confidence interval.",
                               "Captures have different calendar times; this is offline cross-capture evaluation, not a prospective deployment.",
                               "No reproduction of Granomaly TGN metrics is claimed.",
                               "The point-FPR calibration target may fail under capture or distribution shift."],
               "train_rows": train_end, "calibration_rows": len(cal_x), "normal_holdout_rows": len(test_x),
               "embargo_rows": 32, "features": FEATURES, "thresholds": limits, "results": result,
               "fit_seconds": fit_time, "source_manifest": manifest,
               "protocol_sha256": hashlib.sha256(Path(__file__).with_name("protocol.json").read_bytes()).hexdigest()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/granomaly")
    parser.add_argument("--output", default="results/external")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        experiment(args.data, args.output)
