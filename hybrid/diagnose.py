"""Controlled diagnosis of the earlier model comparison.

The original generator, protocol and benchmark are imported without changes.
All new arms are explanatory interventions on five additional random seeds.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from threadpoolctl import threadpool_limits
from . import benchmark, external, simulation
from .benchmark import Models, flatten, summarize, threshold
from .external import features, labels, memory, read
from .simulation import FEATURES, PROTOCOL, run, split

ROOT = Path(__file__).parent
DIAG = json.loads((ROOT / "diagnostic_protocol.json").read_text())


def altered_training(seed, attacked):
    """One-factor data and label changes; neither touches calibration or test."""
    exposed = copy.deepcopy(attacked)
    clean_labels = [r.y.copy() for r in attacked]
    changed = []
    for i, (original, modified) in enumerate(zip(attacked, exposed)):
        for event in original.events:
            if event["kind"] == "integrity":
                clean_labels[i][event["start"]:event["end"], event["nf"]] = False
        event = [e for e in modified.events if e["kind"] == "fanout"][1]
        child = int(np.random.SeedSequence([seed, 20, i]).generate_state(1)[0])
        clean_run = run(child, original.run_id, attacks=False, lifecycle=i % 4 != 0)
        start, end, nf = event["start"], event["end"], event["nf"]
        modified.x[start:end, nf] = clean_run.x[start:end, nf]
        modified.x[start:end, nf, 5] += np.linspace(.02, .18, end - start)
        event["kind"] = "sequence"
        mask = np.zeros_like(original.y)
        mask[start:end, nf] = True
        assert np.array_equal(original.x[~mask], modified.x[~mask])
        assert np.array_equal(original.y, modified.y)
        assert np.array_equal(original.context, modified.context)
        changed.append({"run": original.run_id, "nf": nf, "start": start, "end": end})
    return exposed, clean_labels, changed


def fit_hgb(z, y, seed, deeper=False):
    return HistGradientBoostingClassifier(
        max_iter=300 if deeper else 100, max_depth=6 if deeper else 3,
        learning_rate=.08, l2_regularization=1., random_state=seed,
        early_stopping=False).fit(z, y)


def simulation_diagnosis(seed):
    normal, attacked, calibration, test = [split(seed, part) for part in ("normal", "attack", "calibration", "test")]
    model = Models(seed, normal, normal + attacked)
    x, c, y = [flatten(normal + attacked, field) for field in ("x", "context", "y")]
    z = model.residual(x, c)
    exposed, cleaned, changes = altered_training(seed, attacked)
    ey = np.concatenate([flatten(normal, "y"), np.concatenate([a.ravel() for a in cleaned])])
    ex, ec = flatten(normal + exposed, "x"), flatten(normal + exposed, "context")
    ez = model.residual(ex, ec)
    estimators = {"hgb_original": model.hgb_context,
                  "hgb_behavior_labels": fit_hgb(z, ey, seed),
                  "hgb_sequence_exposure": fit_hgb(ez, y, seed),
                  "hgb_behavior_labels_and_exposure": fit_hgb(ez, ey, seed),
                  "hgb_deeper": fit_hgb(z, y, seed, deeper=True)}

    def score(r):
        rz = model.residual(r.x.reshape(-1, len(FEATURES)), r.context.reshape(-1, r.context.shape[-1]))
        result = {name: estimator.predict_proba(rz)[:, 1].reshape(r.y.shape)
                  for name, estimator in estimators.items()}
        result["max_residual"] = rz.max(axis=1).reshape(r.y.shape)
        result["if_context"] = -model.if_context.score_samples(rz).reshape(r.y.shape)
        return result

    cal = [score(r) for r in calibration]
    cutoffs = {name: threshold(np.concatenate([s[name].ravel() for s in cal]), .01) for name in DIAG["arms"]}
    ss = {r.run_id: score(r) for r in test}
    metrics = {name: summarize(test, {r.run_id: ss[r.run_id][name] > cutoffs[name] for r in test})
               for name in DIAG["arms"]}
    training_counts = {}
    for r in attacked:
        for event in r.events:
            training_counts[event["kind"]] = training_counts.get(event["kind"], 0) + event["end"] - event["start"]
    # A clean single-coordinate intervention shows feature sensitivity independently of other anomalies.
    cz = model.residual(flatten(calibration, "x"), flatten(calibration, "context"))
    probes = []
    for value in (0., 2., 4., 6., 8.):
        probe = cz[:2048].copy()
        probe[:, 5] = value
        scores = {name: estimator.predict_proba(probe)[:, 1] for name, estimator in estimators.items()}
        scores["max_residual"] = probe.max(axis=1)
        scores["if_context"] = -model.if_context.score_samples(probe)
        probes.append({"sequence_residual": value,
                       "alarm_rates": {name: float((s > cutoffs[name]).mean()) for name, s in scores.items()}})
    return {"seed": seed, "models": metrics, "thresholds": cutoffs,
            "training_points": len(y), "positive_training_points": int(y.sum()),
            "training_positive_by_kind": training_counts,
            "unobservable_integrity_positive_fraction": training_counts["integrity"] / sum(training_counts.values()),
            "training_changes": changes, "single_coordinate_probes": probes}


def packet_signature(row):
    # Descriptive overlap only: omit temporal quantities and label-bearing free text.
    return tuple(row[k] for k in ("ip.src", "ip.dst", "frame.len", "frame.protocols", "http2.header.length", "http2.header.count"))


def external_diagnosis(data):
    data = Path(data)
    normal = read(data / "bridge_normal.csv")
    fit_end, cal_end = int(.6 * len(normal)), int(.8 * len(normal))
    reference = memory(normal[:fit_end])
    nx = features(normal, reference)
    fit_x, cal_x = nx[:fit_end], nx[fit_end + 32:cal_end]
    forest = IsolationForest(n_estimators=100, max_samples=512, random_state=41, n_jobs=1).fit(fit_x)
    cutoff = threshold(-forest.score_samples(cal_x), .01)
    split_counts = np.zeros(fit_x.shape[1], dtype=int)
    for estimator, feature_map in zip(forest.estimators_, forest.estimators_features_):
        local = estimator.tree_.feature
        for index in local[local >= 0]:
            split_counts[feature_map[index]] += 1
    original_scores = forest.score_samples(cal_x)
    changed = cal_x.copy()
    changed[:, -1] = 1 - changed[:, -1]
    constant_change = float(np.max(np.abs(original_scores - forest.score_samples(changed))))
    original_signature_set = {packet_signature(r) for r in normal}
    captures = {}
    for filename in ("bridge_attack_nrf.csv", "bridge_attack_smf.csv"):
        rows = read(data / filename)
        x, y = features(rows, reference), labels(rows)
        score = -forest.score_samples(x)
        novel = x[:, -1] > 0
        alarm = score > cutoff
        overlap = np.array([packet_signature(r) in original_signature_set for r in rows])
        within_capture_normal = {packet_signature(r) for r in rows if r["Label"].strip().lower() == "normal"}
        same_capture_overlap = np.array([packet_signature(r) in within_capture_normal for r in rows])
        captures[filename] = {
            "attack_packets": int(y.sum()), "new_pair_attack_packets": int((y & novel).sum()),
            "known_pair_attack_packets": int((y & ~novel).sum()),
            "new_pair_fraction_of_attacks": float(novel[y].mean()),
            "if_recall_new_pairs": float(alarm[y & novel].mean()) if (y & novel).any() else None,
            "if_recall_known_pairs": float(alarm[y & ~novel].mean()) if (y & ~novel).any() else None,
            "attacks_sharing_static_signature_with_normal_capture": float(overlap[y].mean()),
            "attacks_sharing_static_signature_with_normal_in_same_capture": float(same_capture_overlap[y].mean()),
            "note": "Static signature overlap does not establish equality of all temporal features or an information-theoretic impossibility."}
    return {"features": external.FEATURES,
            "constant_fit_features": [external.FEATURES[i] for i in range(fit_x.shape[1]) if np.ptp(fit_x[:, i]) == 0],
            "split_counts": dict(zip(external.FEATURES, map(int, split_counts))),
            "max_score_change_when_only_new_pair_changes": constant_change,
            "new_pair_fit_values": sorted(map(float, np.unique(fit_x[:, -1]))),
            "captures": captures,
            "cutoff": cutoff,
            "note": "The causal tree-usage check is exact for this fitted forest. Static overlap is descriptive and uses labels only for evaluation."}


def main(data, output, seeds):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = []
    for seed in seeds:
        record = simulation_diagnosis(seed)
        records.append(record)
        print(json.dumps({"seed": seed, "sequence_recall": {m: s["per_kind"]["sequence"]["recall"] for m, s in record["models"].items()}}), flush=True)
    ext = external_diagnosis(data)
    result = {"scope": "Post-hoc mechanism diagnosis on fresh simulator seeds and pinned external captures",
              "records": records, "external": ext, "elapsed_seconds": time.perf_counter() - started,
              "commit": os.getenv("GITHUB_SHA", "local-uncommitted"),
              "diagnostic_protocol_sha256": hashlib.sha256((ROOT / "diagnostic_protocol.json").read_bytes()).hexdigest(),
              "original_protocol_sha256": hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest(),
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT / "benchmark.py", ROOT / "simulation.py", ROOT / "external.py"]}}
    (output / "diagnosis.json").write_text(json.dumps(result, indent=2))
    lines = ["# Model gap mechanism diagnosis", "", "Fresh random seeds; original source and protocol unchanged. This is explanatory follow-up, not a new original benchmark result.", "",
             "| Arm | Sequence episode recall | Overall normal point FPR |", "|---|---:|---:|"]
    for name in DIAG["arms"]:
        recall = np.mean([r["models"][name]["per_kind"]["sequence"]["recall"] for r in records])
        fpr = np.mean([r["models"][name]["fpr"] for r in records])
        lines.append(f"| {name} | {recall:.3%} | {fpr:.3%} |")
    lines += ["", "Exposing a model to sequence examples diagnoses the missing-class mechanism; it must not be advertised as detection of an unseen attack.", "",
              f"Positive training labels with no behavioral change: {records[0]['unobservable_integrity_positive_fraction']:.1%}.", "",
              f"Forest splits using new_pair: {ext['split_counts']['new_pair']}. Maximum forest score change when only new_pair is flipped: {ext['max_score_change_when_only_new_pair_changes']}.", "",
              "## External captures", "", "| Capture | Attacks on new pairs | Shared static signature with a normal packet in same capture |", "|---|---:|---:|"]
    for name, r in ext["captures"].items():
        lines.append(f"| {name} | {r['new_pair_fraction_of_attacks']:.3%} | {r['attacks_sharing_static_signature_with_normal_in_same_capture']:.3%} |")
    report = "\n".join(lines) + "\n"
    (output / "REPORT.md").write_text(report)
    print(report)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/granomaly")
    parser.add_argument("--output", default="results/diagnosis")
    parser.add_argument("--seeds", type=int, nargs="+", default=DIAG["new_simulator_seeds"])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args.data, args.output, args.seeds)
