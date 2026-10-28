"""Frozen benchmark with corrected behavior labels and shared causal features."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import warnings
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from .features import LearnedFeatures, flatten, robust_fit
from .generator import PROTOCOL, datasets


def cutoff(values):
    return float(np.quantile(values, 1 - PROTOCOL["nominal_point_fpr"], method="higher"))


def scaled(values, center, scale):
    return (values - center) / scale


def event_rows(traces, alarms, model):
    result = []
    for r in traces:
        if r.event is None:
            continue
        e = r.event
        hits = np.flatnonzero(alarms[r.run_id][e["start"]:e["end"], e["nf"]])
        normal_id = r.run_id.rsplit("-", 1)[0] + "-False"
        benign_alarm = bool(alarms[normal_id][e["start"]:e["end"], e["nf"]].any())
        result.append({"model": model, **e, "detected": bool(len(hits)),
                       "benign_twin_alarm": benign_alarm,
                       "delay": int(hits[0]) if len(hits) else None,
                       "restricted_delay": int(hits[0]) if len(hits) else e["end"] - e["start"]})
    return result


def metrics(traces, alarms, model):
    events = event_rows(traces, alarms, model)
    normal = np.concatenate([(~r.attack).ravel() for r in traces])
    pred = np.concatenate([alarms[r.run_id].ravel() for r in traces])
    phase = np.concatenate([r.phase.ravel() for r in traces]) & normal
    by_kind = {}
    for kind in PROTOCOL["kinds"]:
        rows = [e for e in events if e["kind"] == kind]
        by_kind[kind] = {"events": len(rows), "detected": sum(e["detected"] for e in rows),
                         "recall": float(np.mean([e["detected"] for e in rows])),
                         "benign_twin_alarm_rate": float(np.mean([e["benign_twin_alarm"] for e in rows])),
                         "paired_detection_gain": float(np.mean([int(e["detected"]) - int(e["benign_twin_alarm"]) for e in rows])),
                         "restricted_delay": float(np.mean([e["restricted_delay"] for e in rows]))}
    observable = [e for e in events if e["observable"]]
    behavior = [e for e in events if e["has_behavior"]]
    heldout = [e for e in events if e["kind"] in PROTOCOL["heldout_kinds"]]
    # Evaluate exactly the same target NF and time interval in its normal twin.
    benign_hits = []
    for r in traces:
        if r.event is not None:
            e = r.event
            normal_id = r.run_id.rsplit("-", 1)[0] + "-False"
            benign_hits.append(bool(alarms[normal_id][e["start"]:e["end"], e["nf"]].any()))
    return {"point_fpr": float(pred[normal].mean()), "normal_points": int(normal.sum()),
            "false_points": int(pred[normal].sum()),
            "restart_fpr": float(pred[phase].mean()),
            "observable_event_recall": float(np.mean([e["detected"] for e in observable])),
            "behavior_event_recall": float(np.mean([e["detected"] for e in behavior])),
            "behavior_paired_gain": float(np.mean([int(e["detected"]) - int(e["benign_twin_alarm"]) for e in behavior])),
            "heldout_recall": float(np.mean([e["detected"] for e in heldout])),
            "benign_episode_alarm_rate": float(np.mean(benign_hits)),
            "restricted_delay": float(np.mean([e["restricted_delay"] for e in observable])),
            "per_kind": by_kind}


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def experiment(seed, output):
    out = Path(output) / f"seed-{seed}"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    normal, train, calibration, test = datasets(seed)
    representation = LearnedFeatures(normal)
    cache = {}
    for r in {r.run_id: r for r in normal + train + calibration + test}.values():
        cache[r.run_id] = representation.transform(r)
    nx = np.concatenate([cache[r.run_id][0].reshape(-1, cache[r.run_id][0].shape[-1]) for r in normal])
    nc = np.concatenate([cache[r.run_id][1].reshape(-1, 7) for r in normal])
    rich_center, rich_scale = robust_fit(nx)
    component_center, component_scale = robust_fit(nc)
    forest = IsolationForest(n_estimators=PROTOCOL["if_trees"], max_samples=PROTOCOL["if_max_samples"],
                              random_state=seed, n_jobs=1).fit(nx)
    forest_fit = -forest.score_samples(nx)
    if_center, if_scale = robust_fit(forest_fit[:, None], .001)
    sx = np.concatenate([cache[r.run_id][0].reshape(-1, nx.shape[1]) for r in train])
    sy = flatten(train, "behavior")
    # Fix label imbalance using training data only; retain every positive window.
    rng = np.random.default_rng(seed + 90000)
    positive, negative = np.flatnonzero(sy), np.flatnonzero(~sy)
    take_normal = min(len(negative), PROTOCOL["supervised_normal_to_positive_ratio"] * len(positive))
    indices = np.r_[positive, rng.choice(negative, take_normal, replace=False)]
    rng.shuffle(indices)
    learn_x = np.clip(scaled(sx[indices], rich_center, rich_scale), -20, 20)
    learn_y = sy[indices]
    hgb = HistGradientBoostingClassifier(max_iter=PROTOCOL["hgb_iterations"], max_depth=PROTOCOL["hgb_depth"],
                                          learning_rate=.08, l2_regularization=1., random_state=seed,
                                          early_stopping=False).fit(learn_x, learn_y)
    mlp = MLPClassifier(hidden_layer_sizes=tuple(PROTOCOL["mlp_hidden"]), max_iter=PROTOCOL["mlp_iterations"],
                        batch_size=256, alpha=.001, random_state=seed, early_stopping=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        mlp.fit(learn_x, learn_y)

    def score(r):
        features, components = cache[r.run_id]
        shape = features.shape[:2]
        x = features.reshape(-1, nx.shape[1])
        comp = components.reshape(-1, 7)
        ifscore = -forest.score_samples(x)
        structure = np.max(scaled(comp, component_center, component_scale), axis=1)
        z = np.clip(scaled(x, rich_center, rich_scale), -20, 20)
        # Explicit use of structural novelty avoids a silent constant-feature failure.
        ifguard = np.maximum(scaled(ifscore[:, None], if_center, if_scale).ravel(), structure)
        return {"raw_max": comp[:, 0].reshape(shape), "mahalanobis": comp[:, 1].reshape(shape),
                "causal_structure": structure.reshape(shape), "if_rich": ifscore.reshape(shape),
                "if_structure_guard": ifguard.reshape(shape),
                "hgb_rich": hgb.predict_proba(z)[:, 1].reshape(shape),
                "mlp_rich": mlp.predict_proba(z)[:, 1].reshape(shape)}

    cal = [score(r) for r in calibration]
    limits = {m: cutoff(np.concatenate([s[m].ravel() for s in cal])) for m in PROTOCOL["models"]}
    block = PROTOCOL["event_length"]
    episode_limits = {m: float(np.quantile(np.concatenate([
        s[m].reshape(-1, block, s[m].shape[1]).max(axis=1).ravel() for s in cal]),
        1 - PROTOCOL["nominal_episode_false_alarm_rate"], method="higher")) for m in PROTOCOL["models"]}
    scores = {r.run_id: score(r) for r in test}
    predictions = {"integrity": {r.run_id: r.hard for r in test}}
    for model in PROTOCOL["models"]:
        predictions[model] = {r.run_id: scores[r.run_id][model] > limits[model] for r in test}
        predictions["hybrid_" + model] = {r.run_id: r.hard | predictions[model][r.run_id] for r in test}
        for r in test:
            assert np.all(predictions["hybrid_" + model][r.run_id][r.hard])
        predictions[model + "@episode"] = {r.run_id: scores[r.run_id][model] > episode_limits[model] for r in test}
        predictions["hybrid_" + model + "@episode"] = {
            r.run_id: r.hard | predictions[model + "@episode"][r.run_id] for r in test}
        for r in test:
            assert np.all(predictions["hybrid_" + model + "@episode"][r.run_id][r.hard])
    result = {name: metrics(test, alarms, name) for name, alarms in predictions.items()}
    rows = []
    for name, alarms in predictions.items():
        rows.extend(event_rows(test, alarms, name))
    write_csv(out / "events.csv", rows)
    # Keep all test observations and model scores for independent re-evaluation.
    arrays = {field: np.stack([getattr(r, field) for r in test])
              for field in ("x", "context", "tokens", "destinations", "phase", "generation", "hard", "unknown", "attack", "behavior", "integrity")}
    for model in PROTOCOL["models"]:
        arrays["score_" + model] = np.stack([scores[r.run_id][model] for r in test])
    np.savez_compressed(out / "test-observations.npz", **arrays)
    summary = {"seed": seed, "scope": PROTOCOL["scope"], "models": result, "thresholds": limits,
               "episode_thresholds": episode_limits,
               "test_runs": len(test), "test_episodes": sum(r.event is not None for r in test),
               "behavior_positive_training_points": int(sy.sum()), "supervised_fit_points": len(indices),
               "unknown_points": int(sum(r.unknown.sum() for r in test)),
               "learned_peer_by_slot": representation.peer.tolist(), "rich_feature_count": nx.shape[1],
               "mlp_iterations": mlp.n_iter_, "mlp_final_training_loss": float(mlp.loss_),
               "mlp_warnings": [str(w.message) for w in caught],
               "elapsed_seconds": time.perf_counter() - started,
               "protocol_sha256": hashlib.sha256(Path(__file__).with_name("protocol.json").read_bytes()).hexdigest(),
               "commit": os.getenv("GITHUB_SHA", "local-uncommitted"),
               "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "sklearn": sklearn.__version__},
               "train_run_ids": [r.run_id for r in train], "calibration_run_ids": [r.run_id for r in calibration],
               "test_run_ids": [r.run_id for r in test]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"seed": seed, "seconds": summary["elapsed_seconds"],
                      "models": {m: {k: result[m][k] for k in ("behavior_event_recall", "heldout_recall", "point_fpr", "benign_episode_alarm_rate")}
                                 for m in ("raw_max", "causal_structure", "if_rich", "hgb_rich", "mlp_rich")}}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", default="results/complex")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        for seed in args.seeds:
            experiment(seed, args.output)
