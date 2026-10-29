"""Fit on live observations, calibrate before test, report availability only."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import warnings
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

FEATURES = ["direct_transport_missing", "log_direct_elapsed_ms", "discovery_missing",
            "log_discovery_elapsed_ms", "profile_count_deviation", "registered_fraction_deviation",
            "ausf_role", "nssf_role"]


def features(slot, obs):
    direct, nrf = obs["direct"], obs["nrf"]
    body = nrf["body"]
    rows = body.get("nfInstances", []) if isinstance(body, dict) else []
    registered = sum(r.get("nfStatus") == "REGISTERED" for r in rows) / max(1, len(rows))
    return [float(not direct["ok"]), np.log1p(direct["elapsed_ms"]),
            float(not nrf["ok"] or nrf["status"] != 200), np.log1p(nrf["elapsed_ms"]),
            abs(len(rows) - 1), 1 - registered, float(slot == "ausf"), float(slot == "nssf")]


def analyze(root):
    observations = [json.loads(s) for s in (root / "observations.jsonl").read_text().splitlines()]
    events = json.loads((root / "events.json").read_text())
    manifest = json.loads((root / "capture-manifest.json").read_text())
    rows, data = [], []
    for r in observations:
        for slot, obs in r["observations"].items():
            rows.append({"tick": r["tick"], "ns": r["monotonic_ns"], "stage": r["stage"],
                         "episode": r["episode"], "slot": slot, "policy": r["profile_policy"][slot]})
            data.append(features(slot, obs))
    x = np.asarray(data)
    fit = np.array([r["stage"] == "fit_normal" for r in rows])
    cal = np.array([r["stage"] == "calibration_normal" for r in rows])
    train = np.array([r["stage"].startswith("train_") for r in rows]) | fit
    normal = np.array([r["stage"] in ("test_normal", "test_baseline") for r in rows])
    recovery = np.array([r["stage"] == "test_recovery" for r in rows])
    truth = np.zeros(len(rows), dtype=int)
    for e in events:
        truth |= np.array([e["start_tick"] <= r["tick"] < e["end_tick"] and r["slot"] == e["slot"] for r in rows])
    scaler = StandardScaler().fit(x[fit])
    z = scaler.transform(x)
    med = np.median(x[fit, :6], axis=0)
    mad = np.maximum(np.median(abs(x[fit, :6] - med), axis=0) * 1.4826, 0.1)
    scores = {"transport_rule": x[:, 0], "robust_max": np.max(abs((x[:, :6] - med) / mad), axis=1)}
    training = {}
    models = {
        "isolation_forest": IsolationForest(n_estimators=150, max_samples=min(256, int(fit.sum())), random_state=manifest["replica"], n_jobs=1),
        "hgb": HistGradientBoostingClassifier(max_iter=150, max_depth=4, min_samples_leaf=5,
                    learning_rate=0.08, l2_regularization=1, early_stopping=False, random_state=manifest["replica"]),
        "mlp": MLPClassifier(hidden_layer_sizes=(48, 24), max_iter=300, batch_size=64,
                    alpha=0.001, early_stopping=False, random_state=manifest["replica"])}
    for name, model in models.items():
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter("always", ConvergenceWarning)
            if name == "isolation_forest":
                model.fit(z[fit])
                scores[name] = -model.score_samples(z)
            else:
                model.fit(z[train], truth[train])
                scores[name] = model.predict_proba(z)[:, 1]
        training[name] = {"seconds": time.perf_counter() - started,
                          "iteration_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught)}
    result = {"manifest": manifest, "features": FEATURES, "training": training,
              "fit_points": int(fit.sum()), "calibration_points": int(cal.sum()),
              "supervised_train_points": int(train.sum()), "supervised_positive_points": int(truth[train].sum()),
              "normal_test_points": int(normal.sum()), "models": {},
              "profile_policy_counts": {k: sum(r["policy"] == k for r in rows) for k in ("UNKNOWN", "UNAVAILABLE", "VIOLATION", "SCHEMA_VALID")},
              "full_original_CU_witness": "UNKNOWN", "python": platform.python_version(), "sklearn": sklearn.__version__,
              "observations_sha256": hashlib.sha256((root / "observations.jsonl").read_bytes()).hexdigest()}
    score_arrays = {}
    for name, score in scores.items():
        threshold = 0.5 if name == "transport_rule" else float(np.quantile(score[cal], 0.99, method="higher"))
        alarms = score > threshold
        episodes = []
        for e in events:
            if e["split"] != "test":
                continue
            ix = np.array([e["start_tick"] <= r["tick"] < e["end_tick"] and r["slot"] == e["slot"] for r in rows])
            hits = np.flatnonzero(ix & alarms)
            episodes.append({**e, "detected": bool(len(hits)),
                             "delay_seconds": (rows[hits[0]]["ns"] - e["start_ns"]) / 1e9 if len(hits) else None,
                             "points": int(ix.sum()), "detected_points": int(alarms[ix].sum())})
        result["models"][name] = {"threshold": threshold, "normal_points": int(normal.sum()),
                                  "false_points": int(alarms[normal].sum()), "normal_point_fpr": float(alarms[normal].mean()),
                                  "recovery_alarm_fraction": float(alarms[recovery].mean()),
                                  "episodes": episodes, "detected_episodes": sum(e["detected"] for e in episodes)}
        score_arrays[name] = score
    np.savez_compressed(root / "scores.npz", x=x, fit=fit, cal=cal, train=train, truth=truth, **score_arrays)
    (root / "evaluation.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    r = analyze(p.parse_args().input)
    print(json.dumps({k: {"detected": v["detected_episodes"], "fpr": v["normal_point_fpr"]} for k, v in r["models"].items()}, indent=2))
