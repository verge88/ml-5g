"""Frozen simulator benchmark. Negative findings never cause CI failure."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from .simulation import PROTOCOL, FEATURES, SLOTS, TYPES, fuse, split


def flatten(runs, field):
    arrays = [getattr(r, field) for r in runs]
    return np.concatenate([a.reshape((-1,) + a.shape[2:]) for a in arrays], axis=0)


def threshold(scores, alpha):
    """Fixed calibration-only empirical quantile with conservative tie handling."""
    return float(np.quantile(scores, 1 - alpha, method="higher"))


def sequential(z, mode):
    signal = np.max(z, axis=2)
    state = np.zeros(signal.shape[1])
    result = np.empty_like(signal)
    for t in range(len(signal)):
        if mode == "ewma":
            state = .25 * signal[t] + .75 * state
        else:
            state = np.maximum(0, state + signal[t] - 1.5)
        result[t] = state
    return result


class Models:
    def __init__(self, seed, normal, labeled):
        x, c = flatten(normal, "x"), flatten(normal, "context")
        self.expected = Ridge(alpha=1.).fit(c, x)
        residual = x - self.expected.predict(c)
        self.center = np.median(residual, axis=0)
        self.scale = np.maximum(np.median(np.abs(residual - self.center), axis=0) * 1.4826, 1e-4)
        z = (residual - self.center) / self.scale
        self.if_raw = IsolationForest(n_estimators=PROTOCOL["if_trees"], max_samples=512,
                                      random_state=seed, n_jobs=1).fit(x)
        self.if_context = IsolationForest(n_estimators=PROTOCOL["if_trees"], max_samples=512,
                                          random_state=seed, n_jobs=1).fit(z)
        lx, lc, ly = flatten(labeled, "x"), flatten(labeled, "context"), flatten(labeled, "y")
        lz = self.residual(lx, lc)
        params = dict(max_iter=PROTOCOL["hgb_iterations"], max_depth=3, learning_rate=.08,
                      l2_regularization=1., random_state=seed, early_stopping=False)
        self.hgb_raw = HistGradientBoostingClassifier(**params).fit(lx, ly)
        self.hgb_context = HistGradientBoostingClassifier(**params).fit(lz, ly)

    def residual(self, x, c):
        return (x - self.expected.predict(c) - self.center) / self.scale

    def scores(self, run):
        shape = run.y.shape
        x, c = run.x.reshape(-1, len(FEATURES)), run.context.reshape(-1, run.context.shape[-1])
        z = self.residual(x, c)
        zz = z.reshape(shape + (len(FEATURES),))
        return {"max_residual": z.max(axis=1).reshape(shape),
                "ewma": sequential(zz, "ewma"), "cusum": sequential(zz, "cusum"),
                "if_raw": -self.if_raw.score_samples(x).reshape(shape),
                "if_context": -self.if_context.score_samples(z).reshape(shape),
                "hgb_raw": self.hgb_raw.predict_proba(x)[:, 1].reshape(shape),
                "hgb_context": self.hgb_context.predict_proba(z)[:, 1].reshape(shape)}


def event_metrics(runs, alarms):
    rows = []
    for r in runs:
        for event in r.events:
            start, end, nf = event["start"], event["end"], event["nf"]
            hits = np.flatnonzero(alarms[r.run_id][start:end, nf])
            delay = int(hits[0]) if len(hits) else None
            rows.append({**event, "detected": bool(len(hits)), "delay_ticks": delay,
                         "restricted_delay_ticks": delay if delay is not None else end - start})
    return rows


def summarize(runs, alarms):
    truth = np.concatenate([r.y.ravel() for r in runs])
    pred = np.concatenate([alarms[r.run_id].ravel() for r in runs])
    event_rows = event_metrics(runs, alarms)
    normal = ~truth
    fp = int(np.sum(pred & normal))
    behavior = [e for e in event_rows if e["kind"] in {"fanout", "burst", "sequence"}]
    observable = [e for e in event_rows if e["kind"] != "stealth"]
    per_kind = {}
    for kind in sorted({e["kind"] for e in event_rows}):
        events = [e for e in event_rows if e["kind"] == kind]
        per_kind[kind] = {"events": len(events), "detected": sum(e["detected"] for e in events),
                          "recall": float(np.mean([e["detected"] for e in events])),
                          "restricted_delay_ticks": float(np.mean([e["restricted_delay_ticks"] for e in events]))}
    phase_mask = np.concatenate([r.phase.ravel() > 0 for r in runs]) & normal
    drift_mask = np.concatenate([np.full(r.y.size, r.regime == "drift") for r in runs]) & normal
    return {"normal_points": int(normal.sum()), "fp": fp, "fpr": fp / int(normal.sum()),
            "point_recall": float(pred[truth].mean()), "events": len(event_rows),
            "event_recall": float(np.mean([e["detected"] for e in event_rows])),
            "observable_event_recall": float(np.mean([e["detected"] for e in observable])),
            "profile_valid_observable_recall": float(np.mean([e["detected"] for e in behavior])),
            "restricted_delay_ticks": float(np.mean([e["restricted_delay_ticks"] for e in event_rows])),
            "restart_fpr": float(pred[phase_mask].mean()), "drift_fpr": float(pred[drift_mask].mean()),
            "per_kind": per_kind}


def schedule(risk, mode):
    """One mandatory round-robin observation plus one optional observation.

    Priority sees t-1 telemetry only; all modes consume exactly 2 credits/tick.
    A credit represents acquisition, not total end-to-end CPU or network cost.
    """
    ticks, n = risk.shape
    selected = np.zeros_like(risk, dtype=bool)
    last = np.full(n, -1)
    for t in range(ticks):
        mandatory = t % n
        candidates = [i for i in range(n) if i != mandatory]
        if mode == "uniform":
            extra = (mandatory + n // 2) % n
        elif mode == "oldest":
            extra = min(candidates, key=lambda i: (last[i], i))
        elif mode == "ml":
            prior = risk[t - 1] if t else np.zeros(n)
            extra = max(candidates, key=lambda i: (prior[i], t - last[i], -i))
        else:
            raise ValueError(mode)
        selected[t, [mandatory, extra]] = True
        last[[mandatory, extra]] = t
    assert np.all(selected.sum(axis=1) == 2)
    for nf in range(n):
        gaps = np.diff(np.r_[-1, np.flatnonzero(selected[:, nf]), ticks - 1])
        assert gaps.max() <= PROTOCOL["scheduler_mandatory_max_gap"]
    return selected


def dump_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def experiment(seed, output):
    output = Path(output) / f"seed-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    normal, attack, calibration, test = [split(seed, p) for p in ("normal", "attack", "calibration", "test")]
    t0 = time.perf_counter()
    model = Models(seed, normal, normal + attack)
    train_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    cal_scores = [model.scores(r) for r in calibration]
    thresholds = {m: threshold(np.concatenate([s[m].ravel() for s in cal_scores]),
                               PROTOCOL["target_calibration_point_fpr"]) for m in PROTOCOL["models"]}
    scores = {r.run_id: model.scores(r) for r in test}
    infer_seconds = time.perf_counter() - t0
    alarms = {"integrity": {r.run_id: r.hard for r in test}}
    for name in PROTOCOL["models"]:
        alarms[name] = {r.run_id: scores[r.run_id][name] > thresholds[name] for r in test}
        alarms["hybrid_" + name] = {
            r.run_id: fuse(r.hard, r.unknown, alarms[name][r.run_id]) for r in test}
        for r in test:
            assert np.all(alarms["hybrid_" + name][r.run_id][r.hard])
    summaries = {name: summarize(test, p) for name, p in alarms.items()}
    events = []
    for name, predictions in alarms.items():
        events.extend({"model": name, **e} for e in event_metrics(test, predictions))
    scheduler = {}
    scheduler_events = []
    for mode in ("uniform", "oldest", "ml"):
        selected = {r.run_id: schedule(scores[r.run_id]["hgb_context"], mode) for r in test}
        confirmed = {r.run_id: selected[r.run_id] & r.hard for r in test}
        rows = [e for e in event_metrics(test, confirmed) if e["kind"] in {"integrity", "mixed"}]
        scheduler[mode] = {"checks": sum(int(a.sum()) for a in selected.values()),
                           "max_gap_bound_ticks": 13,
                           "per_kind": {k: {"events": sum(e["kind"] == k for e in rows),
                             "recall": float(np.mean([e["detected"] for e in rows if e["kind"] == k])),
                             "restricted_delay_ticks": float(np.mean([e["restricted_delay_ticks"] for e in rows if e["kind"] == k]))}
                             for k in ("integrity", "mixed")}}
        scheduler_events.extend({"scheduler": mode, **e} for e in rows)
    assert len({s["checks"] for s in scheduler.values()}) == 1
    # Full test observables and predictions support independent metric auditing.
    arrays = {"x": np.stack([r.x for r in test]), "context": np.stack([r.context for r in test]),
              "truth": np.stack([r.y for r in test]), "hard": np.stack([r.hard for r in test]),
              "unknown": np.stack([r.unknown for r in test]), "phase": np.stack([r.phase for r in test])}
    for name in PROTOCOL["models"]:
        arrays["score_" + name] = np.stack([scores[r.run_id][name] for r in test])
    np.savez_compressed(output / "test-observations.npz", **arrays)
    dump_csv(output / "events.csv", events)
    dump_csv(output / "scheduler-events.csv", scheduler_events)
    result = {"seed": seed, "scope": PROTOCOL["scope"], "models": summaries, "scheduler": scheduler,
              "thresholds": thresholds, "train_seconds": train_seconds, "inference_seconds": infer_seconds,
              "elapsed_seconds": time.perf_counter() - started,
              "unknown_points": int(sum(r.unknown.sum() for r in test)),
              "point_fpr_note": "Empirical calibration target is not a test guarantee or an anytime bound.",
              "run_ids": {"normal_train": [r.run_id for r in normal], "labeled_train": [r.run_id for r in attack],
                          "calibration": [r.run_id for r in calibration], "test": [r.run_id for r in test]},
              "features": FEATURES, "context_columns": TYPES + ["load", "authorized_restart_phase"],
              "protocol_sha256": hashlib.sha256(Path(__file__).with_name("protocol.json").read_bytes()).hexdigest(),
              "commit": os.getenv("GITHUB_SHA", "local-uncommitted"),
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "scipy": scipy.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()}}
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"seed": seed, "elapsed_seconds": result["elapsed_seconds"],
                      "hybrid": summaries["hybrid_hgb_context"], "scheduler": scheduler}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", default="results/simulation")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        for seed in args.seeds:
            experiment(seed, args.output)
