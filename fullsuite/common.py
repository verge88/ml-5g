import csv
import hashlib
import json
import os
from pathlib import Path
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

P = json.loads(Path(__file__).with_name("protocol.json").read_text())
MASTER_SHA = hashlib.sha256(Path(__file__).with_name("protocol.json").read_bytes()).hexdigest()


def provenance(domain):
    return {"suite_domain": domain, "master_protocol_sha256": MASTER_SHA, "code_commit": os.getenv("GITHUB_SHA", os.getenv("SOURCE_COMMIT", "local"))}


def cutoff(x, alpha=None):
    return float(np.quantile(x, 1 - (P["point_alpha"] if alpha is None else alpha), method="higher"))


def robust(x, floor=.1):
    center = np.median(x, axis=0)
    scale = np.maximum(1.4826 * np.median(abs(x - center), axis=0), floor)
    return center, scale


def temporal(signal, mode):
    """Causal statistic with separate state for each NF; reset per whole run."""
    state = np.zeros_like(np.asarray(signal[0]), dtype=float)
    output = np.empty_like(signal, dtype=float)
    for i, value in enumerate(signal):
        state = P["ewma_alpha"] * value + (1-P["ewma_alpha"]) * state if mode == "ewma" else np.maximum(0, state + value - P["cusum_reference"])
        output[i] = state
    return output


def balanced_indices(y, seed):
    yes, no = np.flatnonzero(y), np.flatnonzero(~np.asarray(y, dtype=bool))
    rng = np.random.default_rng(seed)
    selected = np.r_[yes, rng.choice(no, min(len(no), P["supervised_normal_to_positive_ratio"]*len(yes)), replace=False)]
    rng.shuffle(selected)
    return selected


def fit_classifier(model, x, y):
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    return [str(w.message) for w in caught]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2))


def write_csv(path, rows):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
