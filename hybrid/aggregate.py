"""Audit artifacts and summarize all frozen test seeds, including failures."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from .benchmark import dump_csv
from .simulation import PROTOCOL


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261028)
    means = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "lo": float(np.quantile(means, .025)), "hi": float(np.quantile(means, .975))}


def aggregate(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    external = None
    for path in source.rglob("summary.json"):
        record = json.loads(path.read_text())
        if "seed" in record:
            records.append(record)
        elif "source_manifest" in record:
            if external is not None:
                raise ValueError("Duplicate external artifact")
            external = record
    assert sorted(r["seed"] for r in records) == PROTOCOL["test_seeds"], "Missing, extra or duplicated frozen seeds"
    assert external is not None, "External experiment artifact missing"
    hashes = {r["protocol_sha256"] for r in records} | {external["protocol_sha256"]}
    assert len(hashes) == 1, "Protocol changed during the experiment"
    assert len({r["commit"] for r in records}) == 1, "Mixed code revisions"
    rows, aggregate_models = [], {}
    metrics = ["event_recall", "observable_event_recall", "profile_valid_observable_recall", "fpr", "restart_fpr", "drift_fpr", "restricted_delay_ticks"]
    for model in records[0]["models"]:
        estimates = {m: interval([r["models"][model][m] for r in records]) for m in metrics}
        aggregate_models[model] = estimates
        rows.append({"model": model, **{m: estimates[m]["mean"] for m in metrics}})
    paired = {}
    for name in ("if_context", "hgb_context", "max_residual", "ewma", "cusum"):
        paired[name + "_hybrid_minus_integrity"] = interval([
            r["models"]["hybrid_" + name]["profile_valid_observable_recall"] - r["models"]["integrity"]["profile_valid_observable_recall"] for r in records])
    scheduler = {}
    for kind in ("integrity", "mixed"):
        scheduler[kind] = {mode: interval([r["scheduler"][mode]["per_kind"][kind]["restricted_delay_ticks"] for r in records])
                           for mode in ("uniform", "oldest", "ml")}
        paired["scheduler_ml_minus_uniform_" + kind] = interval([
            r["scheduler"]["ml"]["per_kind"][kind]["restricted_delay_ticks"] -
            r["scheduler"]["uniform"]["per_kind"][kind]["restricted_delay_ticks"] for r in records])
    combined = {"seeds": sorted(r["seed"] for r in records), "models": aggregate_models,
                "paired": paired, "scheduler": scheduler, "external": external,
                "protocol_sha256": next(iter(hashes)), "commits": list({r["commit"] for r in records}),
                "bootstrap_note": "Intervals resample simulator seeds; they do not establish external validity or population guarantees."}
    (output / "summary.json").write_text(json.dumps(combined, indent=2))
    dump_csv(output / "model-comparison.csv", rows)
    text = ["# Integrity and ML experimental results", "",
            "This report contains simulator results and a separate external packet-trace evaluation. It is not a live end-to-end 5G trial.", "",
            f"Code: `{combined['commits'][0]}`. Protocol SHA256: `{combined['protocol_sha256']}`.", "",
            "The 10 frozen seeds contain 200 complete test runs and 800 attack episodes; 100 episodes are an unobservable negative control.", "",
            "## Simulator results", "",
            "| Model | Profile-valid observable event recall | Test point FPR | Restart FPR | Drift FPR |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        text.append(f"| {row['model']} | {row['profile_valid_observable_recall']:.3%} | {row['fpr']:.3%} | {row['restart_fpr']:.3%} | {row['drift_fpr']:.3%} |")
    text += ["", "Event recall is not point-adjusted. A single late hit does not relabel all attack points. Undetected events contribute their full duration to restricted delay.", "",
             "The nominal 1% threshold is calibrated on separate normal runs. Reported test FPR is empirical and may exceed it.", "",
             "## Acquisition scheduler", "", "Each mode uses exactly two acquisition credits per tick and keeps a mandatory gap of at most 13 ticks. ML priority sees previous-tick telemetry.", "",
             "| Episode | Uniform delay | Oldest-first delay | ML priority delay |", "|---|---:|---:|---:|"]
    for kind, data in scheduler.items():
        text.append(f"| {kind} | {data['uniform']['mean']:.3f} | {data['oldest']['mean']:.3f} | {data['ml']['mean']:.3f} |")
    text += ["", "These are simulated acquisition costs and tick delays. Passive telemetry, model inference and physical network costs are not included in a credit.", "",
             "## External Open5GS traces", "", "No NFProfile, lifecycle authorization or independent runtime witness is fabricated for these captures. Only behavioral models are evaluated.", "",
             "| Capture | Model | Recall | FPR | PR AUC |", "|---|---|---:|---:|---:|"]
    for capture, results in external["results"].items():
        for model, result in results.items():
            if not isinstance(result, dict):
                continue
            recall = f"{result['recall']:.3%}" if result["recall"] is not None else "n/a"
            pr = f"{result['pr_auc']:.4f}" if result["pr_auc"] is not None else "n/a"
            text.append(f"| {capture} | {model} | {recall} | {result['fpr']:.3%} | {pr} |")
    text += ["", "## Interpretation limits", "",
             "A successful CI run means that the experiment executed and invariants passed. It does not mean that ML beat all statistical baselines or that novelty was established.", "",
             "The synthetic generator defines observable attack differences; external captures contain only one normal session and two attack sessions. Source timestamps preclude interpreting this as a prospective chronological deployment.", "",
             "The original Stage 39 verifier is byte-checked. Witness independence and authorized lifecycle delivery are simulator assumptions; no trusted hardware attestation is claimed.", "",
             "All models, seed selection and thresholds are frozen in protocol.json. Training, calibration and test use disjoint whole simulator runs. The negative control must not be interpreted as a class of attacks reliably observable by ML."]
    report = "\n".join(text) + "\n"
    (output / "REPORT.md").write_text(report)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write(report)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/final")
    args = parser.parse_args()
    aggregate(args.input, args.output)
