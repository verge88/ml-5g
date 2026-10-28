"""Audit all frozen seeds and report both point and episode calibrations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .generator import PROTOCOL
from .benchmark import write_csv


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261029)
    means = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "lo": float(np.quantile(means, .025)), "hi": float(np.quantile(means, .975))}


def aggregate(source, output):
    records = [json.loads(p.read_text()) for p in Path(source).rglob("summary.json")]
    assert sorted(r["seed"] for r in records) == PROTOCOL["test_seeds"]
    hashes = {r["protocol_sha256"] for r in records}
    assert hashes == {hashlib.sha256(Path(__file__).with_name("protocol.json").read_bytes()).hexdigest()}
    commits = {r["commit"] for r in records}
    assert len(commits) == 1
    if os.getenv("GITHUB_SHA"):
        assert commits == {os.environ["GITHUB_SHA"]}
    keys = ["observable_event_recall", "behavior_event_recall", "behavior_paired_gain", "heldout_recall",
            "point_fpr", "benign_episode_alarm_rate", "restart_fpr", "restricted_delay"]
    summaries, rows = {}, []
    for name in records[0]["models"]:
        summaries[name] = {k: interval([r["models"][name][k] for r in records]) for k in keys}
        summaries[name]["per_kind"] = {
            kind: {k: interval([r["models"][name]["per_kind"][kind][k] for r in records])
                   for k in ["recall", "benign_twin_alarm_rate", "paired_detection_gain", "restricted_delay"]}
            for kind in PROTOCOL["kinds"]}
        # Identical observables must yield identical negative-control decisions.
        assert summaries[name]["per_kind"]["invisible"]["paired_detection_gain"]["mean"] == 0
        rows.append({"model": name, **{k: summaries[name][k]["mean"] for k in keys}})
    paired = {}
    for suffix in ("", "@episode"):
        for model in ("if_rich", "hgb_rich", "mlp_rich"):
            name = model + suffix
            paired[name + "_minus_causal_structure"] = interval([
                r["models"][name]["behavior_paired_gain"] -
                r["models"]["causal_structure" + suffix]["behavior_paired_gain"] for r in records])
    counts = {"test_runs": sum(r["test_runs"] for r in records),
              "attack_episodes": sum(r["test_episodes"] for r in records),
              "per_kind": len(records) * PROTOCOL["test_pairs_per_kind"],
              "negative_control_episodes": len(records) * PROTOCOL["test_pairs_per_kind"]}
    result = {"scope": PROTOCOL["scope"], "counts": counts, "models": summaries, "paired": paired,
              "seeds": sorted(r["seed"] for r in records), "commits": list(commits),
              "protocol_sha256": next(iter(hashes)),
              "mlp_warnings": {str(r["seed"]): r["mlp_warnings"] for r in records},
              "runtime_versions": [r["runtime"] for r in records],
              "bootstrap_note": "Intervals resample the 10 simulator seeds; they are not population guarantees for real networks."}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    write_csv(output / "model-comparison.csv", rows)
    lines = ["# Complex behavior benchmark", "", f"Code: `{next(iter(commits))}`.",
             f"Protocol SHA256: `{next(iter(hashes))}`.", "",
             f"{counts['test_runs']} simulated test runs, {counts['attack_episodes']} attack episodes, including {counts['negative_control_episodes']} unobservable negative controls.", "",
             "This is an abstract semantic simulation, not a live 5G core trial or a reproduction of a complete 3GPP procedure."]
    for suffix, title in [("", "Normal point calibration target 1 percent"), ("@episode", "Normal 32 tick block calibration target 1 percent")]:
        lines += ["", "## " + title, "", "Targets are empirical calibration criteria, not guaranteed test bounds.", "",
                  "| Model | Behavior event recall | Held-out recall | Point FPR | Benign matched interval alarm rate | Paired behavior detection gain |",
                  "|---|---:|---:|---:|---:|---:|"]
        names = ["integrity"] + [m + suffix for m in PROTOCOL["models"]]
        for name in names:
            s = summaries[name]
            lines.append(f"| {name} | {s['behavior_event_recall']['mean']:.2%} | {s['heldout_recall']['mean']:.2%} | {s['point_fpr']['mean']:.3%} | {s['benign_episode_alarm_rate']['mean']:.2%} | {s['behavior_paired_gain']['mean']:.2%} |")
        lines += ["", "### Per family detection", "",
                  "| Family | Raw residual max | Causal structure | Rich Isolation Forest | Rich boosting | Rich MLP |",
                  "|---|---:|---:|---:|---:|---:|"]
        for kind in PROTOCOL["kinds"]:
            values = [summaries[m + suffix]["per_kind"][kind]["recall"]["mean"] for m in ("raw_max", "causal_structure", "if_rich", "hgb_rich", "mlp_rich")]
            lines.append("| " + kind + " | " + " | ".join(f"{v:.2%}" for v in values) + " |")
    lines += ["", "## Interpretation", "",
              "Joint shuffle preserves individual raw values. Order permutation, cross-NF desynchronization and sparse reorder preserve operation histograms. Detection therefore cannot be justified by merely increasing a univariate amplitude.", "",
              "Behavior labels exclude integrity-only and invisible episodes. Cross-NF desynchronization and sparse reorder are excluded from supervised attack training. All models receive causal past-and-present features; the simple structural baseline has the same learned relations available to ML.", "",
              "Matched normal twins share randomness, load and legitimate lifecycle events. Paired behavior gain subtracts an alarm in the corresponding normal interval. It is an explanatory effect measure, not a probability guarantee.", "",
              "MLP is a feedforward network over causal history features, not a Transformer or recurrent model. Iteration-limit warnings are retained in summary.json. Raw operation stages and simulated evidence roots do not establish real protocol fidelity or hardware trust.", "",
              "Point-level FPR and event alarm rate have different units. A target on one does not establish a target on the other. Negative-control alarm rates reflect incidental alerts; zero paired gain is checked.", "",
              "Hybrid variants retain every current integrity alert and are included in the CSV and JSON. The base verifier and earlier experiment source files remain unchanged. A successful workflow requires valid execution and controls, not an ML advantage."]
    report = "\n".join(lines) + "\n"
    (output / "REPORT.md").write_text(report)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/complex-final")
    args = parser.parse_args()
    aggregate(args.input, args.output)
