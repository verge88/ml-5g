"""Train/evaluate fixed cross-capture detectors without executing any attacks."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
import joblib
from semantic5g.analyze import candidate, signatures
from .data import P, prepare, windows
from .models import make_model


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def tensor_stream(stream, sequential=False, window=16):
    values = dict(x=torch.from_numpy(stream.x), source=torch.from_numpy(stream.source),
        destination=torch.from_numpy(stream.destination), time=torch.from_numpy(stream.time), y=torch.from_numpy(stream.y))
    if sequential:
        history, lengths = windows(stream.x, window)
        values.update(history=torch.from_numpy(history), lengths=torch.from_numpy(lengths))
    return values


def forward(model, tensors, start, end, sequential):
    if sequential:
        return model(tensors["history"][start:end], tensors["lengths"][start:end])
    return model(tensors["x"][start:end], tensors["source"][start:end], tensors["destination"][start:end], tensors["time"][start:end])


def observe(model, tensors, start, end):
    model.observe(tensors["x"][start:end], tensors["source"][start:end], tensors["destination"][start:end], tensors["time"][start:end])


def fit_neural(model, streams, architecture, seed, config=P):
    sequential = architecture in ("lstm", "transformer")
    tensors = [tensor_stream(s, sequential, config["sequence_length"]) for s in streams]
    total = sum(len(s) for s in streams)
    positives = sum(float(s.y.sum()) for s in streams)
    if not 0 < positives < total:
        raise ValueError("Training must contain both original label classes")
    pos_weight = (total-positives)/positives
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    history, gradient_groups = [], {}
    rng = np.random.default_rng(seed)
    for epoch in range(config["epochs"]):
        model.train(); total_loss = 0.; points = 0
        for stream_index in rng.permutation(len(streams)):
            stream, inputs = streams[stream_index], tensors[stream_index]
            if not sequential:
                model.reset()
            for start in range(0, len(stream), config["batch_size"]):
                end = min(start+config["batch_size"], len(stream))
                optimizer.zero_grad(set_to_none=True)
                logits = forward(model, inputs, start, end, sequential)
                loss = loss_fn(logits, inputs["y"][start:end])
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                if architecture == "tgn":
                    for group in ("message", "updater", "attention", "time_encoder"):
                        value = sum(float(p.grad.abs().sum()) for name,p in model.named_parameters()
                            if name.startswith(group+".") and p.grad is not None)
                        gradient_groups[group] = max(gradient_groups.get(group, 0.), value)
                nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                optimizer.step()
                if not sequential:
                    observe(model, inputs, start, end)
                total_loss += float(loss.detach())*(end-start); points += end-start
        history.append(total_loss/points)
    changed = {name: bool(torch.any(value.detach() != initial[name])) for name,value in model.named_parameters()}
    if architecture == "tgn" and not all(gradient_groups.get(group, 0.) > 0 for group in ("message", "updater", "attention", "time_encoder")):
        raise ValueError("TGN contains an untrained memory/attention module")
    return dict(epoch_loss=history, positive_weight=pos_weight, training_points=total,
        training_positive_requests=int(positives), changed_parameter_tensors=sum(changed.values()),
        parameter_tensors=len(changed), max_gradient_l1_by_module=gradient_groups)


@torch.no_grad()
def predict(model, stream, architecture, config=P):
    model.eval()
    sequential = architecture in ("lstm", "transformer")
    tensors = tensor_stream(stream, sequential, config["sequence_length"])
    if not sequential:
        model.reset()
    scores = []
    for start in range(0, len(stream), config["batch_size"]):
        end = min(start+config["batch_size"], len(stream))
        logits = forward(model, tensors, start, end, sequential)
        scores.append(logits.sigmoid().numpy())
        if not sequential:
            observe(model, tensors, start, end)
    result = np.concatenate(scores)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite prediction")
    return result


def metrics(stream, scores, threshold, banks):
    flags = scores > threshold
    normal = np.array([r["label"] == "normal" for r in stream.rows])
    candidates = np.array([candidate(r) for r in stream.rows])
    broad_semantic = np.array([r["label"] not in ("normal", "portscan") for r in stream.rows])
    all_labels = ~normal
    known_edge = np.array([signatures(r)[0] in banks[0] for r in stream.rows])
    results = {}
    for scope, positive in (("all_request_labels", all_labels), ("semantic_label_requests", broad_semantic),
                            ("producer_candidates", candidates), ("candidate_on_known_edge", candidates & known_edge)):
        selected = normal | positive
        y = positive[selected]
        s = scores[selected]
        tp, fp, n, p = int(flags[positive].sum()), int(flags[normal].sum()), int(normal.sum()), int(positive.sum())
        results[scope] = dict(positive_requests=p, detected_requests=tp, recall=tp/p if p else None,
            normal_requests=n, false_alerts=fp, false_alert_rate=fp/n if n else None,
            precision=tp/(tp+fp) if tp+fp else None,
            pr_auc=float(average_precision_score(y,s)) if y.any() and (~y).any() else None,
            roc_auc=float(roc_auc_score(y,s)) if y.any() and (~y).any() else None)
    results["per_label"] = {label: dict(requests=int(sum(r["label"] == label for r in stream.rows)),
        alerts=int(sum(bool(flag) for r,flag in zip(stream.rows,flags) if r["label"] == label)))
        for label in sorted({r["label"] for r in stream.rows})}
    results["per_candidate_family"] = {family: dict(requests=int(sum(candidate(r) and r["family"] == family for r in stream.rows)),
        alerts=int(sum(bool(flag) for r,flag in zip(stream.rows,flags) if candidate(r) and r["family"] == family)))
        for family in sorted({r["family"] for r in stream.rows if candidate(r)})}
    return results


def run(data, output, architecture, seeds=None, config=P):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["torch_threads"])
    runs = []
    for fold, source_name, target_name in config["folds"]:
        for input_name in config["inputs"]:
            streams, calibration, tests, info, banks = prepare(data, source_name, target_name, input_name == "metadata_api")
            for seed in seeds or config["seeds"]:
                seed_all(seed)
                directory = output/fold/input_name/str(seed)
                directory.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                if architecture == "hgb":
                    xx, yy = np.concatenate([s.x for s in streams]), np.concatenate([s.y for s in streams])
                    positive_weight = float((len(yy)-yy.sum())/yy.sum())
                    model = HistGradientBoostingClassifier(max_iter=config["hgb_iterations"],
                        max_depth=config["hgb_max_depth"], learning_rate=.08, l2_regularization=1.,
                        early_stopping=False, random_state=seed)
                    model.fit(xx, yy, sample_weight=np.where(yy > 0, positive_weight, 1.))
                    training = dict(training_points=len(yy), training_positive_requests=int(yy.sum()), positive_weight=positive_weight)
                    score_fn = lambda s: model.predict_proba(s.x)[:,1]
                    parameters = None
                    joblib.dump(model, directory/"model.joblib")
                else:
                    model = make_model(architecture, info["input_dim"], info["n_nodes"], config)
                    parameters = sum(p.numel() for p in model.parameters())
                    training = fit_neural(model, streams, architecture, seed, config)
                    score_fn = lambda s: predict(model, s, architecture, config)
                    # Only learned parameters; capture state is reset, not persisted.
                    torch.save(model.state_dict(), directory/"weights.pt")
                train_seconds = time.perf_counter()-started
                cal_scores = score_fn(calibration)
                threshold = float(np.quantile(cal_scores, 1-config["calibration_fpr"], method="higher"))
                np.savez_compressed(directory/"calibration-scores.npz", score=cal_scores,
                    raw_row=np.array([r["packet_row"] for r in calibration.rows]))
                results, latencies = {}, {}
                for capture, stream in tests.items():
                    start = time.perf_counter(); score = score_fn(stream)
                    latencies[capture] = (time.perf_counter()-start)/len(stream)
                    results[capture] = metrics(stream, score, threshold, banks)
                    results[capture]["baselines"] = {}
                    for i,name in enumerate(("edge_policy", "operation_policy")):
                        policy = np.array([float(signatures(r)[i] not in banks[i]) for r in stream.rows])
                        results[capture]["baselines"][name] = metrics(stream, policy, .5, banks)
                    np.savez_compressed(directory/(capture.replace(".csv", "")+"-scores.npz"),
                        score=score, label=stream.y, raw_row=np.array([r["packet_row"] for r in stream.rows]),
                        frame=np.array([r["frame"] for r in stream.rows]))
                record = dict(architecture=architecture, fold=fold, source=source_name, target=target_name,
                    input=input_name, seed=seed, features=info, parameters=parameters, threshold=threshold,
                    calibration_observed_fpr=float((cal_scores > threshold).mean()), training=training,
                    train_seconds=train_seconds, inference_seconds_per_request=latencies, results=results)
                (directory/"checkpoint.json").write_text(json.dumps(record, indent=2))
                runs.append(record)
                print(json.dumps(dict(model=architecture, fold=fold, input=input_name, seed=seed,
                    train_seconds=round(train_seconds,2), target_candidates=results[target_name]["producer_candidates"])), flush=True)
    summary = dict(code_commit=os.getenv("GITHUB_SHA", "local"), protocol=config,
        protocol_sha256=hashlib.sha256(Path(__file__).with_name("protocol.json").read_bytes()).hexdigest(),
        torch_version=torch.__version__, numpy_version=np.__version__, architecture=architecture, runs=runs,
        authorization_and_cu="UNKNOWN: no independent authenticated caller/authorization/CU evidence")
    (output/"summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", choices=P["models"], required=True)
    p.add_argument("--seeds", type=int, nargs="+")
    a = p.parse_args()
    with threadpool_limits(limits=P["torch_threads"]):
        run(a.data, a.output, a.model, a.seeds)
