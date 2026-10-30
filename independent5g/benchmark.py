"""Fixed five-model comparison; thresholds are committed to disk before test scoring."""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from neural5g.benchmark import seed_all, tensor_stream, forward, observe, predict
from neural5g.models import make_model
from .collect import P
from .data import prepare, SLOTS


def fit_masked(model, streams, architecture, config):
    """Keep all causal history; excluded intervals contribute no training loss."""
    sequential = architecture in ("lstm", "transformer")
    tensors = [tensor_stream(s, sequential, config["sequence_length"]) for s in streams]
    masks = [torch.tensor([r["selected"] for r in s.rows]) for s in streams]
    total = sum(int(m.sum()) for m in masks)
    positives = sum(float(t["y"][m].sum()) for t,m in zip(tensors,masks))
    if not 0 < positives < total:
        raise ValueError("both training classes required")
    weight = (total-positives)/positives
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    losses, gradients = [], {}
    for _ in range(config["epochs"]):
        model.train()
        total_loss = points = 0
        for s, inputs, mask in zip(streams, tensors, masks):
            if not sequential:
                model.reset()
            for start in range(0,len(s),config["batch_size"]):
                end = min(start+config["batch_size"],len(s))
                selected = mask[start:end]
                optimizer.zero_grad(set_to_none=True)
                logits = forward(model,inputs,start,end,sequential)
                if selected.any():
                    loss = loss_fn(logits[selected], inputs["y"][start:end][selected])
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    loss.backward()
                    if architecture == "tgn":
                        for group in ("message", "updater", "attention", "time_encoder"):
                            value = sum(float(p.grad.abs().sum()) for name,p in model.named_parameters()
                                if name.startswith(group+".") and p.grad is not None)
                            gradients[group] = max(gradients.get(group,0.),value)
                    nn.utils.clip_grad_norm_(model.parameters(),config["gradient_clip"])
                    optimizer.step()
                    n = int(selected.sum())
                    total_loss += float(loss.detach())*n
                    points += n
                if not sequential:
                    observe(model,inputs,start,end)
        losses.append(total_loss/points)
    if architecture == "tgn" and not all(gradients.get(g,0.) > 0 for g in ("message","updater","attention","time_encoder")):
        raise ValueError("untrained TGN module")
    return dict(epoch_loss=losses, selected_training_rows=total, positive_rows=int(positives),
        max_gradient_l1_by_module=gradients)


def evaluate(stream, scores, threshold):
    flags = np.asarray(scores) > threshold
    normal = np.array([r["normal"] for r in stream.rows])
    fault = stream.y.astype(bool)
    selected = normal | fault
    normal_seconds = sum(r["exposure_seconds"] for r in stream.rows if r["normal"] and r["slot"] == SLOTS[0])
    if normal_seconds <= 0:
        raise ValueError("no measured normal exposure")
    episodes, detected, delays = set(), set(), []
    for r,flag in zip(stream.rows,flags):
        if r["episode"] is not None:
            key = r["episode"]
            episodes.add(key)
            if flag and key not in detected:
                detected.add(key)
                delays.append((r["monotonic_ns"]-r["fault_start_ns"])/1e9)
    # Count consecutive false-flag runs separately for each target role.
    false_episodes = 0
    previous = {}
    for r,flag in zip(stream.rows,flags):
        prior_tick,prior_flag = previous.get(r["slot"],(-2,False))
        active = r["normal"] and bool(flag)
        if active and not (prior_flag and prior_tick+1 == r["tick"]):
            false_episodes += 1
        previous[r["slot"]] = (r["tick"],active)
    tp,fp = int(flags[fault].sum()),int(flags[normal].sum())
    n,p = int(normal.sum()),int(fault.sum())
    return dict(replica=int(stream.name), normal_rows=n, fault_rows=p, true_alerts=tp, false_alerts=fp,
        recall=tp/p if p else None, fpr=fp/n, false_alerts_per_hour=fp*3600/normal_seconds,
        normal_seconds=normal_seconds, false_alarm_episodes=false_episodes,
        false_alarm_episodes_per_hour=false_episodes*3600/normal_seconds,
        fault_episodes=len(episodes), detected_episodes=len(detected),
        episode_recall=len(detected)/len(episodes) if episodes else None,
        observed_detection_delays_seconds=delays,
        delay_scope="time to flagged observation; excludes buffering and inference latency",
        average_precision=float(average_precision_score(stream.y[selected],np.asarray(scores)[selected])) if p else None)


def run(root, output, architecture, config=P):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(config["torch_threads"])
    runs = []
    for input_name in config["inputs"]:
        training,calibration,tests,info = prepare(root,input_name)
        xx = np.concatenate([s.x[[r["selected"] for r in s.rows]] for s in training])
        yy = np.concatenate([s.y[[r["selected"] for r in s.rows]] for s in training])
        for seed in config["seeds"]:
            seed_all(seed)
            folder = output/input_name/str(seed)
            folder.mkdir(parents=True,exist_ok=True)
            started = time.perf_counter()
            if architecture == "hgb":
                model = HistGradientBoostingClassifier(max_iter=config["hgb_iterations"],max_depth=config["hgb_max_depth"],
                    learning_rate=.08,l2_regularization=1.,early_stopping=False,random_state=seed)
                model.fit(xx,yy,sample_weight=np.where(yy>0,(len(yy)-yy.sum())/yy.sum(),1.))
                score_fn = lambda s:model.predict_proba(s.x)[:,1]
                training_info = dict(selected_training_rows=len(yy),positive_rows=int(yy.sum()))
                joblib.dump(model,folder/"model.joblib")
            else:
                model = make_model(architecture,info["input_dim"],info["n_nodes"],config)
                training_info = fit_masked(model,training,architecture,config)
                score_fn = lambda s:predict(model,s,architecture,config)
                torch.save(model.state_dict(),folder/"weights.pt")
            train_seconds = time.perf_counter()-started
            cal_scores = np.concatenate([score_fn(s) for s in calibration])
            threshold = float(np.quantile(cal_scores,1-config["calibration_fpr"],method="higher"))
            np.save(folder/"calibration-scores.npy",cal_scores)
            frozen = dict(model=architecture,input=input_name,seed=seed,threshold=threshold,
                calibration_observed_fpr=float((cal_scores>threshold).mean()),features=info,
                training=training_info,train_seconds=train_seconds,protocol=config)
            (folder/"frozen-before-test.json").write_text(json.dumps(frozen,indent=2))
            # No test-dependent hyperparameter, threshold, epoch or feature choice below.
            results = []
            for stream in tests:
                started = time.perf_counter()
                scores = score_fn(stream)
                elapsed = time.perf_counter()-started
                metric = evaluate(stream,scores,threshold)
                metric["inference_seconds_per_row"] = elapsed/len(stream)
                results.append(metric)
                np.savez_compressed(folder/f"test-{stream.name}.npz",score=scores,label=stream.y,
                    tick=np.array([r["tick"] for r in stream.rows]),role=stream.destination)
            frozen["results"] = results
            runs.append(frozen)
            print(json.dumps(dict(model=architecture,input=input_name,seed=seed,threshold=threshold)),flush=True)
    # A fixed direct-transport availability rule, independent of learned thresholds.
    baselines = [evaluate(s,np.array([not r["direct_ok"] for r in s.rows],dtype=float),.5) for s in tests]
    summary = dict(model=architecture,runs=runs,direct_availability_rule=baselines,
        scope=config["scope"],limitations=config["limitations"],torch_version=torch.__version__)
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model",choices=P["models"],required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=P["torch_threads"]):
        run(args.data,args.output,args.model)
