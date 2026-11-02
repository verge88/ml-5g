"""Causal FCFS queue simulator. Time/work units are abstract, not measured ms."""
import argparse
import hashlib
import heapq
import json
import platform
import subprocess
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
PLAN = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))


def exogenous(seed, load):
    """Policy-independent common random numbers for paired causal interventions."""
    rng = np.random.default_rng(seed)
    n = PLAN['tick_count']
    arrivals = []
    for t in range(n):
        count = rng.poisson(load)
        times = np.sort(rng.uniform(t, t + 1, count))
        arrivals.append(list(zip(times.tolist(), rng.exponential(1, count).tolist())))
    return arrivals, rng.normal(0, .08, n), rng.uniform(0, 1, n)


class Gaussian:
    def fit(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        self.mean = np.array([x[y == k].mean(0) for k in range(4)])
        self.var = np.array([np.maximum(x[y == k].var(0), .04) for k in range(4)])
        if not np.isfinite(self.mean).all():
            raise ValueError('Missing training classes')
        return self

    def posterior(self, passive, probe=None):
        a = np.log1p(max(0., passive))
        v = -.5 * (np.log(self.var[:, 0]) + (a-self.mean[:, 0])**2/self.var[:, 0])
        if probe is not None:
            b = np.log1p(max(0., probe))
            v += -.5 * (np.log(self.var[:, 1]) + (b-self.mean[:, 1])**2/self.var[:, 1])
        v = np.exp(v - v.max())
        return v / v.sum()


def choose(policy, t, posterior, fresh, rate, work, credits, used, random_coin):
    """Only already available observations; no simulator state or labels."""
    if policy == 'passive' or credits < 1 or used >= PLAN['optional_budget']:
        return False
    if policy.startswith('guarded_') and rate + .2 * work >= .95:
        return False
    base = policy.removeprefix('guarded_')
    if base == 'periodic':
        return t % 5 == 0
    if base == 'random':
        return random_coin < .2
    return not fresh or float(posterior.max()) < .8


def simulate(seed, load, work, mechanism, policy, mode, model=None, collect=False):
    arrivals, noise, coins = exogenous(seed, load)
    pending = []
    server = 0.
    probe_server = 0.
    last_probe = None
    recent_users = deque(maxlen=12)
    counts = deque(maxlen=20)
    latencies, truth, predictions, samples, sample_labels = [], [], [], [], []
    credits, used, fresh_count, scored, pending_at_end = 1., 0, 0, 0, 0
    first_detection = None
    alarm_healthy, total_healthy = 0, 0
    max_pending = 0
    start, end = PLAN['impairment_interval']
    for t in range(PLAN['tick_count']):
        # Only completions already returned at decision time are observable.
        while pending and pending[0][0] <= t:
            done, kind, value = heapq.heappop(pending)
            if kind == 0:
                recent_users.append(value)
            else:
                last_probe = (done, value)
        counts.append(len(arrivals[t-1]) if t else 0)
        arrival_rate = sum(counts) / len(counts)
        passive = float(np.mean(recent_users)) if recent_users else 1.
        fresh = last_probe is not None and t-last_probe[0] <= PLAN['maximum_probe_age']
        optional = last_probe[1] if fresh else None
        posterior = model.posterior(passive, optional) if model else np.full(4, .25)
        credits = min(2., credits + .2)
        query = choose(policy, t, posterior, fresh, arrival_rate, work, credits, used, coins[t])
        impaired = start <= t < end
        label = mechanism if impaired else 0
        path_wait = PLAN['diagnostic_delay'] if label in (1, 3) else 0.
        app_wait = PLAN['diagnostic_delay'] if label in (2, 3) else 0.
        if query:
            used += 1
            credits -= 1
            if mode == 'shared':
                server = max(server, float(t)) + work
                done = server + app_wait + .1 + abs(noise[t])
            elif mode == 'isolated':
                probe_server = max(probe_server, float(t)) + work
                done = probe_server + app_wait + .1 + abs(noise[t])
            else:
                # Observe the same queue but do not consume its capacity.
                done = max(server, float(t)) + work + app_wait + .1 + abs(noise[t])
            heapq.heappush(pending, (done, 1, done-t))
        for arrival, service in arrivals[t]:
            server = max(server, arrival) + service
            completion = server + app_wait + path_wait
            latency = completion - arrival
            heapq.heappush(pending, (completion, 0, latency))
            if t >= PLAN['burn_in']:
                latencies.append(latency)
        if t >= PLAN['burn_in']:
            scored += 1
            fresh_count += int(fresh)
            truth.append(label)
            prediction = int(posterior.argmax())
            predictions.append(prediction)
            if label == 0:
                total_healthy += 1
                alarm_healthy += int(prediction != 0)
            elif prediction != 0 and first_detection is None:
                first_detection = t-start
            if collect and fresh:
                samples.append([np.log1p(passive), np.log1p(optional)])
                sample_labels.append(label)
        max_pending = max(max_pending, len(pending))
    pending_at_end = len(pending)
    cm = np.zeros((4, 4), dtype=int)
    np.add.at(cm, (truth, predictions), 1)
    return dict(seed=seed, load=load, probe_work=work, mechanism=mechanism, policy=policy, mode=mode,
                queries=used, optional_service_work=used*work, user_p95=float(np.quantile(latencies, .95)),
                user_mean=float(np.mean(latencies)), user_count=len(latencies), confusion=cm.tolist(),
                healthy_false_alarms=alarm_healthy, healthy_decisions=total_healthy,
                fresh_fraction=fresh_count/scored, detection_delay=first_detection,
                pending_at_horizon=pending_at_end, maximum_pending=max_pending), samples, sample_labels


def macro_f1(cm):
    a = np.asarray(cm)
    denominator = a.sum(0) + a.sum(1)
    return float(np.mean(np.divide(2*a.diagonal(), denominator,
                                  out=np.zeros(4), where=denominator != 0)))


def fit():
    samples, labels = [], []
    for seed in range(*PLAN['training_seeds']):
        for load in PLAN['loads']:
            for mechanism in range(4):
                _, x, y = simulate(seed, load, .15, mechanism, 'periodic', 'shared', collect=True)
                samples.extend(x)
                labels.extend(y)
    model = Gaussian().fit(samples, labels)
    return model, len(labels)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def bootstrap(values, seed=771):
    x = np.asarray(values)
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
    return dict(mean=float(x.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(), seeds=len(x))


def evaluate(model, seeds):
    rows = []
    for seed in seeds:
        for load in PLAN['loads']:
            for work in PLAN['probe_work']:
                for mechanism in range(4):
                    for mode in PLAN['modes']:
                        for policy in PLAN['policies']:
                            row, _, _ = simulate(seed, load, work, mechanism, policy, mode, model)
                            rows.append(row)
    return rows


def summarize(rows):
    # Experimental unit is the seed, not an individual queue event/tick.
    grouped = {}
    for row in rows:
        key = (row['load'], row['probe_work'], row['mode'], row['policy'], row['seed'])
        grouped.setdefault(key, []).append(row)
    units = {}
    for key, rr in grouped.items():
        cm = np.sum([r['confusion'] for r in rr], axis=0)
        units[key] = dict(p95=float(np.mean([r['user_p95'] for r in rr])), f1=macro_f1(cm),
                          queries=float(np.mean([r['queries'] for r in rr])),
                          fresh=float(np.mean([r['fresh_fraction'] for r in rr])),
                          fpr=sum(r['healthy_false_alarms'] for r in rr)/sum(r['healthy_decisions'] for r in rr))
    summaries = []
    for load in PLAN['loads']:
        for work in PLAN['probe_work']:
            for mode in PLAN['modes']:
                for policy in PLAN['policies']:
                    uu = [u for k, u in units.items() if k[:4] == (load, work, mode, policy)]
                    summaries.append(dict(load=load, probe_work=work, mode=mode, policy=policy,
                                          **{k:float(np.mean([u[k] for u in uu])) for k in uu[0]}))
    contrasts = []
    seeds = sorted({r['seed'] for r in rows})
    for load in PLAN['loads']:
        for work in PLAN['probe_work']:
            for mode in PLAN['modes']:
                aa = [units[(load,work,mode,'guarded_risk',s)] for s in seeds]
                bb = [units[(load,work,mode,'risk',s)] for s in seeds]
                cc = [units[(load,work,'ghost','risk',s)] for s in seeds]
                latency = bootstrap([(a['p95']-b['p95'])/b['p95'] for a,b in zip(aa,bb)])
                f1 = bootstrap([a['f1']-b['f1'] for a,b in zip(aa,bb)])
                replay_bias = bootstrap([(b['p95']-c['p95'])/c['p95'] for b,c in zip(bb,cc)])
                contrasts.append(dict(load=load, probe_work=work, mode=mode,
                    latency_relative=latency, f1_difference=f1, risk_shared_vs_ghost=replay_bias,
                    joint_success=latency['ci95'][1]<-.10 and f1['ci95'][0]>-.03))
    primary = next(c for c in contrasts if all(c[k]==v for k,v in PLAN['primary_cell'].items()))
    return dict(primary=primary, exploratory_contrasts=contrasts, summaries=summaries)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('out/probe-study'))
    p.add_argument('--stage', choices=['fit','evaluate','aggregate','all'], default='all')
    p.add_argument('--models', type=Path, default=Path('frozen'))
    p.add_argument('--input', type=Path)
    p.add_argument('--start', type=int)
    p.add_argument('--count', type=int)
    p.add_argument('--smoke', action='store_true')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    plan_hash=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()
    if a.stage in ('fit','all'):
        model,n=fit()
        calibration=[]
        for seed in range(*PLAN['calibration_seeds']):
            for mechanism in range(4):
                row,_,_=simulate(seed,.7,.15,mechanism,'periodic','shared',model)
                calibration.append(row)
        write_json(a.output/'frozen.json',dict(mean=model.mean.tolist(),var=model.var.tolist(),
                   training_rows=n,plan_sha256=plan_hash,training_seed_range=PLAN['training_seeds'],
                   calibration=calibration,parameter_selection_using_calibration=False))
        if a.stage=='fit':return
        frozen=a.output/'frozen.json'
    else:frozen=a.models/'frozen.json'
    metadata=json.loads(frozen.read_text(encoding='utf-8'))
    assert metadata['plan_sha256']==plan_hash
    assert metadata['training_seed_range']==PLAN['training_seeds']
    model=Gaussian()
    model.mean=np.array(metadata['mean']);model.var=np.array(metadata['var'])
    seed_start,seed_end=PLAN['test_seeds']
    if a.stage=='aggregate':
        rows=[]
        for path in sorted(a.input.glob('**/per_trace.json')):
            rows.extend(json.loads(path.read_text(encoding='utf-8')))
        expected=set(range(seed_start,seed_end))
        assert {r['seed'] for r in rows}==expected
        keys=[(r['seed'],r['load'],r['probe_work'],r['mechanism'],r['mode'],r['policy']) for r in rows]
        assert len(keys)==len(set(keys))==len(expected)*432
    else:
        first=a.start if a.start is not None else seed_start
        last=first+a.count if a.count is not None else (seed_start+2 if a.smoke else seed_end)
        assert seed_start<=first<last<=seed_end
        rows=evaluate(model, range(first,last))
    write_json(a.output/'per_trace.json',rows)
    result=summarize(rows)
    result['provenance']=dict(plan_sha256=plan_hash,model_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
        python=platform.python_version(),numpy=np.__version__,test_model_fits=0,simulation=True,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        smoke=a.smoke,traces=len(rows),independent_seeds=len({r['seed'] for r in rows}))
    write_json(a.output/'results.json',result)
    primary=result['primary']
    lines=['# Probe intervention computational test','',
           'Synthetic FCFS queue; units are abstract. No live 5G or crypto measurements.',
           f"Independent test seeds: {result['provenance']['independent_seeds']}; traces: {len(rows)}.",'',
           'Primary comparison: guarded_risk minus risk at load .92, probe work .60, shared queue.',
           '```json',json.dumps(primary,indent=2),'```','',
           '| Load | Work | Mode | Policy | User p95 | Macro F1 | Probes | Fresh fraction | Healthy FPR |',
           '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in result['summaries']:
        lines.append(f"|{r['load']}|{r['probe_work']}|{r['mode']}|{r['policy']}|{r['p95']:.3f}|{r['f1']:.3f}|{r['queries']:.2f}|{r['fresh']:.3f}|{r['fpr']:.3f}|")
    lines+=['','Joint success requires latency AND F1 criteria. All other cells exploratory.',
            'Ghost is an idealized noninterfering observation control, not deployment evidence.',
            'FPR is empirical and unconstrained; no 1% false-alarm guarantee is claimed.',
            'Costs exclude mandatory KIE, signing, network bytes and compute not represented by queue work.']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(primary=primary,provenance=result['provenance']),indent=2))


if __name__=='__main__':main()
