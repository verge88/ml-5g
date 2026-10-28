"""Normal-only learned relations and causal operation history shared by models."""
from collections import Counter, defaultdict
import numpy as np
from sklearn.linear_model import Ridge
from .generator import K, N, PROTOCOL


def flatten(traces, field):
    arrays = [getattr(r, field) for r in traces]
    return np.concatenate([a.reshape((-1,) + a.shape[2:]) for a in arrays])


def lag(values, steps):
    out = np.full_like(values, -1)
    out[steps:] = values[:-steps]
    return out


def trailing_mean(values, window):
    sums = np.concatenate([np.zeros_like(values[:1]), np.cumsum(values, axis=0)], axis=0)
    stop = np.arange(1, len(values) + 1)
    start = np.maximum(0, stop - window)
    shape = (len(values),) + (1,) * (values.ndim - 1)
    return (sums[stop] - sums[start]) / (stop - start).reshape(shape)


def robust_fit(x, floor=.1):
    center = np.median(x, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(x - center), axis=0), floor)
    return center, scale


class LearnedFeatures:
    def __init__(self, normal):
        x, c = flatten(normal, "x"), flatten(normal, "context")
        self.expected = Ridge(alpha=1.).fit(c, x)
        residual = x - self.expected.predict(c)
        self.center, self.scale = robust_fit(residual, .001)
        z = (residual - self.center) / self.scale
        self.covariance = np.cov(z.T) + .01 * np.eye(z.shape[1])
        self.precision = np.linalg.inv(self.covariance)
        self.joint = Ridge(alpha=1.).fit(z[:, :1], z[:, 1])
        self.peer = np.full(N, -1, dtype=int)
        for nf in range(N):
            agreement = []
            for other in range(N):
                if nf == other:
                    agreement.append(-1.)
                    continue
                matches, count = 0, 0
                for r in normal:
                    valid = ~r.phase[1:, nf] & ~r.phase[:-1, other]
                    matches += int(((r.tokens[1:, nf] == r.tokens[:-1, other]) & valid).sum())
                    count += int(valid.sum())
                agreement.append(matches / count)
            best = int(np.argmax(agreement))
            if agreement[best] > .8:
                self.peer[nf] = best
        self.transitions = defaultdict(Counter)
        self.edges = defaultdict(Counter)
        self.correspondence = defaultdict(Counter)
        for r in normal:
            previous, before = lag(r.tokens, 1), lag(r.tokens, 2)
            for t in range(len(r.tokens)):
                for nf in range(N):
                    phase = int(r.phase[t, nf])
                    key = (nf, phase, int(before[t, nf]), int(previous[t, nf]))
                    self.transitions[key][int(r.tokens[t, nf])] += 1
                    self.edges[(nf, phase)][int(r.destinations[t, nf])] += 1
                    peer = self.peer[nf]
                    if t and peer >= 0:
                        mismatch = int(r.tokens[t, nf] != r.tokens[t - 1, peer])
                        gkey = (nf, phase, int(r.phase[t - 1, peer]))
                        self.correspondence[gkey][mismatch] += 1

    @staticmethod
    def surprise(counter, value, vocabulary):
        return -np.log((counter.get(value, 0) + .5) / (sum(counter.values()) + .5 * vocabulary))

    def transform(self, r):
        ticks = len(r.tokens)
        z = ((r.x.reshape(-1, 6) - self.expected.predict(r.context.reshape(-1, r.context.shape[-1])))
             - self.center) / self.scale
        mahal = np.sqrt(np.maximum(0, np.einsum("bi,ij,bj->b", z, self.precision, z))).reshape(ticks, N)
        joint = np.abs(z[:, 1] - self.joint.predict(z[:, :1])).reshape(ticks, N)
        z = z.reshape(ticks, N, 6)
        previous, before = lag(r.tokens, 1), lag(r.tokens, 2)
        transition = np.zeros((ticks, N))
        edge = np.zeros_like(transition)
        graph = np.zeros_like(transition)
        graph_mismatch = np.zeros_like(transition)
        support = np.zeros_like(transition)
        for t in range(ticks):
            for nf in range(N):
                phase = int(r.phase[t, nf])
                key = (nf, phase, int(before[t, nf]), int(previous[t, nf]))
                counts = self.transitions.get(key, {})
                transition[t, nf] = self.surprise(counts, int(r.tokens[t, nf]), K)
                support[t, nf] = np.log1p(sum(counts.values()))
                edge[t, nf] = self.surprise(self.edges.get((nf, phase), {}), int(r.destinations[t, nf]), N)
                peer = self.peer[nf]
                if t and peer >= 0:
                    mismatch = int(r.tokens[t, nf] != r.tokens[t - 1, peer])
                    gkey = (nf, phase, int(r.phase[t - 1, peer]))
                    graph[t, nf] = self.surprise(self.correspondence.get(gkey, {}), mismatch, 2)
                    graph_mismatch[t, nf] = mismatch
        tm16, tm32 = trailing_mean(transition, 16), trailing_mean(transition, 32)
        gm16 = trailing_mean(graph, 16)
        components = np.stack([np.max(np.abs(z), axis=2), mahal, joint, tm16, tm32, gm16, edge], axis=2)
        mean = trailing_mean(z, 16)
        std = np.sqrt(np.maximum(0, trailing_mean(z * z, 16) - mean * mean))
        onehot = [np.eye(K + 1)[a + 1] for a in (r.tokens, previous, before)]
        scalar = np.stack([transition, edge, graph, graph_mismatch, support, joint, mahal, tm16, tm32, gm16], axis=2)
        rich = np.concatenate([z, mean, std, *onehot, r.context, np.eye(N)[r.destinations], scalar], axis=2)
        return rich, components
