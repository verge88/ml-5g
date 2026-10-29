"""Independent implementations of LSTM, causal Transformer, GraphSAGE and TGN.

TGN follows the memory/message/aggregator/update/temporal-attention design.
Graphs and memory observe completed previous minibatches only. No labels enter
either online state. This is a supervised event classifier, not a link predictor.
"""
from collections import deque
import math
import torch
from torch import nn


def head(width, hidden, dropout):
    return nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))


class LSTMDetector(nn.Module):
    def __init__(self, input_dim, hidden=48, dropout=.1, **kwargs):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden, batch_first=True)
        self.classifier = head(hidden, hidden, dropout)

    def forward(self, x, lengths):
        h, _ = self.rnn(x)
        return self.classifier(h[torch.arange(len(x), device=x.device), lengths-1]).squeeze(-1)


class TransformerDetector(nn.Module):
    def __init__(self, input_dim, hidden=48, dropout=.1, window=16, heads=4, layers=2, **kwargs):
        super().__init__()
        self.project = nn.Linear(input_dim, hidden)
        self.position = nn.Embedding(window, hidden)
        # Independent construction avoids identical initial weights of cloned layers.
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(hidden, heads,
            dim_feedforward=2*hidden, dropout=dropout, batch_first=True) for _ in range(layers)])
        self.classifier = head(hidden, hidden, dropout)

    def forward(self, x, lengths):
        positions = torch.arange(x.shape[1], device=x.device)
        h = self.project(x) + self.position(positions)[None]
        causal = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), diagonal=1)
        padding = positions[None] >= lengths[:, None]
        for layer in self.layers:
            h = layer(h, src_mask=causal, src_key_padding_mask=padding)
        return self.classifier(h[torch.arange(len(x), device=x.device), lengths-1]).squeeze(-1)


class GraphSAGEDetector(nn.Module):
    """Two mean-aggregation GraphSAGE layers on a historical graph snapshot."""
    def __init__(self, input_dim, n_nodes, hidden=48, dropout=.1, layers=2, **kwargs):
        super().__init__()
        self.n_nodes, self.input_dim = n_nodes, input_dim
        self.project = nn.Linear(input_dim, hidden)
        self.identity = nn.Embedding(n_nodes, hidden)
        self.self_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.neighbor_layers = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])
        self.classifier = head(2*hidden+input_dim, hidden, dropout)
        self.reset()

    def reset(self):
        device = self.project.weight.device
        self.totals = torch.zeros(self.n_nodes, self.input_dim, device=device)
        self.counts = torch.zeros(self.n_nodes, 1, device=device)
        self.adjacency = torch.zeros(self.n_nodes, self.n_nodes, device=device)

    def forward(self, x, source, destination, time=None):
        node_x = self.totals / self.counts.clamp_min(1)
        h = self.project(node_x) + self.identity(torch.arange(self.n_nodes, device=x.device))
        a = self.adjacency / self.adjacency.sum(1, keepdim=True).clamp_min(1)
        for own, neighbor in zip(self.self_layers, self.neighbor_layers):
            h = torch.relu(own(h) + neighbor(a @ h))
        return self.classifier(torch.cat((h[source], h[destination], x), dim=-1)).squeeze(-1)

    @torch.no_grad()
    def observe(self, x, source, destination, time=None):
        ones = torch.ones(len(x), 1, device=x.device)
        for ids in (source, destination):
            self.totals.index_add_(0, ids, x.detach())
            self.counts.index_add_(0, ids, ones)
        # Undirected neighborhood; current directed endpoints remain in the readout.
        self.adjacency[source, destination] = 1
        self.adjacency[destination, source] = 1


class TimeEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.frequency = nn.Parameter(torch.logspace(0, -3, width))
        self.phase = nn.Parameter(torch.zeros(width))

    def forward(self, elapsed):
        return torch.cos(torch.log1p(elapsed.clamp_min(0)).unsqueeze(-1) * self.frequency + self.phase)


class TGNDetector(nn.Module):
    """Persistent GRU memory, last raw message aggregator, temporal attention.

Raw messages from batch k are transformed and applied differentiably when
scoring k+1. State is detached between batches; the updater still gets gradients.
Every prediction in k uses only graph observations from batches < k.
"""
    def __init__(self, input_dim, n_nodes, hidden=48, dropout=.1, neighbors=8, time_dim=8, heads=4, **kwargs):
        super().__init__()
        self.input_dim, self.n_nodes, self.hidden, self.k = input_dim, n_nodes, hidden, neighbors
        self.raw_dim = 2*hidden + input_dim + 2
        self.message = nn.Sequential(nn.Linear(self.raw_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.updater = nn.GRUCell(hidden, hidden)
        self.identity = nn.Embedding(n_nodes, hidden)
        self.time_encoder = TimeEncoder(time_dim)
        self.edge = nn.Linear(input_dim, hidden)
        self.neighbor_project = nn.Linear(2*hidden + time_dim, hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.classifier = head(2*hidden + input_dim, hidden, dropout)
        self.reset()

    def reset(self):
        device = self.edge.weight.device
        self.memory = torch.zeros(self.n_nodes, self.hidden, device=device)
        self.pending = torch.zeros(self.n_nodes, self.raw_dim, device=device)
        self.pending_mask = torch.zeros(self.n_nodes, dtype=torch.bool, device=device)
        self.pending_time = torch.zeros(self.n_nodes, dtype=torch.float64, device=device)
        self.last_update = torch.zeros(self.n_nodes, dtype=torch.float64, device=device)
        self.histories = [deque(maxlen=self.k) for _ in range(self.n_nodes)]
        self.history_nodes = torch.zeros(self.n_nodes, self.k+1, dtype=torch.long, device=device)
        self.history_edges = torch.zeros(self.n_nodes, self.k+1, self.input_dim, device=device)
        self.history_times = torch.zeros(self.n_nodes, self.k+1, dtype=torch.float64, device=device)
        self.history_mask = torch.ones(self.n_nodes, self.k+1, dtype=torch.bool, device=device)
        self.history_mask[:, 0] = False
        self.active_memory = None

    def updated_memory(self):
        ids = self.pending_mask.nonzero(as_tuple=True)[0]
        memory = self.memory
        if len(ids):
            new = self.updater(self.message(self.pending[ids]), memory[ids])
            memory = memory.index_copy(0, ids, new)
        return memory

    def embed(self, ids, time, memory):
        neighbor_ids = self.history_nodes[ids]
        edges = self.history_edges[ids]
        stamps = self.history_times[ids]
        mask = self.history_mask[ids]
        if torch.any((stamps > time[:, None]) & ~mask):
            raise ValueError("Future neighbor in TGN history")
        elapsed = (time[:, None] - stamps).clamp_min(0).float()
        elapsed[:, 0] = 0
        own = memory[ids] + self.identity(ids)
        neighbor = memory[neighbor_ids] + self.identity(neighbor_ids)
        features = torch.cat((neighbor, self.edge(edges), self.time_encoder(elapsed)), dim=-1)
        kv = self.neighbor_project(features)
        # The dummy is exactly a zero context, rather than an unknown-node message.
        kv = kv.clone(); kv[:, 0] = 0
        context, _ = self.attention(own[:, None], kv, kv, key_padding_mask=mask, need_weights=False)
        return own + context[:, 0]

    def forward(self, x, source, destination, time):
        self.active_memory = self.updated_memory()
        src = self.embed(source, time, self.active_memory)
        dst = self.embed(destination, time, self.active_memory)
        return self.classifier(torch.cat((src, dst, x), dim=-1)).squeeze(-1)

    @torch.no_grad()
    def observe(self, x, source, destination, time):
        if self.active_memory is None:
            raise RuntimeError("Score a batch before observing it")
        self.memory = self.active_memory.detach()
        self.last_update[self.pending_mask] = self.pending_time[self.pending_mask]
        self.pending_mask.zero_()
        # Last-message aggregation is chronological, including repeated nodes.
        nodes = torch.stack((source, destination), dim=1).reshape(-1)
        others = torch.stack((destination, source), dim=1).reshape(-1)
        stamps = time.repeat_interleave(2)
        previous = self.last_update[nodes]
        delta = torch.where(previous > 0, (stamps-previous).clamp_min(0), torch.zeros_like(stamps))
        direction = torch.tensor([1., -1.], device=x.device).repeat(len(x))
        extra = torch.stack((torch.log1p(delta).float(), direction), dim=-1)
        raw = torch.cat((self.memory[nodes], self.memory[others], x.detach().repeat_interleave(2, dim=0), extra), dim=-1)
        last = {node: i for i, node in enumerate(nodes.tolist())}
        touched = torch.tensor(list(last), dtype=torch.long, device=x.device)
        indices = torch.tensor(list(last.values()), dtype=torch.long, device=x.device)
        self.pending[touched] = raw[indices]
        self.pending_time[touched] = stamps[indices]
        self.pending_mask[touched] = True
        for i, (src, dst) in enumerate(zip(source.tolist(), destination.tolist())):
            event_x = x[i].detach().clone()
            self.histories[src].append((dst, float(time[i]), event_x))
            self.histories[dst].append((src, float(time[i]), event_x))
        for node in last:
            history = list(self.histories[node]); n = len(history)
            self.history_nodes[node, 1:n+1] = torch.tensor([h[0] for h in history], device=x.device)
            self.history_edges[node, 1:n+1] = torch.stack([h[2] for h in history])
            self.history_times[node, 1:n+1] = torch.tensor([h[1] for h in history], dtype=torch.float64, device=x.device)
            self.history_mask[node, 1:n+1] = False
        self.active_memory = None


def make_model(name, input_dim, n_nodes, protocol):
    args = dict(input_dim=input_dim, n_nodes=n_nodes, hidden=protocol["hidden"], dropout=protocol["dropout"],
        window=protocol["sequence_length"], heads=protocol["attention_heads"],
        neighbors=protocol["temporal_neighbors"], time_dim=protocol["time_encoding_dim"])
    if name == "lstm":
        return LSTMDetector(**args)
    if name == "transformer":
        return TransformerDetector(**args, layers=protocol["transformer_layers"])
    if name == "gnn":
        return GraphSAGEDetector(**args, layers=protocol["graphsage_layers"])
    if name == "tgn":
        return TGNDetector(**args)
    raise ValueError(name)
