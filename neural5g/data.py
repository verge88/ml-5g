"""Training-only feature fitting and chronological request streams."""
import csv
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import numpy as np
from hybrid import external as old
from hybrid.fetch_data import git_blob_sha
from semantic5g.analyze import read_csv, signatures

P = json.loads(Path(__file__).with_name("protocol.json").read_text())
METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD")


@dataclass
class Stream:
    x: np.ndarray
    source: np.ndarray
    destination: np.ndarray
    time: np.ndarray
    y: np.ndarray
    rows: list
    name: str

    def __len__(self):
        return len(self.y)

    def subset(self, mask, name=None):
        ids = np.flatnonzero(mask)
        return Stream(self.x[ids], self.source[ids], self.destination[ids], self.time[ids],
            self.y[ids], [self.rows[i] for i in ids], name or self.name)


def read_packets(path):
    path = Path(path)
    expected = old.PROTOCOL["external_dataset"]["files"][path.name]
    if git_blob_sha(path.read_bytes()) != expected:
        raise ValueError("Pinned source blob mismatch: " + path.name)
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [dict(r, raw_row=i) for i, r in enumerate(csv.DictReader(f))]
    return sorted(rows, key=lambda r: old.number(r["frame.time_epoch"]))


def hash_bin(text, n):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % n


def api_vector(row):
    # Fixed hash mappings do not learn a vocabulary from held-out operations.
    result = np.zeros(len(METHODS) + P["service_hash_bins"] + P["resource_hash_bins"], dtype=np.float32)
    if row["method"] in METHODS:
        result[METHODS.index(row["method"])] = 1
    service = row["path"].split("/")[1] if "/" in row["path"] else ""
    result[len(METHODS) + hash_bin(service, P["service_hash_bins"])] = 1
    result[len(METHODS) + P["service_hash_bins"] + hash_bin(row["path"], P["resource_hash_bins"])] = 1
    return result


def windows(x, length):
    """Right-padded causal windows; the current event is the last valid token."""
    result = np.zeros((len(x), length, x.shape[1]), dtype=np.float32)
    valid = np.zeros(len(x), dtype=np.int64)
    for i in range(len(x)):
        start = max(0, i - length + 1)
        valid[i] = i - start + 1
        result[i, :valid[i]] = x[start:i+1]
    return result, valid


def prepare(data, source_name, target_name, with_api):
    data = Path(data)
    packets = {name: read_packets(data / name) for name in ("bridge_normal.csv", source_name, target_name)}
    normal = packets["bridge_normal.csv"]
    te, ce = int(P["normal_train_fraction"] * len(normal)), int(P["normal_calibration_end_fraction"] * len(normal))
    training_rows = {r["raw_row"] for r in normal[:te]}
    calibration_rows = {r["raw_row"] for r in normal[te + P["embargo_packets"]:ce]}
    test_rows = {r["raw_row"] for r in normal[ce + P["embargo_packets"]:]}
    reference = old.memory(normal[:te])
    raw_features = {name: old.features(rows, reference) for name, rows in packets.items()}
    # Same robust numeric fitting on training-normal packets as the prior suite.
    center = np.median(raw_features["bridge_normal.csv"][:te], axis=0)
    scale = np.maximum(1.4826 * np.median(abs(raw_features["bridge_normal.csv"][:te] - center), axis=0), .1)
    # Identity vocabulary uses training-normal and source TRAINING capture only.
    train_packets = normal[:te] + packets[source_name]
    names = sorted({r[k] for r in train_packets for k in ("ip.src", "ip.dst") if r[k]})
    nodes = {name: i+1 for i, name in enumerate(names)}
    n_nodes = len(nodes) + 1  # 0 is a single frozen unknown-node bucket.
    streams = {}
    for name, rows in packets.items():
        features = raw_features[name]
        features = np.clip((features - center) / scale, -20, 20).astype(np.float32)
        by_raw_row = {r["raw_row"]: i for i, r in enumerate(rows)}
        requests, _ = read_csv(data / name)
        # Stable chronology, retaining multiple requests from the same packet.
        requests.sort(key=lambda r: r["time"])
        xs, sources, destinations = [], [], []
        for r in requests:
            src, dst = nodes.get(r["source"], 0), nodes.get(r["destination"], 0)
            identity = np.zeros(2*n_nodes, dtype=np.float32)
            identity[src] = 1; identity[n_nodes+dst] = 1
            parts = [features[by_raw_row[r["packet_row"]]], identity]
            if with_api:
                parts.append(api_vector(r))
            xs.append(np.concatenate(parts)); sources.append(src); destinations.append(dst)
        streams[name] = Stream(np.asarray(xs, dtype=np.float32), np.asarray(sources, dtype=np.int64),
            np.asarray(destinations, dtype=np.int64), np.asarray([r["time"] for r in requests], dtype=np.float64),
            np.asarray([r["label"] != "normal" for r in requests], dtype=np.float32), requests, name)
    ns = streams["bridge_normal.csv"]
    train_normal = ns.subset([r["packet_row"] in training_rows for r in ns.rows], "normal_train")
    calibration = ns.subset([r["packet_row"] in calibration_rows for r in ns.rows], "normal_calibration")
    holdout = ns.subset([r["packet_row"] in test_rows for r in ns.rows], "normal_holdout")
    bank_edge = {signatures(r)[0] for r in train_normal.rows}
    bank_operation = {signatures(r)[1] for r in train_normal.rows}
    info = dict(n_nodes=n_nodes, node_vocabulary=nodes, numeric_feature_names=old.FEATURES,
        train_packets=te, train_normal_requests=len(train_normal), calibration_requests=len(calibration),
        holdout_requests=len(holdout), source_training_requests=len(streams[source_name]),
        target_requests=len(streams[target_name]), input_dim=train_normal.x.shape[1],
        with_api=with_api, normal_training_row_sha256=hashlib.sha256(json.dumps(sorted(training_rows)).encode()).hexdigest(),
        metadata_sha256=hashlib.sha256(center.tobytes()+scale.tobytes()).hexdigest(),
        numeric_center=center.tolist(), numeric_scale=scale.tolist(),
        reference_source_counts=dict(reference[0]), reference_destination_counts=dict(reference[1]),
        reference_edge_counts=[dict(source=a,destination=b,count=n) for (a,b),n in sorted(reference[2].items())])
    return (train_normal, streams[source_name]), calibration, {target_name: streams[target_name], "normal_holdout": holdout}, info, (bank_edge, bank_operation)
