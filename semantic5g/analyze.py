"""Retrospective, normal-only operation profiling of published 5G captures.

An unfamiliar operation is a policy anomaly, not proof of unauthorized access.
CSV labels identify source annotations, not independently verified outcomes.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

REQUEST = re.compile(r"HEADERS\[(\d+)\]: (GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD) (\S+)")
RESPONSE = re.compile(r"HEADERS\[(\d+)\]: ([1-5]\d\d)(?:\s|$)")
SCP = "10.211.55.11"
REVISION = "51df94b4e88d1c9c5e499416e4b8fee5c9b93cd7"
EXPECTED = {"nrf_uuid_uir_curl": "nf_management", "udm_authvec_curl": "authentication_data",
    "udm_subscriber_data_curl": "subscription_data", "amf_dereg_ue": "deregistration_notification"}
# Service producer addresses are corroborated by ordinary forwarded requests.
# They do not establish the authenticated identity of a consumer.
PRODUCER = {"nf_management": "10.211.55.7", "authentication_data": "10.211.55.8",
    "subscription_data": "10.211.55.8", "deregistration_notification": "10.211.55.4"}


def candidate(r):
    return EXPECTED.get(r["label"]) == r["family"] and r["destination"] == PRODUCER.get(r["family"])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def template(uri):
    # Wireshark's Info column separates subsequent HTTP/2 frames with a comma.
    path = urlsplit(uri.rstrip(",")).path
    parts = []
    for part in path.split("/"):
        if part.startswith(("imsi-", "suci-", "nai-")):
            part = "{subscriber}"
        elif re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", part):
            part = "{uuid}"
        elif part.isdecimal():
            part = "{number}"
        elif re.fullmatch(r"[0-9a-fA-F]{16,}", part):
            part = "{opaque-id}"
        parts.append(part)
    # Query values are unavailable to the policy; do not mistake them for bodies.
    return "/".join(parts)


def family(path):
    if path.startswith("/nudm-ueau/") and path.endswith("/generate-auth-data"):
        return "authentication_data"
    if path.startswith("/nudm-sdm/") and path.endswith("/am-data"):
        return "subscription_data"
    if path.startswith("/namf-callback/") and path.endswith("/dereg-notify"):
        return "deregistration_notification"
    if path.startswith("/nnrf-nfm/") and "/nf-instances" in path:
        return "nf_management"
    return "other"


def read_csv(path):
    requests, replies = [], defaultdict(list)
    generation = Counter()
    skipped = 0
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            info = row["_ws.col.info"]
            client = (row["ip.src"], row["srcport"], row["ip.dst"], row["dstport"])
            if "[SYN]" in info:
                generation[client] += 1
            if "Retransmission" in info:
                skipped += 1
                continue
            for stream, method, uri in REQUEST.findall(info):
                key = (*client, generation[client], stream)
                requests.append(dict(frame=int(row["frame.number"]),
                    time=float(row["frame.time_epoch"]), source=client[0], destination=client[2],
                    method=method, path=template(uri), label=row["Label"], key=key))
            reverse = (client[2], client[3], client[0], client[1])
            for stream, status in RESPONSE.findall(info):
                replies[(*reverse, generation[reverse], stream)].append(
                    (float(row["frame.time_epoch"]), int(status)))
    occurrences = Counter(r["key"] for r in requests)
    for r in requests:
        matches = [(t, s) for t, s in replies[r["key"]] if t >= r["time"] and s >= 200]
        r["status"] = matches[0][1] if len(matches) == 1 and occurrences[r["key"]] == 1 else None
        r["response_correlation"] = "UNIQUE" if r["status"] is not None else "UNKNOWN"
        r["family"] = family(r["path"])
        r.pop("key")
    return requests, skipped


def signatures(r):
    edge = (r["source"], r["destination"])
    return edge, (*edge, r["method"], r["path"]), (r["source"], r["method"], r["path"])


def rate(n, d):
    return n / d if d else None


def csv_experiment(data, out):
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8-sig"))
    for item in manifest:
        if digest(data / item["name"]) != item["sha256"]:
            raise ValueError("Source digest mismatch: " + item["name"])
    normal_all, _ = read_csv(data / "bridge_normal.csv")
    with (data / "bridge_normal.csv").open(encoding="utf-8-sig", newline="") as f:
        normal_packets = sorted(csv.DictReader(f), key=lambda r: float(r["frame.time_epoch"]))
    training_frames = {int(r["frame.number"]) for r in normal_packets[:int(.6 * len(normal_packets))]}
    normal = [r for r in normal_all if r["frame"] in training_frames]
    banks = [set(x[i] for x in map(signatures, normal)) for i in range(3)]
    results = {"source_revision": REVISION, "source_files": manifest,
        "design": "retrospective exploratory; fixed normal capture only for profiling; no attack execution",
        "normal_training_requests": len(normal), "normal_signature_counts": [len(b) for b in banks],
        "normal_training_packets": len(training_frames), "normal_total_requests": len(normal_all),
        "limitations": ["No request bodies, tokens, peer certificates, or trusted caller roles in CSV",
            "An IP can aggregate multiple NF callers; inventory is not an identity oracle",
            "SCP forwarding hides original caller; no authenticated origin is inferred",
            "Capture annotations include interleaved normal operations and potential label spillover",
            "Metrics count observed request hops, not independent attacks or unique UEs",
            "HTTP success is not proof of unauthorized data disclosure or UE disconnection"], "captures": {}}
    methods = ["edge_novelty", "edge_operation_novelty", "caller_operation_novelty"]
    for name in ("bridge_attack_nrf.csv", "bridge_attack_smf.csv"):
        requests, skipped = read_csv(data / name)
        counts = defaultdict(Counter)
        for r in requests:
            for i, method in enumerate(methods):
                r[method] = signatures(r)[i] not in banks[i]
            r["caller_identity"] = "UNKNOWN"
            r["authorization_outcome"] = "UNKNOWN"
            r["proxied_hop"] = r["source"] == SCP
            c = counts[(r["label"], r["family"])]
            c["requests"] += 1
            c["paired_responses"] += r["status"] is not None
            c["http_2xx"] += r["status"] is not None and 200 <= r["status"] < 300
            c["proxied_hops"] += r["proxied_hop"]
            for method in methods:
                c[method] += r[method]
        normal_test = [r for r in requests if r["label"] == "normal"]
        attacks = [r for r in requests if r["label"] not in ("normal", "portscan")]
        # Exact source label and operation agreement; not a new independent ground truth.
        targeted = [r for r in attacks if EXPECTED.get(r["label"]) == r["family"]]
        candidates = [r for r in attacks if candidate(r)]
        metrics = {}
        for method in methods:
            metrics[method] = dict(normal_requests=len(normal_test), false_alerts=sum(r[method] for r in normal_test),
                false_alert_rate=rate(sum(r[method] for r in normal_test), len(normal_test)),
                all_attack_label_requests=len(attacks), all_attack_label_alerts=sum(r[method] for r in attacks),
                all_attack_label_recall=rate(sum(r[method] for r in attacks), len(attacks)),
                label_operation_agreement_requests=len(targeted),
                label_operation_agreement_alerts=sum(r[method] for r in targeted),
                label_operation_agreement_recall=rate(sum(r[method] for r in targeted), len(targeted)),
                producer_candidate_requests=len(candidates), producer_candidate_alerts=sum(r[method] for r in candidates),
                producer_candidate_recall=rate(sum(r[method] for r in candidates), len(candidates)))
        results["captures"][name] = dict(requests=len(requests), retransmission_rows_skipped=skipped,
            paired_responses=sum(r["status"] is not None for r in requests), metrics=metrics,
            by_label_operation=[dict(label=k[0], family=k[1], **v) for k, v in sorted(counts.items())])
        fields = ["frame", "time", "source", "destination", "method", "path", "label", "status", "response_correlation",
            "family", *methods, "caller_identity", "authorization_outcome", "proxied_hop"]
        with (out / (name.removesuffix(".csv") + "-requests.csv")).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(requests)
    return results


def as_list(value):
    return value if isinstance(value, list) else [value]


def h2_frames(value):
    for item in as_list(value):
        if isinstance(item, dict):
            for frame in as_list(item.get("http2.stream", [])):
                if isinstance(frame, dict):
                    yield frame


def native_experiment(json_path, pcap_path, out):
    packets = json.loads(json_path.read_text(encoding="utf-8-sig"))
    if not packets:
        raise ValueError("Empty HTTP/2 decoding cannot support a native semantic result")
    streams = defaultdict(lambda: {"request": {}, "response": {}, "request_body": bytearray(),
        "response_body": bytearray(), "frames": [], "complete": set()})
    seen = set()
    for packet in packets:
        layers = packet["_source"]["layers"]
        tcp = layers.get("tcp", {})
        tcp_stream = str(tcp.get("tcp.stream", ""))
        source_port = str(tcp.get("tcp.srcport", ""))
        for h in h2_frames(layers.get("http2", [])):
            sid = str(h.get("http2.streamid", "0"))
            if sid == "0" or h.get("http2.type") not in ("0", "1"):
                continue
            headers = {x["http2.header.name"]: x.get("http2.header.value", "")
                for x in as_list(h.get("http2.header", [])) if isinstance(x, dict) and "http2.header.name" in x}
            # A server response originates from the recorded SBI listening port.
            direction = "response" if source_port == "7777" else "request"
            if ":method" in headers:
                direction = "request"
            if ":status" in headers:
                direction = "response"
            s = streams[(tcp_stream, sid)]
            body = h.get("http2.data.data", "")
            identity = (tcp_stream, sid, direction, str(tcp.get("tcp.seq", "")), str(h.get("http2.type")), body)
            if identity in seen:
                continue
            seen.add(identity)
            s[direction].update(headers)
            if body:
                s[direction + "_body"].extend(bytes.fromhex(body.replace(":", "")))
            flags = h.get("http2.flags_tree", {})
            if flags.get("http2.flags.end_stream") == "1":
                s["complete"].add(direction)
            s["frames"].append(int(layers["frame"]["frame.number"]))
    outcomes, events = Counter(), []
    valid_json = Counter()
    services = Counter()
    negative_responses = Counter()
    profile_binding = Counter()
    for (connection, stream), s in streams.items():
        req, res = s["request"], s["response"]
        path = req.get(":path", "")
        if path:
            services[template(path).split("/")[1]] += 1
            if res.get(":status", "0").isdigit() and int(res.get(":status", "0")) >= 400:
                negative_responses[(path.split("/")[1], res[":status"])] += 1
        objects = {}
        for direction in ("request", "response"):
            if s[direction + "_body"] and "json" in s[direction].get("content-type", "").lower():
                try:
                    objects[direction] = json.loads(s[direction + "_body"])
                    valid_json[direction] += 1
                except (ValueError, UnicodeDecodeError):
                    valid_json[direction + "_decode_failure"] += 1
        request_obj = objects.get("request", {})
        if req.get(":method") == "PUT" and re.fullmatch(r"/nnrf-nfm/v1/nf-instances/[^/]+", path):
            instance = request_obj.get("nfInstanceId") if isinstance(request_obj, dict) else None
            profile_binding["MATCH" if instance == path.rsplit("/", 1)[1] else "UNKNOWN" if instance is None else "MISMATCH"] += 1
        obj = objects.get("response", {})
        auth = obj.get("authResult") if isinstance(obj, dict) else None
        if auth is not None:
            status = res.get(":status", "UNKNOWN")
            complete = set(("request", "response")).issubset(s["complete"])
            outcomes[(status, auth, complete)] += 1
            events.append(dict(tcp_stream=connection, http2_stream=stream,
                request_template=template(path), method=req.get(":method", "UNKNOWN"),
                status=status, authResult=auth, complete=complete,
                first_frame=min(s["frames"]), last_frame=max(s["frames"]),
                classification="STANDARD_NEGATIVE_TEST" if auth == "AUTHENTICATION_FAILURE" else "STANDARD_TEST_RESULT"))
    (out / "native-authentication-results.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    return dict(source="official Open5GS v2.7.6 registration test capture; not malicious attack execution",
        pcap_sha256=digest(pcap_path), decoded_packets=len(packets), observed_http2_streams=len(streams),
        json_bodies=dict(valid_json), requests_by_service=dict(services),
        nf_profile_uri_body_binding=dict(profile_binding),
        http_negative_responses=[dict(service=k[0], status=k[1], observed_hops=v) for k, v in sorted(negative_responses.items())],
        auth_results=[dict(status=k[0], authResult=k[1], complete=k[2], observed_hops=v) for k, v in sorted(outcomes.items())],
        authorization_and_cu_verdict="UNKNOWN: no independent authenticated caller/authorization/CU evidence")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--h2-json", type=Path)
    p.add_argument("--pcap", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    result = {"recorded_attacks": csv_experiment(a.data, a.out)}
    if a.h2_json and a.pcap:
        result["native_standard_tests"] = native_experiment(a.h2_json, a.pcap, a.out)
    (a.out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({name: {m: v for m, v in cap["metrics"].items()} for name, cap in result["recorded_attacks"]["captures"].items()}, indent=2))
    if "native_standard_tests" in result:
        print(json.dumps(result["native_standard_tests"], indent=2))


if __name__ == "__main__":
    main()
