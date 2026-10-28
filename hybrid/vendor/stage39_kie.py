from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import hmac
import json
import math
import os
import secrets
import statistics
import tempfile
import time
from pathlib import Path

NF_TYPES = ("NSSF","AUSF","BSF","UDM","AMF","SMF","PCF","NRF","SCP")
KIE_VERSION = 1
MAX_CLOCK_SKEW_MS = 30_000


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",",":"), ensure_ascii=False).encode()


def sha256_json(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def hmac_hex(key, value):
    return hmac.new(key, canonical_bytes(value), hashlib.sha256).hexdigest()


class SecurityError(ValueError):
    pass


def normalize_service(service):
    endpoints = [{
        "ip": str(e.get("ip","")),
        "port": int(e.get("port",0)),
        "transport": str(e.get("transport","TCP")).upper(),
    } for e in service.get("endpoints",[])]
    endpoints.sort(key=lambda x:(x["ip"],x["port"],x["transport"]))
    return {
        "serviceName": str(service.get("serviceName","")),
        "scheme": str(service.get("scheme","http")).lower(),
        "status": str(service.get("status","REGISTERED")),
        "versions": sorted(str(v) for v in service.get("versions",[])),
        "endpoints": endpoints,
    }


def normalize_profile(profile):
    services = [normalize_service(s) for s in profile.get("nfServices",[])]
    services.sort(key=canonical_bytes)
    return {
        "slot": str(profile.get("slot","")),
        "nfInstanceId": str(profile.get("nfInstanceId","")).lower(),
        "nfType": str(profile.get("nfType","")),
        "nfStatus": str(profile.get("nfStatus","REGISTERED")),
        "fqdn": str(profile.get("fqdn","")),
        "ipv4Addresses": sorted(str(x) for x in profile.get("ipv4Addresses",[])),
        "nfServices": services,
    }


class FullProfilePolicy:
    def __init__(self, doc):
        self.document = copy.deepcopy(doc)
        self.slots = doc.get("slots")
        if int(doc.get("version",0)) != 1 or not isinstance(self.slots,dict) or not self.slots:
            raise ValueError("invalid policy")
        for slot,item in self.slots.items():
            if item.get("nf_type") not in NF_TYPES or not item.get("endpoint") or not item.get("fqdn"):
                raise ValueError("invalid slot policy: "+slot)
        self.digest = sha256_json(doc)

    @classmethod
    def load(cls,path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def expected_profile(self,slot,nf_instance_id):
        p=self.slots[slot]
        return normalize_profile({
            "slot":slot,
            "nfInstanceId":nf_instance_id,
            "nfType":p["nf_type"],
            "nfStatus":"REGISTERED",
            "fqdn":p["fqdn"],
            "ipv4Addresses":[p["endpoint"]],
            "nfServices":p.get("services",[]),
        })

    def matches(self,profile):
        slot=str(profile.get("slot",""))
        if slot not in self.slots:
            return False
        n=normalize_profile(profile)
        return n == self.expected_profile(slot,n["nfInstanceId"])


class PersistentRegistry:
    def __init__(self,path):
        self.path=Path(path)

    def write(self,profiles):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.path.write_text(json.dumps(profiles,sort_keys=True),encoding="utf-8")

    def read(self):
        return json.loads(self.path.read_text(encoding="utf-8"))


class FreshnessAnchor:
    def __init__(self,path,mirror_path=None,seal_key=None,must_recover=False):
        self.path=Path(path)
        self.mirror_path=Path(mirror_path) if mirror_path else None
        self.seal_key=seal_key
        self.state={"generation":1,"subjects":{}}
        if self.path.exists():
            self.state=json.loads(self.path.read_text(encoding="utf-8"))
            self._validate()
        elif self.mirror_path and self.mirror_path.exists():
            if not seal_key:
                raise SecurityError("mirror requires seal key")
            env=json.loads(self.mirror_path.read_text(encoding="utf-8"))
            state,mac=env.get("state"),env.get("mac")
            if not isinstance(state,dict) or not hmac.compare_digest(str(mac),hmac_hex(seal_key,state)):
                raise SecurityError("bad recovery mirror")
            self.state=state
            self._validate()
            self._save_primary()
        elif must_recover:
            raise SecurityError("freshness state unavailable; refusing unsafe reset")

    def _validate(self):
        if not isinstance(self.state,dict) or not isinstance(self.state.get("subjects"),dict):
            raise SecurityError("corrupt freshness anchor")

    def _save_primary(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        tmp=self.path.with_suffix(self.path.suffix+".tmp")
        tmp.write_text(json.dumps(self.state,sort_keys=True),encoding="utf-8")
        os.replace(tmp,self.path)

    def _save(self):
        self._save_primary()
        if self.mirror_path:
            if not self.seal_key:
                raise SecurityError("mirror requires seal key")
            self.mirror_path.parent.mkdir(parents=True,exist_ok=True)
            env={"state":self.state,"mac":hmac_hex(self.seal_key,self.state)}
            tmp=self.mirror_path.with_suffix(self.mirror_path.suffix+".tmp")
            tmp.write_text(json.dumps(env,sort_keys=True),encoding="utf-8")
            os.replace(tmp,self.mirror_path)

    def check_and_commit(self,controller,subject,cycle,nonce,issued_at_ms,now_ms):
        if abs(now_ms-issued_at_ms) > MAX_CLOCK_SKEW_MS:
            raise SecurityError("stale KIE")
        key=controller+"|"+subject
        row=self.state["subjects"].setdefault(key,{"last_cycle":-1,"nonce_hashes":[]})
        if int(cycle) <= int(row["last_cycle"]):
            raise SecurityError("replayed cycle")
        nh=hashlib.sha256(nonce.encode()).hexdigest()
        if nh in row["nonce_hashes"]:
            raise SecurityError("replayed nonce")
        row["last_cycle"]=int(cycle)
        row["nonce_hashes"]=(row["nonce_hashes"][-255:]+[nh])
        self._save()


@dataclasses.dataclass(frozen=True)
class KIERequest:
    controller_id:str
    subject_id:str
    cycle:int
    issued_at_ms:int
    nonce:str
    policy_sha256:str
    telemetry_digest:str
    mac:str

    @classmethod
    def create(cls,controller_id,subject_id,cycle,policy_sha256,telemetry_digest,key,now_ms=None,nonce=None):
        core={
            "kind":"KIE_REQUEST","version":KIE_VERSION,
            "controller_id":controller_id,"subject_id":subject_id,"cycle":int(cycle),
            "issued_at_ms":int(now_ms if now_ms is not None else time.time()*1000),
            "nonce":nonce or secrets.token_hex(16),
            "policy_sha256":policy_sha256,"telemetry_digest":telemetry_digest,
        }
        return cls(controller_id,subject_id,int(cycle),core["issued_at_ms"],core["nonce"],policy_sha256,telemetry_digest,hmac_hex(key,core))

    def core(self):
        return {
            "kind":"KIE_REQUEST","version":KIE_VERSION,
            "controller_id":self.controller_id,"subject_id":self.subject_id,"cycle":self.cycle,
            "issued_at_ms":self.issued_at_ms,"nonce":self.nonce,
            "policy_sha256":self.policy_sha256,"telemetry_digest":self.telemetry_digest,
        }

    def verify(self,key,anchor,expected_policy,expected_telemetry_digest,now_ms=None):
        if not hmac.compare_digest(self.mac,hmac_hex(key,self.core())):
            raise SecurityError("invalid KIE MAC")
        if self.policy_sha256 != expected_policy or self.telemetry_digest != expected_telemetry_digest:
            raise SecurityError("KIE binding mismatch")
        anchor.check_and_commit(
            self.controller_id,self.subject_id,self.cycle,self.nonce,self.issued_at_ms,
            int(now_ms if now_ms is not None else time.time()*1000)
        )


@dataclasses.dataclass(frozen=True)
class KIEResponse:
    controller_id:str
    subject_id:str
    cycle:int
    request_nonce:str
    issued_at_ms:int
    report:dict
    mac:str

    @classmethod
    def create(cls,request,report,key,now_ms=None):
        core={
            "kind":"KIE_RESPONSE","version":KIE_VERSION,
            "controller_id":request.controller_id,"subject_id":request.subject_id,
            "cycle":request.cycle,"request_nonce":request.nonce,
            "issued_at_ms":int(now_ms if now_ms is not None else time.time()*1000),
            "report":copy.deepcopy(report),
        }
        return cls(core["controller_id"],core["subject_id"],core["cycle"],core["request_nonce"],core["issued_at_ms"],core["report"],hmac_hex(key,core))

    def core(self):
        return {
            "kind":"KIE_RESPONSE","version":KIE_VERSION,
            "controller_id":self.controller_id,"subject_id":self.subject_id,
            "cycle":self.cycle,"request_nonce":self.request_nonce,
            "issued_at_ms":self.issued_at_ms,"report":self.report,
        }

    def verify(self,key,request,now_ms=None):
        if not hmac.compare_digest(self.mac,hmac_hex(key,self.core())):
            raise SecurityError("invalid KIE response MAC")
        if (self.controller_id,self.subject_id,self.cycle,self.request_nonce)!=(request.controller_id,request.subject_id,request.cycle,request.nonce):
            raise SecurityError("KIE response mismatch")
        now=int(now_ms if now_ms is not None else time.time()*1000)
        if abs(now-self.issued_at_ms)>MAX_CLOCK_SKEW_MS:
            raise SecurityError("stale KIE response")


class StreamRelationalVerifier:
    def __init__(self,policy):
        self.policy=policy
        self.active_slots=set(policy.slots)
        self.transition=None

    def begin_transition(self,operation,slot):
        if operation not in {"stop","start","restart"} or slot not in self.policy.slots or self.transition:
            raise ValueError("invalid transition")
        if operation=="stop" and slot not in self.active_slots:
            raise ValueError("stop absent")
        if operation=="start" and slot in self.active_slots:
            raise ValueError("start active")
        if operation=="restart" and slot not in self.active_slots:
            raise ValueError("restart absent")
        self.transition={"operation":operation,"slot":slot}

    def end_transition(self):
        if not self.transition:
            raise ValueError("no transition")
        op,slot=self.transition["operation"],self.transition["slot"]
        if op=="stop":
            self.active_slots.remove(slot)
        elif op=="start":
            self.active_slots.add(slot)
        self.transition=None

    def verify(self,profiles,witness_rows):
        alerts=[]
        rows=[normalize_profile(p) for p in profiles]
        ids=[p["nfInstanceId"] for p in rows]
        if len(ids)!=len(set(ids)):
            alerts.append("duplicate_nfInstanceId")
        by_slot={}
        for p in rows:
            by_slot.setdefault(p["slot"],[]).append(p)
            if p["slot"] not in self.policy.slots:
                alerts.append("unknown_slot:"+p["slot"])

        moving=self.transition["slot"] if self.transition else None
        expected=set(self.active_slots)
        if self.transition and self.transition["operation"]=="start":
            expected.add(moving)

        for slot in sorted(self.policy.slots):
            candidates=by_slot.get(slot,[])
            if len(candidates)>1:
                alerts.append("ambiguous_slot:"+slot)
                continue
            present=len(candidates)==1
            if slot==moving and self.transition:
                if not present:
                    continue
            elif slot in expected and not present:
                alerts.append("missing_stable_slot:"+slot)
                continue
            elif slot not in expected and present:
                alerts.append("unexpected_inactive_slot:"+slot)

            if present:
                profile=candidates[0]
                if not self.policy.matches(profile):
                    alerts.append("semantic_mismatch:"+slot)
                w=witness_rows.get(slot)
                if not isinstance(w,dict):
                    alerts.append("missing_witness:"+slot)
                else:
                    p=self.policy.slots[slot]
                    if w.get("nf_type")!=p["nf_type"]: alerts.append("witness_type:"+slot)
                    if w.get("endpoint")!=p["endpoint"]: alerts.append("witness_endpoint:"+slot)
                    if str(w.get("nf_instance_id","")).lower()!=profile["nfInstanceId"]: alerts.append("witness_uuid:"+slot)
                    if w.get("process_alive") is not True: alerts.append("witness_process:"+slot)
                    if w.get("direct_sbi_tcp_reachable") is not True: alerts.append("witness_reachability:"+slot)
        return sorted(set(alerts))


def default_profiles(policy):
    profiles,witnesses=[],{}
    for i,slot in enumerate(sorted(policy.slots),1):
        uid=f"00000000-0000-0000-0000-{i:012d}"
        profiles.append(policy.expected_profile(slot,uid))
        p=policy.slots[slot]
        witnesses[slot]={
            "slot":slot,"nf_type":p["nf_type"],"endpoint":p["endpoint"],"nf_instance_id":uid,
            "process_alive":True,"direct_sbi_tcp_reachable":True,
        }
    return profiles,witnesses


def mutate(profiles,slot,kind):
    out=copy.deepcopy(profiles)
    p=next(x for x in out if x["slot"]==slot)
    if kind=="identity":
        p["nfInstanceId"]="ffffffff-ffff-ffff-ffff-ffffffffffff"
    elif kind=="endpoint":
        p["ipv4Addresses"]=["203.0.113.99"]
    elif kind=="service":
        p.setdefault("nfServices",[]).append({"serviceName":"unauthorized","scheme":"http","status":"REGISTERED","versions":["v1"],"endpoints":[{"ip":"203.0.113.99","port":80,"transport":"TCP"}]})
    elif kind=="fqdn":
        p["fqdn"]="poisoned.invalid"
    elif kind=="type":
        p["nfType"]="SMF" if p["nfType"]!="SMF" else "AMF"
    elif kind=="compound":
        p["nfInstanceId"]="eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
        p.setdefault("nfServices",[]).append({"serviceName":"unauthorized-compound","scheme":"http","status":"REGISTERED","versions":["v1"],"endpoints":[{"ip":"203.0.113.98","port":8080,"transport":"TCP"}]})
    elif kind=="duplicate":
        out.append(copy.deepcopy(p))
    else:
        raise ValueError(kind)
    return out


def nearest_rank(values, probability):
    if not values:
        raise ValueError("empty sample")
    ordered=sorted(values)
    rank=max(1,math.ceil(probability*len(ordered)))
    return ordered[rank-1]


def lifecycle_case(policy,profiles,witnesses,slot):
    results={}

    verifier=StreamRelationalVerifier(policy)
    verifier.begin_transition("stop",slot)
    absent=[p for p in profiles if p["slot"]!=slot]
    absent_witness={k:v for k,v in witnesses.items() if k!=slot}
    if verifier.verify(absent,absent_witness):
        raise RuntimeError("stop transition rejected for "+slot)
    verifier.end_transition()
    if verifier.verify(absent,absent_witness):
        raise RuntimeError("post-stop state rejected for "+slot)
    results["stop"]=True

    verifier.begin_transition("start",slot)
    if verifier.verify(absent,absent_witness):
        raise RuntimeError("start transition absence rejected for "+slot)
    if verifier.verify(profiles,witnesses):
        raise RuntimeError("start transition profile rejected for "+slot)
    verifier.end_transition()
    if verifier.verify(profiles,witnesses):
        raise RuntimeError("post-start state rejected for "+slot)
    results["start"]=True

    verifier=StreamRelationalVerifier(policy)
    verifier.begin_transition("restart",slot)
    if verifier.verify(absent,absent_witness):
        raise RuntimeError("restart transition absence rejected for "+slot)
    changed=copy.deepcopy(profiles)
    row=next(p for p in changed if p["slot"]==slot)
    new_uuid=hashlib.sha256(("restart:"+slot).encode()).hexdigest()[:32]
    row["nfInstanceId"]=f"{new_uuid[:8]}-{new_uuid[8:12]}-{new_uuid[12:16]}-{new_uuid[16:20]}-{new_uuid[20:32]}"
    changed_witness=copy.deepcopy(witnesses)
    changed_witness[slot]["nf_instance_id"]=row["nfInstanceId"]
    if verifier.verify(changed,changed_witness):
        raise RuntimeError("restart new UUID rejected for "+slot)
    verifier.end_transition()
    if verifier.verify(changed,changed_witness):
        raise RuntimeError("post-restart state rejected for "+slot)
    results["restart"]=True
    return results


def run_ci(policy,output,soak_seconds):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    profiles,witnesses=default_profiles(policy)
    if StreamRelationalVerifier(policy).verify(profiles,witnesses):
        raise RuntimeError("clean baseline rejected")

    article_kinds=("identity","endpoint","service","fqdn","type","compound")
    boundary_kinds=("duplicate",)
    kinds=article_kinds+boundary_kinds
    attack_matrix={}
    latencies=[]
    per_nf_type={}
    for slot in sorted(policy.slots):
        typ=policy.slots[slot]["nf_type"]
        per_nf_type.setdefault(typ,{"tested":0,"detected":0})
        slot_results={}
        for kind in kinds:
            attacked=mutate(profiles,slot,kind)
            verifier=StreamRelationalVerifier(policy)
            t0=time.perf_counter_ns()
            alerts=verifier.verify(attacked,witnesses)
            ms=(time.perf_counter_ns()-t0)/1_000_000
            if not alerts:
                raise RuntimeError(f"missed {kind} on {slot}")
            slot_results[kind]={
                "detected":True,
                "alerts":alerts,
                "detector_execution_latency_ms":ms,
            }
            latencies.append(ms)
            per_nf_type[typ]["tested"]+=1
            per_nf_type[typ]["detected"]+=1
        attack_matrix[slot]=slot_results

    poison_results={}
    with tempfile.TemporaryDirectory() as td:
        for slot in sorted(policy.slots):
            for kind in ("fqdn","service"):
                path=Path(td)/f"{slot}-{kind}.json"
                registry=PersistentRegistry(path)
                registry.write(mutate(profiles,slot,kind))
                reopened=PersistentRegistry(path).read()
                alerts=StreamRelationalVerifier(policy).verify(reopened,witnesses)
                expected="semantic_mismatch:"+slot
                if expected not in alerts:
                    raise RuntimeError(f"persistent {kind} poisoning survived policy for {slot}")
                poison_results[f"{slot}:{kind}"]={"detected":True,"alerts":alerts}

    lifecycle_results={}
    for slot in sorted(policy.slots):
        lifecycle_results[slot]=lifecycle_case(policy,profiles,witnesses,slot)

    kie_slot="nssf1" if "nssf1" in policy.slots else sorted(policy.slots)[0]
    with tempfile.TemporaryDirectory() as td:
        primary=Path(td)/"anchor.json"
        mirror=Path(td)/"anchor_mirror.json"
        seal=hashlib.sha256(b"stage39-anchor-seal").digest()
        ckey=hashlib.sha256(b"stage39-controller").digest()
        skey=hashlib.sha256(b"stage39-subject").digest()
        now=int(time.time()*1000)
        digest=sha256_json(profiles)
        req=KIERequest.create("verifier-a",kie_slot,1,policy.digest,digest,ckey,now,"0123456789abcdef0123456789abcdef")
        anchor=FreshnessAnchor(primary,mirror,seal)
        req.verify(ckey,anchor,policy.digest,digest,now)
        primary.unlink()
        recovered=FreshnessAnchor(primary,mirror,seal)
        response=KIEResponse.create(
            req,
            {
                "profile_digest":digest,
                "witness_digest":sha256_json(witnesses[kie_slot]),
                "process_alive":True,
                "sbi_reachable":True,
            },
            skey,
            now,
        )
        response.verify(skey,req,now)
        try:
            recovered.check_and_commit(req.controller_id,req.subject_id,req.cycle,req.nonce,req.issued_at_ms,now)
            raise RuntimeError("replay accepted")
        except SecurityError:
            pass
        tampered=dataclasses.replace(req,telemetry_digest="0"*64)
        try:
            tampered.verify(
                ckey,
                FreshnessAnchor(Path(td)/"tamper.json"),
                policy.digest,
                "0"*64,
                now,
            )
            raise RuntimeError("tamper accepted")
        except SecurityError:
            pass

        primary.unlink(missing_ok=True)
        mirror.unlink(missing_ok=True)
        try:
            FreshnessAnchor(primary,mirror,seal,must_recover=True)
            raise RuntimeError("total freshness-state loss allowed unsafe reset")
        except SecurityError:
            pass

    soak_verifier=StreamRelationalVerifier(policy)
    started=time.perf_counter()
    iterations=false_alerts=0
    while time.perf_counter()-started < max(0.0,float(soak_seconds)):
        iterations+=1
        false_alerts+=int(bool(soak_verifier.verify(profiles,witnesses)))
        time.sleep(0.005)
    if false_alerts:
        raise RuntimeError("clean soak false alert")

    p95=nearest_rank(latencies,0.95)
    total_article_attacks=len(policy.slots)*len(article_kinds)
    total_boundary_attacks=len(policy.slots)*len(boundary_kinds)
    total_attacks=total_article_attacks+total_boundary_attacks
    total_poison=len(poison_results)
    total_lifecycle=len(policy.slots)*3
    report={
        "stage":39,
        "analysis_kind":"GitHub Actions CI integration harness; not a new Open5GS production trial",
        "policy_sha256":policy.digest,
        "nf_types":sorted({x["nf_type"] for x in policy.slots.values()}),
        "slots":len(policy.slots),
        "kie":{
            "version":KIE_VERSION,
            "auth":"HMAC-SHA256",
            "freshness":["timestamp_window","monotonic_cycle","nonce_cache","persistent_anchor","sealed_recovery_mirror"],
            "telemetry_binding":True,
            "policy_binding":True,
            "replay_after_restart_rejected":True,
            "primary_anchor_loss_recovered":True,
            "total_anchor_loss_fail_closed":True,
            "tamper_rejected":True,
        },
        "streaming":{
            "uses_future_post_reference":False,
            "lifecycle_operations":["stop","start","restart"],
            "lifecycle_cases":total_lifecycle,
            "lifecycle_passed":total_lifecycle,
            "by_slot":lifecycle_results,
        },
        "independent_full_profile_policy":{
            "fqdn":True,
            "services":True,
            "endpoint":True,
            "nf_type":True,
            "stage38_two_instance_types_restored":["NSSF","AUSF","BSF","UDM"],
        },
        "persistent_reference_poisoning":{
            "file_backed_registry_survived_reopen":True,
            "cases":total_poison,
            "detected":total_poison,
            "by_slot_and_field":poison_results,
        },
        "attacks":{
            "article_families":list(article_kinds),
            "article_cases":total_article_attacks,
            "article_detected":total_article_attacks,
            "boundary_families":list(boundary_kinds),
            "boundary_cases":total_boundary_attacks,
            "boundary_detected":total_boundary_attacks,
            "total_cases":total_attacks,
            "total_detected":total_attacks,
            "per_nf_type":per_nf_type,
            "matrix":attack_matrix,
        },
        "detector_execution_latency_ms":{
            "measurement_scope":"time from in-process verifier call entry to return; not end-to-end attack-to-alert latency",
            "sample_count":len(latencies),
            "median":statistics.median(latencies),
            "p95_nearest_rank":p95,
            "max":max(latencies),
        },
        "soak":{"requested_seconds":float(soak_seconds),"iterations":iterations,"false_alerts":false_alerts},
        "limitations":[
            "CI harness, not a new live Open5GS measurement.",
            "Detector timing excludes transport, NRF/SCP propagation, collection and scheduling delays.",
            "Test keys are ephemeral; production key custody and hardware attestation are not established.",
            "Repository policy is independent of NRF observations, but repository compromise remains a common risk.",
            "Recovery mirror covers single-anchor loss, not simultaneous destruction of all copies.",
        ],
    }
    (output/"stage39_report.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n",
        encoding="utf-8",
    )
    (output/"stage39_report.md").write_text(
        "# Stage39 KIE / streaming CI report\n\n"
        f"- NF types: {', '.join(report['nf_types'])}\n"
        f"- Slots: {report['slots']}\n"
        f"- Future post-reference used: {report['streaming']['uses_future_post_reference']}\n"
        f"- Lifecycle stop/start/restart: {total_lifecycle}/{total_lifecycle} cases passed\n"
        f"- KIE replay after restart rejected: {report['kie']['replay_after_restart_rejected']}\n"
        f"- Persistent FQDN/service poisoning: {total_poison}/{total_poison} detected\n"
        f"- Article mutation matrix: {total_article_attacks}/{total_article_attacks} detected across all configured slots\n"
        f"- Duplicate boundary matrix: {total_boundary_attacks}/{total_boundary_attacks} detected\n"
        f"- Detector execution median / p95 / max: "
        f"{report['detector_execution_latency_ms']['median']:.3f} / "
        f"{report['detector_execution_latency_ms']['p95_nearest_rank']:.3f} / "
        f"{report['detector_execution_latency_ms']['max']:.3f} ms "
        f"(n={len(latencies)}, in-process only)\n"
        f"- Clean soak: {iterations} iterations, {false_alerts} false alerts\n",
        encoding="utf-8",
    )
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--policy",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--soak-seconds",type=float,default=2.0)
    args=ap.parse_args()
    print(json.dumps(run_ci(FullProfilePolicy.load(args.policy),args.output,args.soak_seconds),indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
