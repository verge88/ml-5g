"""Real native CU/witness sampling during matched local availability tests."""
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import secrets
import signal
import time
from types import SimpleNamespace
from hybrid.vendor.stage39_kie import KIERequest,KIEResponse,SecurityError,sha256_json
from cu5g import driver as core
from cu5g.common import POLICY,atomic,assess,native_profile,validate_witness
from live5g import capture
from .common import P,provenance


class Sampler:
    def __init__(self,verifier,anchor):
        self.verifier,self.anchor=verifier,anchor
        self.latest={};self.keys={s:Path(f"/nfkeys/{s}.key").read_bytes() for s in POLICY.slots}
        self.cycle=10000;self.last_issue=-1e30;self.challenge=None;self.accepted=set();self.evidence=[]

    def observe(self,observations):
        now=time.monotonic();profiles={}
        for slot in POLICY.slots:
            body=observations[slot]["nrf"]["body"]
            rows=body.get("nfInstances",[]) if isinstance(body,dict) else []
            if len(rows)==1:profiles[slot]=rows[0]
            observations[slot]["deployment_policy_alert"]=bool(rows) and (len(rows)!=1 or not POLICY.matches(native_profile(slot,rows[0])))
        if now-self.last_issue>=P["native_cu_request_period_seconds"]:
            self.cycle+=1
            for slot,profile in profiles.items():
                normalized=native_profile(slot,profile)
                if POLICY.matches(normalized):
                    req=KIERequest.create("native-controller",slot,self.cycle,POLICY.digest,sha256_json(normalized),self.keys[slot])
                    self.latest[slot]=req
                    atomic(f"/nf/{slot}/request.json",asdict(req))
            self.last_issue=now
        try:
            bundle=json.loads(Path("/witness/response.json").read_text())
            validate_witness(bundle,"/public/witness.pub",self.challenge)
        except Exception:
            bundle=None
        states={};auth_alarm=False;normalized=[]
        for slot in POLICY.slots:
            try:
                profile=profiles[slot];p=native_profile(slot,profile);normalized.append(p)
                req=self.latest[slot]
                response=KIEResponse(**profile["customInfo"]["labKIE"]["response"])
                if (response.cycle,response.request_nonce)!=(req.cycle,req.nonce):
                    states[slot]="PENDING_CURRENT_CU";continue
                response.verify(self.keys[slot],req)
                if response.report["telemetry_digest"]!=sha256_json(p) or response.report["runtime"]["nf_instance_id"]!=p["nfInstanceId"]:
                    states[slot]="UNKNOWN_BINDING_AFTER_CHANGE";continue
                token=(slot,req.cycle,req.nonce)
                if token not in self.accepted:
                    self.anchor.check_and_commit(req.controller_id,req.subject_id,req.cycle,req.nonce,req.issued_at_ms,int(time.time()*1000))
                    self.accepted.add(token)
                    self.evidence.append({"slot":slot,"request":asdict(req),"response":asdict(response),"native_profile":profile})
                states[slot]="AUTHENTICATED_CURRENT_CU"
            except SecurityError as exc:
                if "stale" in str(exc):states[slot]="UNKNOWN_STALE_CU"
                else:states[slot]="AUTHENTICATION_ALARM";auth_alarm=True
            except (KeyError,TypeError,ValueError):states[slot]="UNKNOWN_CU_OR_PROFILE_MISSING"
        decision={"verdict":"UNKNOWN","reason":"pending_cu_or_witness","alerts":[]}
        if len(normalized)==len(POLICY.slots) and all(s=="AUTHENTICATED_CURRENT_CU" for s in states.values()):
            decision=assess(normalized,bundle,"/public/witness.pub",self.challenge,self.verifier)
        hard=auth_alarm or bool(decision["alerts"]) or bool(decision.get("authentication_alarm",False))
        monitor={"states":states,"decision":decision,"hard_alarm":hard,
                 "witness_available":bundle is not None,"current_request_cycle":self.cycle,
                 "confirmed":decision["verdict"]=="CONFIRMED_MATCH_IN_DECLARED_SCOPE"}
        for slot in POLICY.slots:observations[slot]["integrity_monitor"]=monitor
        self.challenge={"cycle":self.cycle,"nonce":secrets.token_hex(16),"policy_sha256":POLICY.digest}
        atomic("/control/witness-request.json",self.challenge)
        return observations


def local_command(*args):
    slot=args[-1]
    if slot not in POLICY.slots:raise ValueError("not a local test slot")
    if args[0]=="stop":core.stop(slot)
    elif args[0]=="start":core.start(slot)
    elif args[0]=="pause":os.kill(core.PROCS[slot].pid,signal.SIGSTOP)
    elif args[0]=="unpause":
        try:os.kill(core.PROCS[slot].pid,signal.SIGCONT)
        except ProcessLookupError:pass
    else:raise ValueError("unsupported local operation")
    return "local native process operation completed"


async def run():
    original_cycle=core.cycle
    original_observe=capture.observe
    async def hooked_cycle(number,label,verifier,anchor):
        if number!=5:return await original_cycle(number,label,verifier,anchor)
        sampler=Sampler(verifier,anchor)
        async def observed():return sampler.observe(await original_observe())
        capture.observe=observed;capture.command=local_command;capture.COMPOSE=["local-process"]
        capture.subprocess=SimpleNamespace(run=lambda args,**kwargs:local_command("unpause",args[-1]))
        output=Path("/out/native")
        await capture.run(SimpleNamespace(output=output,replica=int(os.environ["REPLICA"])))
        manifest=json.loads((output/"capture-manifest.json").read_text())
        manifest.update(provenance("native_capture"),commit=os.environ["SOURCE_COMMIT"],full_cu_witness_status="per-observation integrity_monitor",native_cu_request_period_seconds=P["native_cu_request_period_seconds"])
        atomic(output/"capture-manifest.json",manifest)
        atomic("/out/sampler-accepted-cus.json",sampler.evidence)
        # Shared response anchor and monotonically higher final cycle preserve
        # anti-replay after the continuous sampler has consumed newer CUs.
        return await original_cycle(sampler.cycle+1,label,verifier,anchor)
    core.cycle=hooked_cycle
    await core.run()


if __name__=="__main__":asyncio.run(run())
