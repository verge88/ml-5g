"""Audit archived public captures without executing any upstream programs."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import struct
import subprocess
import urllib.request
from zipfile import ZipFile
from semantic5g.analyze import template

P=json.loads(Path(__file__).with_name("protocol.json").read_text())
MAGICS={b"\xd4\xc3\xb2\xa1":("<",1e6),b"\xa1\xb2\xc3\xd4":(">",1e6),
    b"\x4d\x3c\xb2\xa1":("<",1e9),b"\xa1\xb2\x3c\x4d":(">",1e9)}


def packets(stream):
    header=stream.read(24)
    if len(header)!=24 or header[:4] not in MAGICS:raise ValueError("unsupported PCAP header")
    endian,resolution=MAGICS[header[:4]]
    if struct.unpack(endian+"I",header[20:24])[0]!=1:raise ValueError("audit expects Ethernet PCAP")
    while True:
        record=stream.read(16)
        if not record:break
        if len(record)!=16:raise ValueError("truncated record")
        seconds,fraction,n,_=struct.unpack(endian+"IIII",record)
        if n>16*1024*1024:raise ValueError("oversized packet")
        data=stream.read(n)
        if len(data)!=n:raise ValueError("truncated packet")
        yield seconds+fraction/resolution,record,data,header


def decode(data):
    if len(data)<14:return None
    offset=14;kind=int.from_bytes(data[12:14],"big")
    for _ in range(2):
        if kind not in (0x8100,0x88a8):break
        if len(data)<offset+4:return None
        kind=int.from_bytes(data[offset+2:offset+4],"big");offset+=4
    if kind!=0x0800 or len(data)<offset+20:return None
    ip=bytearray(data[offset:]);ihl=(ip[0]&15)*4
    if ip[0]>>4!=4 or ihl<20 or len(ip)<ihl:return None
    length=int.from_bytes(ip[2:4],"big")
    if length<ihl or length>len(ip):return None
    ip=ip[:length];src=".".join(map(str,ip[12:16]));dst=".".join(map(str,ip[16:20]))
    proto=ip[9];payload=bytes(ip[ihl:]);fragment=int.from_bytes(ip[6:8],"big")&0x1fff
    sport=dport=None;marker=None;preface_port=None
    if proto in (6,17) and not fragment and len(payload)>=8:
        sport,dport=struct.unpack("!HH",payload[:4])
        if proto==17 and P["known_marker_port"] in (sport,dport):
            body=payload[8:]
            if len(body)>=P["marker_bytes"]:
                marker=dict(id=int.from_bytes(body[:4],"big"),start=bool(body[4]&128),
                    attack=bool(body[4]&64),type=body[5:25].decode("ascii",errors="replace").strip("\x00 "))
        if proto==6 and len(payload)>=20:
            tcp_length=(payload[12]>>4)*4
            if payload[tcp_length:].startswith(b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"):
                preface_port=dport
    checksum=sum(int.from_bytes(ip[i:i+2],"big") for i in range(0,ihl,2))
    while checksum>>16:checksum=(checksum&65535)+(checksum>>16)
    canonical=ip.copy();canonical[10:12]=b"\0\0";canonical[12:20]=b"\0"*8
    if not fragment and proto==6 and len(canonical)>=ihl+18:canonical[ihl+16:ihl+18]=b"\0\0"
    if not fragment and proto==17 and len(canonical)>=ihl+8:canonical[ihl+6:ihl+8]=b"\0\0"
    return dict(src=src,dst=dst,proto=proto,sport=sport,dport=dport,marker=marker,
        canonical=hashlib.sha256(canonical).digest(),checksum_valid=checksum==65535,
        source_mac_hash=hashlib.sha256(data[6:12]).hexdigest(),preface_port=preface_port)


def assign_label(parsed,states):
    marker=parsed["marker"]
    if marker is not None:
        key="attack" if marker["attack"] else "benign"
        states[key]=marker if marker["start"] else None
        return None
    attacker=P["marker_attacker_ipv4"] in (parsed["src"],parsed["dst"])
    if attacker and states.get("attack"):
        m=states["attack"];return 1,m["type"],str(m["id"])
    if not attacker and states.get("benign"):
        m=states["benign"];return 0,m["type"],str(m["id"])
    return -1,"UNKNOWN",""


def ordered(z,names):
    def first(name):
        with z.open(name) as stream:
            item=next(packets(stream),None)
        return item[0] if item else float("inf")
    return sorted(names,key=first)


def write_record(output,item,header_holder):
    _,record,data,header=item
    if not header_holder:
        output.write(header);header_holder.append(header)
    if header_holder[0]!=header:raise ValueError("different capture encodings")
    output.write(record);output.write(data)


def prepare_captures(z,scratch):
    names=z.namelist()
    benign=ordered(z,[n for n in names if re.fullmatch(r"CTD5G/benign/processed/benign_processed \(\d+\)\.pcap",n)])
    mixed=ordered(z,[n for n in names if re.fullmatch(r"CTD5G/both/original/both_original \(\d+\)\.pcap",n)])
    if len(benign)!=10 or len(mixed)!=4:raise ValueError("published archive layout changed")
    stats=Counter();ports=set();macs=defaultdict(set);quality=defaultdict(Counter)
    db=sqlite3.connect(scratch/"labels.sqlite")
    db.execute("CREATE TABLE labels(frame INTEGER PRIMARY KEY,label INTEGER,kind TEXT,episode TEXT,mac TEXT,ip_valid INTEGER)")
    with (scratch/"benign.pcap").open("wb") as out:
        header=[]
        for name in benign:
            with z.open(name) as stream:
                for item in packets(stream):
                    parsed=decode(item[2]);stats["benign_processed_packets"]+=1
                    if parsed:
                        stats["benign_marker_remnants"]+=parsed["marker"] is not None
                        if parsed["preface_port"]:ports.add(parsed["preface_port"])
                    write_record(out,item,header)
    frame=0;batch=[];episodes=set()
    with (scratch/"mixed.pcap").open("wb") as out:
        header=[]
        for name in mixed:
            processed=name.replace("/original/","/processed/").replace("both_original","both_processed")
            states={}
            with z.open(name) as original,z.open(processed) as cleaned:
                cleaned_iter=iter(packets(cleaned))
                for original_item in packets(original):
                    parsed=decode(original_item[2])
                    if parsed is None:continue
                    labelled=assign_label(parsed,states)
                    if labelled is None:
                        stats["original_marker_packets"]+=1;continue
                    item=next(cleaned_iter,None)
                    if item is None:raise ValueError("processed capture ended early")
                    clean=decode(item[2]);frame+=1
                    if clean is None or abs(item[0]-original_item[0])>.000002 or clean["canonical"]!=parsed["canonical"]:
                        raise ValueError("original/processed packet alignment failed")
                    label,kind,episode=labelled
                    if label>=0:
                        batch.append((frame,label,kind,name+":"+episode,clean["source_mac_hash"],int(clean["checksum_valid"])))
                        macs[label].add(clean["source_mac_hash"])
                        quality[str(label)]["packets"]+=1
                        quality[str(label)]["invalid_ipv4_checksum"]+=not clean["checksum_valid"]
                        if label==1:episodes.add((name,kind,episode))
                    else:stats["unknown_mixed_packets"]+=1
                    stats["mixed_processed_packets"]+=1
                    stats["mixed_marker_remnants"]+=clean["marker"] is not None
                    if clean["preface_port"]:ports.add(clean["preface_port"])
                    write_record(out,item,header)
                    if len(batch)>=10000:
                        db.executemany("INSERT INTO labels VALUES(?,?,?,?,?,?)",batch);batch=[]
                if next(cleaned_iter,None) is not None:raise ValueError("extra processed packets")
    if batch:db.executemany("INSERT INTO labels VALUES(?,?,?,?,?,?)",batch)
    db.commit()
    return db,dict(packet_counts=dict(stats),checksum_by_label={k:dict(v) for k,v in quality.items()},
        source_mac_counts={str(k):len(v) for k,v in macs.items()},
        source_mac_overlap=len(macs[0]&macs[1]),decoded_marker_episodes=len(episodes),
        segment_counts=dict(benign=10,mixed=4),collection_counts=dict(benign=1,mixed=1)),ports


def requests(path,ports,scratch):
    fields=["frame.number","frame.time_epoch","ip.src","ip.dst","http2.headers.method","http2.headers.path",
        "http.request.method","http.request.uri"]
    command=["tshark","-n","-r",str(path)]
    for port in sorted(ports):command += ["-d",f"tcp.port=={port},http2"]
    command += ["-Y","http2.headers.method || http.request.method","-T","fields","-E","separator=\t","-E","quote=d",
        "-E","occurrence=a","-E","aggregator=|"]
    for field in fields:command += ["-e",field]
    rows=[];ambiguous=0
    with (scratch/(path.stem+"-tshark.log")).open("w") as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log,text=True,encoding="utf-8",errors="replace")
        for row in csv.reader(process.stdout,delimiter="\t",quotechar='"'):
            if len(row)!=len(fields) or any("|" in row[i] for i in (2,3,4,5,6,7)):
                ambiguous+=1;continue
            frame,stamp,source,dest,h2_method,h2_resource,h1_method,h1_resource=row
            if h2_method and h1_method:ambiguous+=1;continue
            method,resource,wire=(h2_method,h2_resource,"HTTP2") if h2_method else (h1_method,h1_resource,"HTTP1")
            if method not in ("GET","POST","PUT","PATCH","DELETE","OPTIONS","HEAD") or not resource:
                ambiguous+=1;continue
            rows.append(dict(frame=int(frame),time=float(stamp),source=source,destination=dest,
                method=method,path=template(resource),wire=wire))
        if process.wait()!=0:raise ValueError("offline tshark decoding failed; inspect scratch log")
    return sorted(rows,key=lambda r:r["time"]),ambiguous


def coverage(benign,mixed,db):
    if not benign:return dict(status="NO_DECODED_BENIGN_SBI_REQUESTS")
    end=int(P["normal_train_fraction"]*len(benign))
    signature=lambda r:(r["source"],r["destination"],r["method"],r["path"])
    edges={(r["source"],r["destination"]) for r in benign[:end]}
    operations={signature(r) for r in benign[:end]}
    operation_wires={(*signature(r),r["wire"]) for r in benign[:end]}
    by_type=defaultdict(Counter);unknown=0;normal=Counter();request_macs=defaultdict(set)
    request_quality=defaultdict(Counter);wire_counts=defaultdict(Counter)
    for r in benign[end:]:
        normal["requests"]+=1;normal["new_edge"]+=(r["source"],r["destination"]) not in edges
        normal["new_operation"]+=signature(r) not in operations
    for r in mixed:
        label=db.execute("SELECT label,kind,episode,mac,ip_valid FROM labels WHERE frame=?",(r["frame"],)).fetchone()
        if label is None:unknown+=1;continue
        key="NORMAL" if label[0]==0 else label[1]
        c=by_type[key];c["requests"]+=1
        request_macs[label[0]].add(label[3]);request_quality[str(label[0])]["requests"]+=1
        request_quality[str(label[0])]["invalid_ipv4_checksum"]+=not label[4]
        wire_counts[str(label[0])][r["wire"]]+=1
        known_edge=(r["source"],r["destination"]) in edges
        known_operation=signature(r) in operations
        c["known_edge"]+=known_edge;c["known_operation"]+=known_operation
        c["known_operation_and_wire"]+=(*signature(r),r["wire"]) in operation_wires
    matched=sum(v["known_operation"] for k,v in by_type.items() if k!="NORMAL")
    matched_wire=sum(v["known_operation_and_wire"] for k,v in by_type.items() if k!="NORMAL")
    return dict(status="COVERAGE_OBSERVED",benign_train_requests=end,benign_holdout=dict(normal),
        labelled_mixed_requests={k:dict(v) for k,v in by_type.items()},unknown_mixed_requests=unknown,
        attack_requests_on_known_operations=matched,attack_requests_on_known_operations_and_wire=matched_wire,
        request_source_mac_counts={str(k):len(v) for k,v in request_macs.items()},
        request_source_mac_overlap=len(request_macs[0]&request_macs[1]),
        request_checksum_by_label={k:dict(v) for k,v in request_quality.items()},
        request_wire_protocol_by_label={k:dict(v) for k,v in wire_counts.items()},
        exploratory_coverage_gate=matched_wire>=P["min_known_operation_requests"] and
            normal["new_operation"]>=P["min_novel_normal_requests"],
        independent_confirmation_supported=False)


def download(path):
    if path.exists() and path.stat().st_size==P["dataset_bytes"]:
        md5=hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b""):md5.update(chunk)
        if md5.hexdigest()==P["dataset_md5"]:return
    request=urllib.request.Request(P["dataset_url"],headers={"User-Agent":"AcademicDatasetCoverageAudit/1.0"})
    md5=hashlib.md5();total=0
    with urllib.request.urlopen(request,timeout=60) as response,path.open("wb") as out:
        while True:
            chunk=response.read(8*1024*1024)
            if not chunk:break
            total+=len(chunk)
            if total>P["dataset_bytes"]:raise ValueError("unexpected archive size")
            md5.update(chunk);out.write(chunk)
    if total!=P["dataset_bytes"] or md5.hexdigest()!=P["dataset_md5"]:
        raise ValueError("published archive checksum mismatch")


def main(scratch,output):
    scratch.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(scratch).free<6*1024**3:raise ValueError("less than 6 GiB free scratch space")
    archive=scratch/"CTD5G.zip";download(archive)
    with ZipFile(archive) as z:
        db,packet_audit,ports=prepare_captures(z,scratch)
    benign,ambiguous_normal=requests(scratch/"benign.pcap",ports,scratch)
    mixed,ambiguous_mixed=requests(scratch/"mixed.pcap",ports,scratch)
    counts=coverage(benign,mixed,db);db.close()
    report=dict(protocol=P,archive_md5_verified=True,packet_audit=packet_audit,coverage=counts,
        decoded_benign_requests=len(benign),decoded_mixed_requests=len(mixed),
        ambiguous_request_packets_excluded=dict(benign=ambiguous_normal,mixed=ambiguous_mixed),
        tshark_version=subprocess.check_output(["tshark","--version"],text=True).splitlines()[0])
    with archive.open("rb") as stream:report["archive_sha256"]=hashlib.file_digest(stream,"sha256").hexdigest()
    (output/"audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    lines=["# Проверка пригодности CTD5G", "", "Опубликованный архив полностью скачан и проверен по MD5. "
        "Выполнен только офлайн-анализ; модели не обучались.", "", f'Штатных SBI-запросов: {len(benign)}. '
        f'Запросов в смешанной записи: {len(mixed)}.', "", "```json",json.dumps(counts,indent=2,ensure_ascii=False),"```", "",
        "Сегменты PCAP не считаются независимыми сеансами. В архиве одна смешанная коллекция; "
        "независимое подтверждение на отдельных коллекциях не обеспечено. Маркеры связывают "
        "пакеты с процедурой, но не доказывают её вредоносный результат. Непустая ячейка покрытия "
        "не доказывает превосходство ML. MAC-адреса, контрольные суммы, маркеры и идентификаторы "
        "процедур исключены из допустимых будущих признаков."]
    (output/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(dict(status=counts["status"],benign=len(benign),mixed=len(mixed),coverage=counts)),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--scratch",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();main(a.scratch,a.output)
