from io import BytesIO
import sqlite3
import struct
from zipfile import ZipFile
import pytest
from coverage5g.audit import P,decode,assign_label,packets,prepare_captures,coverage


def checksum(data):
    value=sum(int.from_bytes(data[i:i+2],"big") for i in range(0,len(data),2))
    while value>>16:value=(value&65535)+(value>>16)
    return (~value)&65535


def udp(source="10.100.200.1",dest="10.100.200.3",payload=b"sample",port=8000,broken_checksum=False):
    octets=lambda x:bytes(map(int,x.split(".")))
    body=struct.pack("!HHHH",40000,port,8+len(payload),0)+payload
    header=bytearray(struct.pack("!BBHHHBBH4s4s",0x45,0,20+len(body),1,0,64,17,0,octets(source),octets(dest)))
    header[10:12]=struct.pack("!H",checksum(header))
    if broken_checksum:header[10]^=1
    return b"\x00\x01\x02\x03\x04\x05\x10\x11\x12\x13\x14\x15\x08\x00"+header+body


def marker(attack=True,start=True):
    payload=struct.pack("!I",7)+bytes([(128 if start else 0)|(64 if attack else 0)])+b"example".ljust(20,b"\0")
    return udp(payload=payload,port=9999)


def pcap(items):
    out=BytesIO();out.write(struct.pack("<IHHIIII",0xa1b2c3d4,2,4,0,0,65535,1))
    for stamp,data in items:out.write(struct.pack("<IIII",stamp,0,len(data),len(data))+data)
    return out.getvalue()


def test_address_and_checksum_rewrite_do_not_change_packet_identity():
    a=decode(udp(source=P["marker_attacker_ipv4"]))
    b=decode(udp(source="10.100.200.12",broken_checksum=True))
    assert a["canonical"]==b["canonical"]
    assert a["checksum_valid"] and not b["checksum_valid"]


def test_concurrent_markers_never_turn_unscoped_attacker_into_benign():
    states={}
    assert assign_label(decode(marker()),states) is None
    assert assign_label(decode(udp(source=P["marker_attacker_ipv4"])),states)[0]==1
    assert assign_label(decode(udp()),states)[0]==-1
    assign_label(decode(marker(attack=False)),states)
    assert assign_label(decode(udp()),states)[0]==0
    assign_label(decode(marker(start=False)),states)
    assert assign_label(decode(udp(source=P["marker_attacker_ipv4"])),states)[0]==-1


def test_truncated_pcap_is_rejected():
    data=pcap([(1,udp())])
    with pytest.raises(ValueError,match="truncated"):
        list(packets(BytesIO(data[:-2])))


@pytest.mark.parametrize("permuted",[False,True])
@pytest.mark.parametrize("omitted_source",[False,True])
def test_archive_alignment_and_label_map(tmp_path,permuted,omitted_source):
    path=tmp_path/"fixture.zip"
    with ZipFile(path,"w") as z:
        for i in range(1,11):
            z.writestr(f"CTD5G/benign/processed/benign_processed ({i}).pcap",pcap([(i,udp())]))
        for i in range(1,5):
            original=[(i*10,marker()),(i*10+1,udp(source=P["marker_attacker_ipv4"])),(i*10+2,marker(start=False)),
                (i*10+3,marker(attack=False)),(i*10+4,udp()),(i*10+5,marker(attack=False,start=False))]
            if omitted_source:original.insert(1,(i*10,udp(source=P["marker_attacker_ipv4"],payload=b"omitted")))
            processed=[(i*10+1,udp(source="10.100.200.12")),(i*10+4,udp())]
            z.writestr(f"CTD5G/both/original/both_original ({i}).pcap",pcap(original))
            number=(i%4)+1 if permuted else i
            z.writestr(f"CTD5G/both/processed/both_processed ({number}).pcap",pcap(processed))
    scratch=tmp_path/"scratch";scratch.mkdir()
    with ZipFile(path) as z:db,audit,ports=prepare_captures(z,scratch)
    assert db.execute("SELECT count(*) FROM labels WHERE label=1").fetchone()[0]==4
    assert db.execute("SELECT count(*) FROM labels WHERE label=0").fetchone()[0]==4
    assert audit["collection_counts"]["mixed"]==1 and audit["segment_counts"]["mixed"]==4
    assert audit["packet_counts"]["mixed_marker_remnants"]==0
    assert audit["packet_counts"].get("source_packets_omitted_from_processed",0)==(4 if omitted_source else 0)
    db.close()


def test_known_operation_coverage_controls_wire_version_and_independence(monkeypatch):
    monkeypatch.setitem(P,"min_known_operation_requests",2)
    monkeypatch.setitem(P,"min_novel_normal_requests",2)
    db=sqlite3.connect(":memory:")
    db.execute("CREATE TABLE labels(frame INTEGER PRIMARY KEY,label INTEGER,kind TEXT,episode TEXT,mac TEXT,ip_valid INTEGER)")
    db.executemany("INSERT INTO labels VALUES(?,?,?,?,?,?)",[(1,1,"example","1","m",0),(2,1,"example","2","m",0)])
    row=dict(source="a",destination="b",method="GET",path="/known",wire="HTTP2")
    benign=[dict(row) for _ in range(3)]+[dict(row,path="/new") for _ in range(2)]
    mixed=[dict(row,frame=1),dict(row,frame=2)]
    result=coverage(benign,mixed,db)
    assert result["exploratory_coverage_gate"] and not result["independent_confirmation_supported"]
    result=coverage(benign,[dict(r,wire="HTTP1") for r in mixed],db)
    assert result["attack_requests_on_known_operations"]==2
    assert result["attack_requests_on_known_operations_and_wire"]==0 and not result["exploratory_coverage_gate"]


def test_both_wire_decoders_preserve_http1_and_deduplicate_http2(tmp_path,monkeypatch):
    import coverage5g.audit as module
    first="1\t1.0\ta\tb\t\t\tGET\t/known\n"
    second="2\t1.1\ta\tb\tPOST\t/known\t\t\n"
    calls=[]
    class Process:
        def __init__(self,command,**kwargs):
            calls.append(command)
            generic=any(arg.endswith(",http") for arg in command)
            self.stdout=__import__("io").StringIO(first+second if generic else second)
        def wait(self):return 0
    monkeypatch.setattr(module.subprocess,"Popen",Process)
    rows,ambiguous=module.requests(tmp_path/"example.pcap",{8000},tmp_path)
    assert len(calls)==2 and ambiguous==0
    assert [r["wire"] for r in rows]==["HTTP1","HTTP2"]
