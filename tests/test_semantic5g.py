import csv
import json
from semantic5g.analyze import candidate, family, native_experiment, read_csv, signatures, template


def test_template_removes_info_delimiter_and_subscriber():
    assert template('/nudm-ueau/v1/suci-0-123-45-0-0-0-abcdef/security-information/generate-auth-data,') == '/nudm-ueau/v1/{subscriber}/security-information/generate-auth-data'
    assert template('/nudm-sdm/v2/imsi-123456/am-data?fields=secret') == '/nudm-sdm/v2/{subscriber}/am-data'


def test_api_family_does_not_confuse_udr_and_udm():
    assert family('/nudr-dr/v1/subscription-data/{subscriber}/am-data') == 'other'
    assert family('/nudm-sdm/v2/{subscriber}/am-data') == 'subscription_data'


def test_same_edge_different_operation_is_distinguishable():
    ordinary = dict(source='scp', destination='amf', method='POST', path='/namf-callback/v1/{subscriber}/sm-context-status')
    unfamiliar = dict(ordinary, method='PUT', path='/namf-callback/v1/{subscriber}/dereg-notify')
    assert signatures(ordinary)[0] == signatures(unfamiliar)[0]
    assert signatures(ordinary)[1] != signatures(unfamiliar)[1]


def test_proxied_interleaved_request_is_not_direct_producer_candidate():
    row = dict(label='udm_subscriber_data_curl', family='subscription_data', destination='10.211.55.11')
    assert not candidate(row)
    assert candidate(dict(row, destination='10.211.55.8'))


def test_response_pairing_separates_connection_generations(tmp_path):
    fields = ['frame.time_epoch', 'frame.number', 'ip.src', 'srcport', 'ip.dst', 'dstport', '_ws.col.info', 'Label']
    data = [
        ['1','1','a','123','b','7777','[SYN]','normal'],
        ['2','2','a','123','b','7777','HEADERS[1]: GET /test/v1/resource','normal'],
        ['3','3','b','7777','a','123','HEADERS[1]: 200 OK','normal'],
        ['4','4','a','123','b','7777','[SYN]','normal'],
        ['5','5','a','123','b','7777','HEADERS[1]: GET /test/v1/resource','normal'],
        ['6','6','b','7777','a','123','HEADERS[1]: 404 Not Found','normal'],
    ]
    path = tmp_path/'recorded.csv'
    with path.open('w',newline='') as f:
        w = csv.writer(f); w.writerow(fields); w.writerows(data)
    rows, _ = read_csv(path)
    assert [r['status'] for r in rows] == [200,404]


def test_native_auth_result_uses_body_and_connection_id(tmp_path):
    # Parser fixture only: these messages are never transmitted or counted as attacks.
    packets=[]
    for conn, outcome in [('1','AUTHENTICATION_SUCCESS'),('2','AUTHENTICATION_FAILURE')]:
        for response in (False,True):
            headers = {':status':'200','content-type':'application/json'} if response else {':method':'PUT',':path':'/nausf-auth/v1/ue-authentications/context/5g-aka-confirmation'}
            frame = {'http2.type':'1','http2.streamid':'1','http2.header':[{'http2.header.name':k,'http2.header.value':v} for k,v in headers.items()]}
            h2 = [{'http2.stream':frame}]
            if response:
                h2.append({'http2.stream':{'http2.type':'0','http2.streamid':'1','http2.data.data':json.dumps({'authResult':outcome}).encode().hex(), 'http2.flags_tree':{'http2.flags.end_stream':'1'}}})
            else:
                frame['http2.flags_tree']={'http2.flags.end_stream':'1'}
            packets.append({'_source':{'layers':{'frame':{'frame.number':str(len(packets)+1)},'tcp':{'tcp.stream':conn,'tcp.srcport':'7777' if response else '123','tcp.seq':'2' if response else '1'},'http2':h2}}})
    saved=tmp_path/'h2.json'; saved.write_text(json.dumps(packets))
    pcap=tmp_path/'placeholder';pcap.write_bytes(b'fixture')
    result=native_experiment(saved,pcap,tmp_path)
    assert {r['authResult'] for r in result['auth_results']} == {'AUTHENTICATION_SUCCESS','AUTHENTICATION_FAILURE'}
    assert all(r['status']=='200' and r['complete'] and r['observed_hops']==1 for r in result['auth_results'])
