"""Verify public evidence and test frozen models without fitting to new trials."""
import argparse,hashlib,json,itertools
from pathlib import Path
import dpkt,numpy as np,torch
from cu5g.common import verified,validate_witness
from joint5g.deep import Net,windows,score
from joint5g.evaluate import vector
from .evidence import receipt,dnn_result

def metrics(rows,events,alarm):
    normal=np.array([r['stage'] in ['test_normal','test_baseline'] for r in rows]);out=[]
    for e in events:
        ids=[i for i,r in enumerate(rows) if e['start_tick']<=r['tick']<e['end_tick'] and alarm[i]]
        out.append(dict(episode=e['episode'],slot=e['slot'],detected=bool(ids),delay_seconds=(rows[ids[0]]['monotonic_ns']-e['start_ns'])/1e9 if ids else None))
    return dict(normal=int(normal.sum()),fp=int(np.asarray(alarm)[normal].sum()),fpr=float(np.asarray(alarm)[normal].mean()),episodes=out,detected=sum(e['detected'] for e in out),tested=len(out))

def pcap_audit(path):
    count=0;addresses=set()
    with path.open('rb') as stream:
        reader=dpkt.pcap.Reader(stream);link=reader.datalink()
        for ts,data in reader:
            try:
                if link in [12,101,228]: ip=dpkt.ip.IP(data)
                elif link==1:ip=dpkt.ethernet.Ethernet(data).data
                elif link==113:ip=dpkt.sll.SLL(data).data
                else:continue
                if isinstance(ip.data,dpkt.tcp.TCP) and 18081 in [ip.data.sport,ip.data.dport]:
                    count+=1;addresses.add('.'.join(map(str,ip.src)));addresses.add('.'.join(map(str,ip.dst)))
            except (ValueError,dpkt.dpkt.NeedData,AttributeError):continue
    assert count>0 and any(a.startswith('10.45.') and a!='10.45.0.1' for a in addresses)
    return dict(linktype=link,application_tcp_packets=count,addresses=sorted(addresses),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def run(inputs,modelroot,out):
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=True);frozen=[]
    for p in modelroot.rglob('frozen.json'):
        for d in json.loads(p.read_text())['models']:
            if d['arm']=='broad_full':frozen.append((d,p.parent/(d['name']+'.pt')))
    assert len(frozen)==18
    loaded=[]
    for d,p in frozen:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==d['weights_sha256']
        net=Net(len(d['scaler_mean']),d['arch'],d['length']);net.load_state_dict(torch.load(p,map_location='cpu',weights_only=True));loaded.append((d,net))
    results=[];evidence=[];schedule=[]
    for folder in sorted(inputs.glob('e2e-native-*')):
        complete=json.loads((folder/'complete.json').read_text());replica=complete['replica'];public=json.loads((folder/'guest-public.json').read_text());assert public['boot_id']!=public['host_boot_id']
        ue=[json.loads(s) for s in (folder/'ue/receipts.jsonl').read_text().splitlines()];seen=set();valid=0
        for row in ue:
            if row['returncode']==0:
                assert receipt(row,public,now_ns=row['unix_ns']);nonce=row['request']['nonce'];assert nonce not in seen;seen.add(nonce);valid+=1
        assert valid>0
        nf_count=0
        for slot in ['ausf','nssf']:
            for line in (folder/f'nf-public-{slot}.jsonl').read_text().splitlines():
                p=verified(json.loads(line),folder/f'{slot}.pub');assert p['mac_checked_before_signing'] is True and p['request']['subject_id']==slot and p['response']['request_nonce']==p['request']['nonce'];nf_count+=1
        ue_logs='\n'.join(p.read_text(errors='replace') for p in (folder/'ue').glob('ue-*.log'))
        assert 'Registration accept received' in ue_logs and 'PDU Session establishment is successful' in ue_logs
        subscription=json.loads((folder/'live-subscription-controls.json').read_text());denial=subscription['denial']
        state=dnn_result((folder/f"ue/ue-{denial['attempt']}.log").read_text(errors='replace'))
        assert state['registered'] and state['denied'] and not state['established'] and receipt(subscription['restored_application'],public,now_ns=subscription['restored_application']['unix_ns'])
        packet=pcap_audit(folder/'ue-user-plane.pcap');witness_count=0
        for mode in complete['modes']:
            p=folder/mode;rows=[json.loads(s) for s in (p/'observations.jsonl').read_text().splitlines()];events=json.loads((p/'events.json').read_text());manifest=json.loads((p/'manifest.json').read_text())
            boundaries=json.loads((p/'boundary-recovery.json').read_text());assert len(boundaries)==4 and all(b['outside_scored_windows'] and receipt(b['restored_application'],public,now_ns=b['restored_application']['unix_ns']) for b in boundaries)
            req=json.loads((p/'cu-requests.json').read_text());comp=json.loads((p/'cu-completions.json').read_text());assert len(req)==manifest['cu_requests'] and len({v['tick'] for v in req})==len(req)
            assert len(req)<=len(rows) and len(rows)==156
            bound=json.loads((p/'bound-observations.json').read_text());assert len(bound)==len(rows)
            for r,b in zip(rows,bound):
                m=r['observations']['ausf']['integrity_monitor'];assert m['observation_id']==b['observation_id']
                if b['witness']:
                    validate_witness(b['witness'],folder/'runtime-witness.pub',b['challenge'],now_ns=b['observed_unix_ns']);witness_count+=1
                if r['application_ok']:assert receipt(r['application'],public,now_ns=r['unix_ns'])
            byreq={(r['slot'],r['request']['cycle']):r for r in req};latencies=[]
            for c in comp:
                t=verified(c['public_bundle'],folder/f"{c['slot']}.pub");assert t['request']['cycle']==c['cycle']
                if (c['slot'],c['cycle']) in byreq:latencies.append((c['observed_ns']-byreq[c['slot'],c['cycle']]['issued_ns'])/1e9)
            hard=np.array([r['observations']['ausf']['integrity_monitor']['hard_alarm'] for r in rows]);app=np.array([not r['application_ok'] for r in rows]);cu=np.array([any(r['cu_unavailable'].values()) for r in rows])
            for method,a in [('integrity_only',hard),('application_path',app),('cu_availability',cu)]:results.append(dict(replica=replica,schedule=mode,model='none',method=method,**metrics(rows,events,a)))
            raw=[];ctx=[]
            for r in rows:a,c=vector(r['observations']);raw.append(a);ctx.append(c)
            data=dict(raw=np.asarray(raw),context=np.asarray(ctx));arrays=dict(hard=hard,application_missing=app,cu_unavailable=cu)
            for d,net in loaded:
                v,c=windows(((data[d['composition']]-np.asarray(d['scaler_mean']))/np.asarray(d['scaler_scale'])).astype('float32'),d['length']);s=score(net,v,c);ml=s>d['threshold'];full=hard|ml;extended=full|app
                arrays['score_'+d['name']]=s
                for method,a in [('ml',ml),('full',full),('full_with_application_witness',extended)]:results.append(dict(replica=replica,schedule=mode,model=d['arch'],composition=d['composition'],seed=d['seed'],method=method,**metrics(rows,events,a)))
            np.savez_compressed(out/f'{replica}-{mode}.npz',**arrays)
            schedule.append(dict(replica=replica,schedule=mode,order=manifest['order'],ticks=len(rows),requests=len(req),completed=len(comp),median_cu_seconds=float(np.median(latencies)) if latencies else None,unknown=sum(r['observations']['ausf']['integrity_monitor']['decision']['verdict']=='UNKNOWN' for r in rows),bytes=sum(r['received_bytes'] for r in rows),total_collection_seconds=sum(r['collection_elapsed_ms'] for r in rows)/1000,manual_boundary_resets=len(boundaries),manual_reset_seconds=sum((b['end_ns']-b['start_ns'])/1e9 for b in boundaries),unassisted_application_at_recovery_end=sum(b['application_at_fixed_recovery_end'] for b in boundaries),selected={s:sum(r['selected']==s for r in rows) for s in ['ausf','nssf']},cu_availability=metrics(rows,events,cu),application=metrics(rows,events,app)))
        evidence.append(dict(replica=replica,verified_application_receipts=valid,verified_nf_signatures=nf_count,verified_runtime_witnesses=witness_count,pcap=packet,guest=public,roles=complete['roles'],live_subscription_control=subscription))
    assert len(evidence)==3 and len(schedule)==9
    report=dict(protocol=json.loads(Path(__file__).with_name('plan.json').read_text()),evidence=evidence,schedules=schedule,results=results,models=[d for d,_ in frozen],new_model_fits_in_evaluation=0,physical_core_deployments=3,physical_fault_episodes=36)
    (out/'results.json').write_text(json.dumps(report,indent=2))
    lines=['# Сквозной лабораторный опыт 5GC','','Три развёртывания полного Open5GS, виртуальные абонент и базовая станция UERANSIM, 36 локальных пауз процессов. Три расписания проверены последовательно внутри каждого развёртывания с чередованием порядка. Это не девять независимых сетей.','',
      'Подписи прикладных квитанций получены в виртуальной машине с отдельным ядром. Трафик прошёл через абонентский туннель и UPF: сохранены журналы регистрации/сеанса и захват на ogstun. Свидетель подтверждает получение прикладного запроса, а не идентичность NF.','',
      '| Расписание | Запросы / циклы | Завершено КЦ | Среднее медиан времени КЦ, с | UNKNOWN, % |','|---|---:|---:|---:|---:|']
    for mode in ['fixed','risk','ml_risk']:
        v=[s for s in schedule if s['schedule']==mode];n=sum(s['ticks'] for s in v);lat=[s['median_cu_seconds'] for s in v if s['median_cu_seconds'] is not None]
        lines.append(f"| {mode} | {sum(s['requests'] for s in v)} / {n} | {sum(s['completed'] for s in v)} | {np.mean(lat) if lat else float('nan'):.3f} | {100*sum(s['unknown'] for s in v)/n:.2f} |")
    lines+=['','18 ранее обученных MLP/LSTM/Transformer проверены с прежними весами, масштабами и порогами. Обучения на этих тестах нет. В таблице указаны результаты объединения прежнего контроля целостности и модели; свидетель прикладного пути рассмотрен отдельно.','',
      '| Расписание | Модель / признаки | Эпизоды из12: 11 /29 /47 | Средняя доля ложных тревог, % |','|---|---|---|---:|']
    for mode,arch,composition in itertools.product(['fixed','risk','ml_risk'],['mlp','lstm','transformer'],['raw','context']):
        v=[r for r in results if r['schedule']==mode and r['model']==arch and r.get('composition')==composition and r['method']=='full'];counts=[sum(r['detected'] for r in v if r['seed']==seed) for seed in [11,29,47]]
        lines.append(f"| {mode} | {arch} / {composition} | {' / '.join(map(str,counts))} | {100*np.mean([r['fpr'] for r in v]):.4f} |")
    lines+=['','На каждом развёртывании после всех оцениваемых испытаний дополнительно проверены штатный отказ зарегистрированному абоненту при запросе DNN вне его подписки и восстановление разрешённой услуги. Это проверка соответствия правилам допуска; она не доказывает обнаружение семантических атак моделью.','',
      'После каждого фиксированного окна наблюдения восстановления выполнялся одинаковый ручной перезапуск тестовых SMF/UPF и UE за пределами оцениваемых окон. Его время и действия сохранены отдельно. Автоматическое восстановление ядра не утверждается.','',
      'Ограничения: лаборатория и имитируемый радиоканал; общий физический хост/гипервизор; прежнее удостоверение NF использует NF-происхождение экспорта и охватывает AUSF/NSSF; отрицательные контроли авторизационного договора выполняются офлайн. Проверка производственной сети и реальных несанкционированных семантических операций не выполнена. Подписи новых NF-транскриптов проверяются по открытым ключам; исходные MAC проверялись при сборе, секреты не публикуются.','',
      'Равенство числа запланированных запросов не означает равенства фактических затрат. В results.json сохранены принятые запросы, завершения, байты и время. Модели могут потерять переносимость при изменении расписания и стенда; все результаты публикуются независимо от знака эффекта.']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.input,a.models,a.output)
