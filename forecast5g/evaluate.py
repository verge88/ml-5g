import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from .data import PLAN,load,evaluate
from .models import predict

def run(inputs,models,out):
    out.mkdir(parents=True,exist_ok=True);trials,provenance=load(inputs,PLAN['test']);scores,frozen=predict(trials,models)
    summaries={d['name']:evaluate(trials,scores[d['name']],d['threshold']) for d in frozen['models']}
    primary=summaries[PLAN['primary_model']];reference=summaries[frozen['selected_statistical']];paired=[]
    for replica in PLAN['test']:
        aa=[r for r in primary['records'] if r['replica']==replica and r['outcome']];bb=[r for r in reference['records'] if r['replica']==replica and r['outcome']]
        assert [r['trial'] for r in aa]==[r['trial'] for r in bb]
        difference=float(np.mean([(a['lead_seconds'] if a['early_detected'] else 0)-(b['lead_seconds'] if b['early_detected'] else 0) for a,b in zip(aa,bb)])) if aa else None
        paired.append(dict(replica=replica,measured_positive_trials=len(aa),mean_lead_difference_seconds=difference))
    differences=[p['mean_lead_difference_seconds'] for p in paired if p['mean_lead_difference_seconds'] is not None]
    rng=np.random.default_rng(20261102);boot=np.mean(rng.choice(differences,size=(10000,len(differences)),replace=True),axis=1) if differences else np.array([0.])
    ci=np.percentile(boot,[2.5,97.5]).tolist();no_extra_false=(primary['benign_false_alerts']<=reference['benign_false_alerts'] and primary['premature_false_alerts']<=reference['premature_false_alerts'])
    report=dict(plan=PLAN,frozen_sha256=hashlib.sha256((models/'frozen.json').read_bytes()).hexdigest(),source_commit=frozen['commit'],test_inputs=provenance,models=summaries,selected_statistical=frozen['selected_statistical'],primary_model=PLAN['primary_model'],paired_deployments=paired,
        primary_mean_lead_difference_seconds=float(np.mean(differences)) if differences else None,exploratory_deployment_bootstrap_95=ci,no_additional_test_false_alerts=no_extra_false,
        hypothesis_supported_on_this_test=bool(differences and ci[0]>0 and no_extra_false),test_model_fits=0,limitations=['three test deployments only; bootstrap is exploratory','bounded laboratory path-delay profiles; not attacks or production faults','normal UE-down intervals are abstentions, coverage reported','one alert per laboratory trial; no claim about continuous alarm policies'])
    (out/'results.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'test-scores.npz',**{k:np.stack(v) for k,v in scores.items()})
    (out/'test-order.json').write_text(json.dumps([dict(replica=t['replica'],trial=t['name']) for t in trials]))
    with (out/'metrics.csv').open('w',newline='') as f:
        names=['model','positive_trials','early_detected','benign_trials','benign_false_alerts','premature_false_alerts','eligible_rows','rows','mean_early_lead_including_misses','median_correct_lead_seconds'];w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for name,r in summaries.items():w.writerow(dict(model=name,**{k:r[k] for k in names if k!='model'}))
    lines=['# Проверка прогноза ухудшения услуги 5G','',f"Первичная модель: {PLAN['primary_model']}. Статистический метод, выбранный только на калибровке: {frozen['selected_statistical']}.",'',
        '| Метод | Раннее обнаружение | Ложные тревоги на штатных эпизодах | Преждевременные тревоги | Среднее упреждение с нулями за пропуски, с |','|---|---:|---:|---:|---:|']
    for name,r in summaries.items():lines.append(f"| {name} | {r['early_detected']}/{r['positive_trials']} | {r['benign_false_alerts']}/{r['benign_trials']} | {r['premature_false_alerts']} | {r['mean_early_lead_including_misses'] or 0:.3f} |")
    lines+=['',f"Разница среднего упреждения первичной модели и выбранного статистического метода: {report['primary_mean_lead_difference_seconds']} с; исследовательский интервал по трём развёртываниям {ci}. Дополнительных ложных тревог у первичной модели: {'нет' if no_extra_false else 'есть'}. Гипотеза на данном тесте: {'поддержана' if report['hypothesis_supported_on_this_test'] else 'не подтверждена'}.",'',
        'Исход определяется фактическими задержками подписанных ответов, а не настройкой воздействия. Прогноз считается только при текущих успешных ответах и актуальных свидетельствах. Первая тревога ранее горизонта8с считается преждевременной и не получает последующего зачёта. Три тестовых развёртывания не дают эксплуатационной сертификации. Применялась ограниченная задержка собственного пользовательского тракта; сетевые атаки и компрометация функций не воспроизводились.']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.input,a.models,a.output)
