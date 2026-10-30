"""Aggregate every registered fit; article tables and exportable scientific plots."""
import argparse,csv,json,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

P=json.loads(Path(__file__).with_name('protocol.json').read_text())
LABELS={'tgn':'Временная графовая сеть (TGN)','gnn':'Графовая сеть (GraphSAGE)','lstm':'Долгая краткосрочная память (LSTM)',
        'transformer':'Трансформер','granomaly_tgn':'Granomaly, TGN','deep_svdd':'Deep SVDD','kitnet':'KitNET','isolation_forest':'Изолирующий лес'}

def main(source,output):
    output.mkdir(parents=True,exist_ok=True);records=[]
    for path in sorted(source.rglob('summary.json')):
        data=json.loads(path.read_text());records.extend(data['records'])
    keys={(r['dataset'],r['method'],r['input'],r['seed']) for r in records}
    expected={(d,m,x,s) for d in P['datasets'] for m in P['models'] for x in P['inputs'] for s in (P['kitnet_seeds'] if m=='kitnet' else P['seeds'])}
    if keys!=expected or len(keys)!=len(records):raise ValueError(f'incomplete/duplicate fits: observed {len(keys)}, expected {len(expected)}, missing {expected-keys}')
    rows=[];counts={}
    for r in records:
        for target,points in r['test_results'].items():
            count_key=r['dataset']+'/'+target
            c=points['0.001'];count={k:c[k] for k in ('rows','labelled','unknown_excluded')}
            if count_key in counts and counts[count_key]!=count:raise ValueError('methods did not score identical complete test rows')
            counts[count_key]=count
            for alpha,metrics in points.items():
                rows.append(dict(dataset=r['dataset'],target=target,method=r['method'],input=r['input'],seed=r['seed'],
                    alpha=float(alpha),threshold=r['thresholds'][alpha],training_seconds=r['training_seconds'],
                    test_seconds=r['test_seconds'][target],training_rows=r['training_rows'],input_dim=r['input_dim'],
                    **{k:metrics[k] for k in ('rows','labelled','unknown_excluded','tp','fp','fn','tn','recall','fpr','precision','f1','average_precision','roc_auc')}))
    with (output/'all-operating-points.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups=defaultdict(list)
    for row in rows:groups[(row['dataset'],row['target'],row['method'],row['input'],row['alpha'])].append(row)
    aggregate=[]
    for (d,t,m,x,alpha),values in sorted(groups.items()):
        item=dict(dataset=d,target=t,method=m,input=x,alpha=alpha,fits=len(values),rows=values[0]['rows'],labelled=values[0]['labelled'])
        for metric in ('average_precision','roc_auc','recall','fpr','f1','precision','training_seconds','test_seconds','fp','tp'):
            vv=[v[metric] for v in values if v[metric] is not None]
            item[metric+'_mean']=float(np.mean(vv)) if vv else None
            item[metric+'_std']=float(np.std(vv,ddof=1)) if len(vv)>1 else None
            item[metric+'_min']=min(vv) if vv else None;item[metric+'_max']=max(vv) if vv else None
        aggregate.append(item)
    with (output/'aggregate.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(aggregate[0]));writer.writeheader();writer.writerows(aggregate)
    summary=dict(expected_fits=len(expected),completed_fits=len(records),unique_test_counts=counts,aggregate=aggregate,
        protocol=P,analysis_plan=json.loads(Path(__file__).with_name('analysis-plan.json').read_text()))
    (output/'aggregate.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# Финальный эксперимент: обучение и сравнение с опубликованными методами','',
        f'Завершено обучений: {len(records)}. Все пригодные тестовые запросы оценены без сокращения размера выборки. Правила в сравнении отсутствуют.','',
        'Основная конфигурация до обучения: TGN с контекстом. Основной аналог: Granomaly TGN; дополнительные — KitNET, Deep SVDD и изолирующий лес.','',
        '## Размер проверочных частей','', '| Набор / запись | Запросов с оценкой | С известной меткой | Неизвестная метка |','|---|---:|---:|---:|']
    for k,v in counts.items():lines.append(f'| {k} | {v["rows"]} | {v["labelled"]} | {v["unknown_excluded"]} |')
    lines+=['','Число строк, повторов обучения и начальных состояний не равно числу независимых развёртываний. В CTD5G одна смешанная коллекция. Данные и их покрытие уже изучались ранее: это ретроспективный опыт.','',
        '## Сравнение: контекст, калибровочная цель 0,1%','',
        'AP — средняя точность по ранжированию (ступенчатая площадь под кривой точность–полнота). Средние и разброс относятся к начальным состояниям модели. KitNET имеет единственное фиксированное авторами начальное состояние.','',
        '| Набор / запись | Метод | AP | Полнота, % | Фактические ложные тревоги, % | F1 |','|---|---|---:|---:|---:|---:|']
    for a in aggregate:
        if a['input']!='context' or a['alpha']!=.001 or a['target']=='clean_holdout':continue
        def f(k,mult=1):return '—' if a[k] is None else f'{a[k]*mult:.4f}'
        lines.append(f'| {a["dataset"]}/{a["target"]} | {LABELS[a["method"]]} | {f("average_precision_mean")} | {f("recall_mean",100)} | {f("fpr_mean",100)} | {f("f1_mean")} |')
    lines+=['','Полные результаты обоих составов признаков, всех начальных состояний и обеих калибровочных целей находятся в CSV. Порог не выбирался по тестовым меткам. Превышение фактического лимита ложных тревог сохраняется в результатах.','',
        '## Ограничения и воспроизводимость','',
        'Сравнение выполнено на одинаковых входных матрицах и разбиениях. Granomaly использует опубликованные нейросетевые модули без изменений; загрузчик, признаки, число эпох, размеры пакетов, калибровка и обновление памяти приведены к общему протоколу. Это адаптация для прямого сравнения, а не воспроизведение исходных процентов статьи.',
        'KitNET использует авторский алгоритм на общих признаках сервисных запросов; полный исходный тракт Kitsune из 115 пакетных признаков не воспроизводится. Deep SVDD реализует одноклассовую функцию потерь с фиксированным центром и сетью без смещений, адаптированную к табличным данным.',
        'Обучение использует только штатные данные. Отрицательные примеры графовых и последовательностных моделей создаются перестановкой концов обучающих связей и не добавляются в историю. Реальные метки атак не участвуют в обучении, калибровке и обновлении памяти.',
        'Ни контрольные суммы, ни MAC-адреса, ни маркеры, идентификаторы процедур, ответы HTTP, абсолютные даты или неподтверждённые роли не являются входами моделей. Подтверждение вредоносного результата и совместные свидетельства контрольных единиц в данных отсутствуют. Опыт проверяет компоненты машинного обучения, а не полный метод контроля целостности.',
        'Время оценки графовых моделей включает воспроизведение штатной обучающей истории. Это пакетная производительность, а не задержка тревоги в действующем ядре. Условные доверительные интервалы по блокам внутри записи не характеризуют перенос на независимые развёртывания.','',
        '## Источники аналогов','',
        '1. [Fritz et al. Granomaly, NOMS 2025](https://doi.org/10.1109/NOMS57970.2025.11073582). [Исходная реализация](https://github.com/TF0x42/granomaly).',
        '2. [Mirsky et al. Kitsune, NDSS 2018](https://arxiv.org/abs/1802.09089). [Авторский KitNET](https://github.com/ymirsky/KitNET-py).',
        '3. [Ruff et al. Deep One-Class Classification, ICML 2018](https://proceedings.mlr.press/v80/ruff18a.html).',
        '4. [Liu et al. Isolation Forest, ICDM 2008](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf).']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    plots(aggregate,output)
    print(json.dumps(dict(fits=len(records),unique_test_counts=counts)),flush=True)

def plots(aggregate,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none'})
    targets=[('ctd5g','mixed'),('granomaly','nrf'),('granomaly','smf')]
    fig,axes=plt.subplots(1,3,figsize=(13,5),sharey=True)
    for ax,(dataset,target) in zip(axes,targets):
        for offset,composition,color in [(-.16,'base','#8595a5'),(.16,'context','#176b9c')]:
            vv=[next(a for a in aggregate if (a['dataset'],a['target'],a['method'],a['input'],a['alpha'])==(dataset,target,m,composition,.001)) for m in P['models']]
            means=np.array([v['average_precision_mean'] for v in vv]);low=np.array([v['average_precision_min'] for v in vv]);high=np.array([v['average_precision_max'] for v in vv])
            ax.barh(np.arange(len(vv))+offset,means,height=.28,color=color,label='Базовые признаки' if composition=='base' else 'С контекстом')
            ax.errorbar(means,np.arange(len(vv))+offset,xerr=np.maximum(0,np.stack((means-low,high-means))),fmt='none',ecolor='#333333',capsize=2)
        ax.set_title(dataset+'/'+target);ax.set_xlim(0,1.04);ax.set_xlabel('Средняя точность по ранжированию (AP)');ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(np.arange(len(P['models'])),[LABELS[m] for m in P['models']]);axes[0].invert_yaxis();axes[-1].legend(loc='lower right',fontsize=8)
    fig.suptitle('Общие тестовые записи; полосы разброса по начальным состояниям');fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(output/('comparison-ap.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,(dataset,target) in zip(axes,targets):
        for i,m in enumerate(P['models']):
            a=next(v for v in aggregate if (v['dataset'],v['target'],v['method'],v['input'],v['alpha'])==(dataset,target,m,'context',.001))
            ax.scatter(a['fpr_mean']*100,a['recall_mean']*100,label=LABELS[m],s=40,marker=['o','s','^','D','v','P','X','*'][i])
        ax.axvline(.1,color='#bb3333',linestyle='--',linewidth=1);ax.set_title(dataset+'/'+target);ax.set_xlabel('Фактические ложные тревоги, %');ax.grid(alpha=.2)
    axes[0].set_ylabel('Полнота, %');axes[-1].legend(bbox_to_anchor=(1.04,1),loc='upper left',fontsize=8)
    fig.suptitle('Контекстные признаки, калибровочная цель 0,1%');fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(output/('recall-fpr.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.input,a.output)
