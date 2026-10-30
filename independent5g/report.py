"""Cluster bootstrap across deployments; seeds never enlarge the sample size."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .collect import P

RATIOS = {"recall": ("true_alerts","fault_rows"), "fpr": ("false_alerts","normal_rows"),
    "episode_recall": ("detected_episodes","fault_episodes"),
    "false_alerts_per_hour": ("false_alerts","normal_seconds"),
    "false_alarm_episodes_per_hour": ("false_alarm_episodes","normal_seconds")}


def ratio(rows, key):
    a,b = RATIOS[key]
    factor = 3600 if key.endswith("per_hour") else 1
    return factor*sum(r[a] for r in rows)/sum(r[b] for r in rows)


def interval(values):
    return [float(x) for x in np.quantile(values,[.025,.975])]


def run(root, output):
    root,output = Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    runs = []
    baseline = None
    for model in P["models"]:
        summary = json.loads((root/f"independent-model-{model}"/"summary.json").read_text())
        runs.extend(summary["runs"])
        if baseline is None:
            baseline = summary["direct_availability_rule"]
    expected = {(m,i,s) for m in P["models"] for i in P["inputs"] for s in P["seeds"]}
    if len(runs) != len(expected) or {(r["model"],r["input"],r["seed"]) for r in runs} != expected:
        raise ValueError("missing or repeated fit")
    for r in runs:
        if [x["replica"] for x in r["results"]] != P["splits"]["test"]:
            raise ValueError("changed final deployment assignment")
    rng = np.random.default_rng(P["bootstrap_seed"])
    draws = rng.integers(0,3,size=(P["bootstrap_draws"],3))
    rows, samples = [], {}
    for model in P["models"]:
        for composition in P["inputs"]:
            rr = [r for r in runs if r["model"] == model and r["input"] == composition]
            row = dict(model=model,input=composition,test_deployments=3,initialization_repeats=len(rr))
            for key in RATIOS:
                point = float(np.mean([ratio(r["results"],key) for r in rr]))
                boot = np.array([np.mean([ratio([r["results"][j] for j in ids],key) for r in rr]) for ids in draws])
                samples[(model,composition,key)] = boot
                lo,hi = interval(boot)
                row.update({key:point,key+"_lo":lo,key+"_hi":hi,
                    key+"_seed_min":min(ratio(r["results"],key) for r in rr),
                    key+"_seed_max":max(ratio(r["results"],key) for r in rr)})
            rows.append(row)
    comparisons = []
    for row in rows:
        for reference_model,reference_input in (("hgb",row["input"]),(row["model"],"metadata")):
            if (row["model"],row["input"]) == (reference_model,reference_input):
                continue
            for key in RATIOS:
                delta = samples[(row["model"],row["input"],key)]-samples[(reference_model,reference_input,key)]
                lo,hi = interval(delta)
                reference = next(r for r in rows if r["model"]==reference_model and r["input"]==reference_input)
                comparisons.append(dict(model=row["model"],input=row["input"],reference_model=reference_model,
                    reference_input=reference_input,metric=key,delta=row[key]-reference[key],lo=lo,hi=hi))
    for name,data in (("comparison.csv",rows),("paired-differences.csv",comparisons)):
        with (output/name).open("w",newline="",encoding="utf-8") as f:
            writer = csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    text = ["# Независимые развёртывания Open5GS: сравнение моделей", "",
        "Разделение: 3 развёртывания для обучения, 3 для калибровки, 3 для проверки. "
        "Повторы инициализации учитываются отдельно. Интервалы — исследовательская кластерная "
        "бутстреп-оценка по трём проверочным развёртываниям. Они не дают надёжной оценки всей генеральной совокупности.", "",
        "| Модель | Признаки | Полнота | Ложные срабатывания | Тревог/ч | Полнота по эпизодам |",
        "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        text.append(f'| {r["model"]} | {r["input"]} | {r["recall"]:.3f} | {r["fpr"]:.4f} | {r["false_alerts_per_hour"]:.2f} | {r["episode_recall"]:.3f} |')
    text.extend(["",f'Прямое правило доступности: полнота {ratio(baseline,"recall"):.3f}; '
        f'ложных тревог/ч {ratio(baseline,"false_alerts_per_hour"):.2f}.',"",
        "Пороги выбраны только на калибровочных данных. Различия фактической частоты ложных тревог "
        "на проверке необходимо учитывать при сравнении полноты. Все пары сравнений доступны в CSV; "
        "интервалы не скорректированы на множественные сравнения.","",
        "Задержки в исходных результатах относятся ко времени первого отмеченного наблюдения; "
        "они не включают ожидание формирования пакета входных данных и вычисление модели.","",
        "Ограничения: только управляемые отказы доступности, постоянный состав из двух типов функций, "
        "фиксированные обращения чтения; семантические атаки не исследовались. "
        "Разные периоды опроса меняют измерительную нагрузку, а не абонентский трафик."])
    (output/"REPORT.md").write_text("\n".join(text),encoding="utf-8")
    (output/"summary.json").write_text(json.dumps(dict(protocol=P,rows=rows,paired_differences=comparisons,
        direct_availability_rule=baseline),indent=2))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    run(args.input,args.output)
