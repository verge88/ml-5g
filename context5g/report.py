"""Show complete comparisons and mark unavailable primary evidence explicitly."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

P=json.loads(Path(__file__).with_name("protocol.json").read_text())


def run(root,output):
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in P["models"]+["rules"]:
        summary=json.loads((root/f"context-{name}"/"summary.json").read_text())
        if summary["protocol"]!=P:raise ValueError("protocol mismatch")
        expected=20 if name=="rules" else 12
        if len(summary["records"])!=expected:raise ValueError("missing fits/rules")
        records.extend(summary["records"])
    detailed=[]
    for record in records:
        target=record["target"]
        for point in record["operating_points"]:
            metrics=point["results"][target];holdout=point["results"]["normal_holdout"]
            row=dict(model=record["model"],fold=record["fold"],input=record["input"],seed=record["seed"],
                alpha=point["alpha"],threshold=point["threshold"],calibration_fpr=point["calibration_fpr"],
                calibration_budget_met=point["calibration_budget_met"],test_budget_met=point["test_budget_met"],
                target_normal_requests=metrics["normal_requests"],target_false_alerts=metrics["false_alerts"],
                target_fpr=metrics["fpr"],holdout_normal_requests=holdout["normal_requests"],
                holdout_false_alerts=holdout["false_alerts"],holdout_fpr=holdout["fpr"],
                holdout_false_marks_per_span_hour=holdout["false_marks_per_request_span_hour"])
            for scope,value in metrics["scopes"].items():
                row.update({scope+"_requests":value["requests"],scope+"_detected":value["detected"],
                    scope+"_recall":value["recall"]})
            detailed.append(row)
    groups=defaultdict(list)
    for row in detailed:groups[(row["model"],row["fold"],row["input"],row["alpha"])].append(row)
    aggregated=[]
    for (model,fold,composition,alpha),rows in groups.items():
        row=dict(model=model,fold=fold,input=composition,alpha=alpha,initializations=len(rows),
            all_budget_met=all(r["test_budget_met"] and r["calibration_budget_met"] for r in rows),
            target_fpr=mean(r["target_fpr"] for r in rows),holdout_fpr=mean(r["holdout_fpr"] for r in rows),
            holdout_false_marks_per_span_hour=mean(r["holdout_false_marks_per_span_hour"] for r in rows))
        for scope in [P["primary_scope"],*P["secondary_scopes"]]:
            values=[r[scope+"_recall"] for r in rows]
            row[scope+"_requests"]=rows[0][scope+"_requests"]
            row[scope+"_recall"]=mean(values) if all(v is not None for v in values) else None
            row[scope+"_seed_sd"]=pstdev(values) if all(v is not None for v in values) else None
        aggregated.append(row)
    hypotheses=[]
    references=P["rules"]+["shallow_tree"]
    for row in aggregated:
        if row["model"] not in P["models"]:continue
        for scope in [P["primary_scope"],"known_edge_candidates","data_access_all_edges"]:
            candidates=[r[scope+"_recall"] for r in aggregated if r["fold"]==row["fold"] and
                r["alpha"]==row["alpha"] and r["model"] in references and r["all_budget_met"] and
                r[scope+"_recall"] is not None]
            positive=row[scope+"_requests"];recall=row[scope+"_recall"]
            best=max(candidates) if candidates else None
            eligible=positive>0 and row["all_budget_met"] and best is not None
            gain=recall-best if eligible else None
            status="NOT_IDENTIFIABLE_NO_EXAMPLES" if not positive else (
                "MODEL_FALSE_ALERT_BUDGET_FAILED" if not row["all_budget_met"] else
                "NO_ELIGIBLE_RULE_COMPARATOR" if best is None else
                "EXPLORATORY_CRITERION_MET" if gain>=P["gain_required"] else "CRITERION_NOT_MET")
            hypotheses.append(dict(model=row["model"],fold=row["fold"],input=row["input"],alpha=row["alpha"],
                scope=scope,requests=positive,best_eligible_reference_recall=best,gain=gain,status=status))
    for name,rows in (("all-operating-points.csv",detailed),("comparison.csv",aggregated),("hypotheses.csv",hypotheses)):
        with (output/name).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=["# Контекстное обнаружение на опубликованных записях 5G", "",
        "Основная гипотеза о доступе к абонентским данным на известных связях НЕ ПРОВЕРЕНА: "
        "в обоих проверочных захватах нет подходящих размеченных примеров. Вторичные результаты "
        "не заменяют отсутствующее свидетельство по основной гипотезе.", "",
        "Выполнены 60 обучений пяти основных моделей и четыре обучения малого решающего дерева. "
        "С ними сравниваются восемь зафиксированных правил и статистических детекторов. "
        "Пороги выбраны по штатной калибровке для целевых долей 0,1% и 1%. "
        "Записи изучались ранее; это ретроспективное исследование, а не новая независимая проверка.", "",
        "Таблица ниже соответствует калибровочной цели 0,1%. Выполнение бюджета требует "
        "непревышения этой доли как на чистой отложенной записи, так и среди штатных запросов "
        "проверочного захвата при всех инициализациях. Пороги по проверке не менялись.", "",
        "| Захват | Метод | Признаки | Кандидатов на известных связях | Полнота | Ложные в захвате | Ложные в чистой записи | Бюджет |",
        "|---|---|---|---:|---:|---:|---:|---|"]
    for row in aggregated:
        if row["alpha"]!=P["primary_alpha"]:continue
        recall=row["known_edge_candidates_recall"]
        text.append(f'| {row["fold"]} | {row["model"]} | {row["input"]} | '
            f'{row["known_edge_candidates_requests"]} | {"нет примеров" if recall is None else f"{100*recall:.2f}%"} | '
            f'{100*row["target_fpr"]:.3f}% | {100*row["holdout_fpr"]:.3f}% | {"да" if row["all_budget_met"] else "нет"} |')
    text += ["", "Наблюдаемые адреса не доказывают роли и полномочия вызывающих функций. "
        "Проверка последовательностей относится к истории наблюдаемого отправителя, а не к "
        "полному автомату абонентской процедуры. Результат по меткам набора не доказывает утечку данных.", "",
        "Часовые частоты в CSV относятся только к временному охвату запросов чистой отложенной "
        "записи. Запросы и повторные инициализации не являются независимыми атаками; "
        "доверительные интервалы для генеральной совокупности не рассчитывались. "
        "Ни успех задания Actions, ни вторичный выигрыш не требуют победы сложной модели."]
    (output/"REPORT.md").write_text("\n".join(text),encoding="utf-8")
    summary=dict(protocol=P,primary_status="NOT_IDENTIFIABLE_NO_EXAMPLES",main_model_fits=60,
        shallow_tree_fits=4,rows=aggregated,hypotheses=hypotheses,limitations=P["limitations"])
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();run(a.input,a.output)
