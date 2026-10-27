# -*- coding: utf-8 -*-
"""
scenarios/build_validation_sources.py - таблицы этапа 4 ТЗ от 13.10.2026 (п. 1 и итог для fit_vs_holdout).

Собирает ИЗ ИМЕЮЩИХСЯ файлов, ничего не добавляя от себя:
  data/experimental_points.json            - источник, рисунок, препарат, n, правило погрешности, оцифровка
  unified_outputs/barbour_*_report.json    - χ² по наборам из единой подгонки
Результат (XLSX, CSV не используется):
  validation_data/validation_sources.xlsx  - experiment -> preparation -> stimulus -> observable ->
                                             model observable -> uncertainty -> fit / hold-out
  validation_data/fit_vs_holdout.xlsx      - χ² и χ²/точку по наборам и вариантам модели
Поля, которых нет в исходных файлах (возраст животных, DOI вне Matthews), помечены
"не указано в data/experimental_points.json: проверить по статье" и не заполнялись догадками.

Запуск из папки diplom:  python -m scenarios.build_validation_sources
"""
import json
from pathlib import Path

import openpyxl

from scenarios.unified_fit import FIT_SETS, VALIDATION_SETS

DATA = Path("data/experimental_points.json")
REPORTS = {"baseline_physical (Ω=0)": "unified_outputs/barbour_baseline_physical_report.json",
           "exploratory_drift (Ω≠0)": "unified_outputs/barbour_exploratory_drift_report.json"}
OUT = Path("validation_data")
UNKNOWN = "не указано в data/experimental_points.json: проверить по статье"
# DOI из ТЗ от 13.10.2026 (раздел 7) и из самого файла данных (Matthews)
DOI = {"Matthews": "10.1093/cercor/bhab440", "Clements": "10.1126/science.1359647",
       "Diamond J.S. 2005": "10.1523/JNEUROSCI.5125-04.2005"}
# модельная наблюдаемая по виду источника: из source_kind и note файла данных
MODEL_OBS = {
    "synaptic": "свободный глутамат c(r,t) -> заполнение датчика iGluSnFr (Kd, τ_off) -> размытие PSF -> пик ΔF/F от расстояния",
    "uncaging": "c(r,t) от точечного выброса N_uncaging молекул -> датчик iGluSnFr -> PSF -> пик ΔF/F",
    "uncaging_nmda": "c(r,t) -> схема NMDA-рецептора -> амплитуда ответа от расстояния",
    "uncaging_ampa": "c(r,t) -> схема AMPA-рецептора -> пик открытых каналов от расстояния",
}


def doi_for(source):
    for k, v in DOI.items():
        if k in source:
            return v
    return "не указан в файле данных и в ТЗ"


def role(ds_id):
    if ds_id in FIT_SETS:
        return "fit"
    if ds_id in VALIDATION_SETS:
        return "hold-out"
    return "check-only (в подгонку не входит)"


def save(path, header, rows, sheet):
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = sheet[:31]; ws.append(header)
    for r in rows:
        ws.append([v if isinstance(v, (str, int, float, type(None))) else json.dumps(v, ensure_ascii=False) for v in r])
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = min(60, max(12, max(len(str(c.value or "")) for c in col) // 2))
    wb.save(path)


def main():
    OUT.mkdir(exist_ok=True)
    d = json.loads(DATA.read_text(encoding="utf-8"))
    header = ["id", "публикация", "DOI", "рисунок/панель", "препарат", "возраст", "тип синапса/источника",
              "метод измерения", "n (по авторам)", "х", "у", "наблюдаемая (что измерено)", "модельная наблюдаемая",
              "неопределённость (правило)", "оцифровка", "результат авторов", "число точек", "роль"]
    rows = []
    for ds in d["datasets"]:
        kind = ds.get("source_kind")
        if str(ds.get("type")) == "equation_fitted_to_data":
            model_obs = "средняя по щели концентрация C(t) (проверка, не подгонка)"
        elif ds["id"].startswith("clements"):
            model_obs = "c(t) в щели -> схема NMDA-R с конкурентным антагонистом -> нормированный EPSC (note в данных)"
        else:
            model_obs = MODEL_OBS.get(kind, UNKNOWN)
        pts = ds.get("points") or ds.get("x") or []
        rows.append([ds["id"], ds["source"], doi_for(ds["source"]), ds.get("figure", UNKNOWN), ds.get("preparation", UNKNOWN),
                     UNKNOWN, kind or ds.get("type_ru", UNKNOWN), ds.get("type_ru", UNKNOWN), ds.get("n", UNKNOWN),
                     ds.get("x_name", ""), ds.get("y_name", ""), ds.get("what", ""), model_obs,
                     ds.get("sigma_rule", UNKNOWN), ds.get("digitization", UNKNOWN), ds.get("authors_result", ""),
                     len(pts), role(ds["id"])])
    for s in d.get("scalar_checks", []):
        rows.append([s["id"], s["source"], doi_for(s["source"]), "-", s.get("conditions", UNKNOWN), UNKNOWN, "скаляр",
                     s.get("type", ""), UNKNOWN, "", s.get("unit", ""), s.get("what", ""),
                     "время очистки / спад сигнала в модели (определение времени сверять со статьёй)",
                     f"σ = {s['sigma']} {s.get('unit', '')}" if "sigma" in s else UNKNOWN, "число из статьи",
                     str(s.get("value", "")), 1, "check-only (скалярная проверка)"])
    save(OUT / "validation_sources.xlsx", header, rows, "validation_sources")

    npts = {ds["id"]: len(ds.get("points") or ds.get("x") or []) for ds in d["datasets"]}
    h2 = ["вариант модели", "набор", "роль", "точек", "χ²", "χ²/точку", "χ² всей подгонки", "AIC", "свободных параметров"]
    r2 = []
    for label, p in REPORTS.items():
        rep = json.loads(Path(p).read_text(encoding="utf-8"))
        for i, c in rep["chi2_by_dataset"].items():
            r2.append([label, i, "fit", npts.get(i), c, c / npts[i] if npts.get(i) else None, rep["chi2_total"], rep["aic"], rep["n_free"]])
        for i, c in rep["validation_chi2_not_in_fit"].items():
            r2.append([label, i, "hold-out", npts.get(i), c, c / npts[i] if npts.get(i) else None, rep["chi2_total"], rep["aic"], rep["n_free"]])
    save(OUT / "fit_vs_holdout.xlsx", h2, r2, "fit_vs_holdout")
    print(f"validation_sources.xlsx: {len(rows)} строк; fit_vs_holdout.xlsx: {len(r2)} строк")
    for r in r2:
        print(f"{r[0]:26s} {r[1]:32s} {r[2]:9s} n={r[3]}  chi2={r[4]:.1f}  chi2/n={r[5]:.2f}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
