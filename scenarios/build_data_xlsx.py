# -*- coding: utf-8 -*-
"""
scenarios/build_data_xlsx.py - собирает data/experimental_points.xlsx из
data/experimental_points.json и отчётов единой подгонки (unified_outputs/*_report.json).

Таблица - копия для чтения; источник данных для кода - только JSON.
Запуск из папки diplom:  python -m scenarios.build_data_xlsx
"""
import json
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

DATA = Path("data/experimental_points.json")
OUT = Path("data/experimental_points.xlsx")
REP = Path("unified_outputs")
VARIANTS = [  # (файл отчёта, подпись)
    ("original_report.json", "исходная модель (config.py)"),
    ("unified_report.json", "единая модель (основной результат)"),
    ("unified_nodrift_report.json", "единая, без дрейфа"),
    ("unified_cleftdrift_report.json", "единая, дрейф только в щели"),
    ("unified_literature_report.json", "единая, a и захват из литературы"),
    ("unified_D330_report.json", "единая, D_cleft = 330 (измеренный)"),
    ("unified_with_ampa_report.json", "единая, AMPA включён в подгонку"),
]
MAIN = "unified_report.json"

BLUE = Font(color="1F4E99")
BOLD = Font(bold=True)
HEAD = PatternFill("solid", fgColor="CED8E4")
WRAP = Alignment(wrap_text=True, vertical="top")


def load_reports():
    reps = {}
    for f, label in VARIANTS:
        p = REP / f
        if p.exists():
            reps[f] = (label, json.loads(p.read_text(encoding="utf-8")))
    return reps


def chi_cell(rep, i):
    """χ² набора: из подгонки, или из проверки (со звёздочкой), или пусто."""
    if i in rep["chi2_by_dataset"]:
        return round(rep["chi2_by_dataset"][i], 2)
    v = (rep.get("validation_chi2_not_in_fit") or {}).get(i)
    return f"{v:.1f} (проверка)" if v is not None else "-"


def sheet_name(ds_id):
    return ds_id.replace("matthews", "Matthews").replace("clements", "Clements")[:31]


def main():
    d = json.loads(DATA.read_text(encoding="utf-8"))
    reps = load_reports()
    main_pts = reps[MAIN][1]["model_points"] if MAIN in reps else {}
    orig_pts = reps["original_report.json"][1]["model_points"] if "original_report.json" in reps else {}
    wb = openpyxl.Workbook()

    # --- лист описания наборов
    ws = wb.active; ws.title = "Описание"
    ws["A1"] = "Экспериментальные точки для единой модели диффузии глутамата"; ws["A1"].font = Font(bold=True, size=13)
    ws["A2"] = ("Источник данных для кода: data/experimental_points.json. Эта таблица собрана скриптом "
                "scenarios/build_data_xlsx.py. Синий шрифт - оцифрованные данные, χ² считается формулами.")
    cols = ["Лист", "Тип данных", "Источник", "Рисунок, файл", "Что измерено", "Препарат и условия", "n",
            "Результат авторов", "Как задана погрешность σ", "Оцифровка", "В единой подгонке"]
    for j, c in enumerate(cols, 1):
        cell = ws.cell(4, j, c); cell.font = BOLD; cell.fill = HEAD; cell.alignment = WRAP
    fit_ids = set(reps[MAIN][1]["chi2_by_dataset"]) if MAIN in reps else set()
    for i, ds in enumerate(d["datasets"], 5):
        src = ds["source"] if isinstance(ds["source"], str) else json.dumps(ds["source"], ensure_ascii=False)
        row = [sheet_name(ds["id"]), ds.get("type_ru", ""), src, f'{ds.get("figure", "")}; {ds.get("file", "")}',
               ds.get("what", ""), ds.get("preparation", ""), ds.get("n", ""), ds.get("authors_result", ""),
               ds.get("sigma_rule", ""), ds.get("digitization", ""),
               "да" if ds["id"] in fit_ids else ("нет: проверка, модель не воспроизводит (docs/unified_model.md, п. 7.5)"
                                                 if "AMPA" in ds["id"] else "нет: оценка авторов через их модель, только проверка")]
        for j, v in enumerate(row, 1):
            ws.cell(i, j, str(v)).alignment = WRAP
    for j, w in enumerate([24, 14, 40, 24, 45, 45, 12, 35, 40, 40, 22], 1):
        ws.column_dimensions[get_column_letter(j)].width = w

    # --- листы с точками
    for ds in d["datasets"]:
        w = wb.create_sheet(sheet_name(ds["id"]))
        w["A1"] = f'{ds.get("figure", "")}: {ds.get("what", "")}'; w["A1"].font = BOLD
        w["A2"] = f'Тип: {ds.get("type_ru", "")}'
        hdr = [ds.get("x_name", "x"), ds.get("y_name", "y"), "σ", "единая модель", "(модель - данные)/σ", "вклад в χ²",
               "исходная модель", "вклад в χ²"]
        for j, c in enumerate(hdr, 1):
            cell = w.cell(4, j, c); cell.font = BOLD; cell.fill = HEAD; cell.alignment = WRAP
        pts = ds["points"]; m = main_pts.get(ds["id"]); o = orig_pts.get(ds["id"])
        for i, p in enumerate(pts, 5):
            for j, v in enumerate((p["x"], p["y"], p["sigma"]), 1):
                w.cell(i, j, v).font = BLUE
            if m is not None:
                w.cell(i, 4, round(m[i - 5], 5))
                w.cell(i, 5, f"=(D{i}-B{i})/C{i}"); w.cell(i, 6, f"=E{i}^2")
            if o is not None:
                w.cell(i, 7, round(o[i - 5], 5)); w.cell(i, 8, f"=((G{i}-B{i})/C{i})^2")
        last = 4 + len(pts)
        if m is not None:
            w.cell(last + 2, 5, "χ² ="); w.cell(last + 2, 6, f"=SUM(F5:F{last})").font = BOLD
        if o is not None:
            w.cell(last + 2, 8, f"=SUM(H5:H{last})").font = BOLD
        if m is None:
            w.cell(last + 2, 1, "Набор не входит в подгонку: это оценка авторов через их модель, используется только для проверки.")
        if ds["id"] not in fit_ids and m is not None:
            w.cell(last + 3, 1, "Набор не входит в χ² подгонки: проверка модели (см. docs/unified_model.md, п. 7.5).")
        for j, wd in enumerate([16, 18, 10, 16, 18, 12, 16, 12], 1):
            w.column_dimensions[get_column_letter(j)].width = wd

    # --- сравнение вариантов
    w = wb.create_sheet("Единая подгонка")
    w["A1"] = "Единая МНК-подгонка: один набор параметров для всех наборов (scenarios/unified_fit.py, 12.10.2026)"; w["A1"].font = BOLD
    ids = list(reps[MAIN][1]["chi2_by_dataset"]) if MAIN in reps else []
    ids += [i for i in ("matthews2022_fig4C_AMPA", "matthews2022_fig4D_AMPA_TBOA") if i not in ids]
    hdr = ["Вариант", "Свободных k", "Точек n", "χ²", "χ²/(n-k)", "AIC = χ²+2k"] + ids
    for j, c in enumerate(hdr, 1):
        cell = w.cell(3, j, c); cell.font = BOLD; cell.fill = HEAD; cell.alignment = WRAP
    r = 4
    for f, (label, rep) in reps.items():
        row = [label, rep["n_free"], rep["n_points"], round(rep["chi2_total"], 2), round(rep["chi2_per_dof"], 3),
               round(rep["aic"], 2)] + [chi_cell(rep, i) for i in ids]
        for j, v in enumerate(row, 1):
            w.cell(r, j, v)
        r += 1
    r += 1
    w.cell(r, 1, "Параметры").font = BOLD; r += 1
    keys = ["cleft_radius_nm", "D_cleft", "D_pm", "omega_scale", "omega_pm_ratio", "Vmax_pm", "Km", "molecules",
            "psf_xy_um", "psf_z_um", "uncaging_molecules", "alpha", "cleft_height_nm", "transition_width_nm"]
    w.cell(r, 1, "параметр").font = BOLD
    for j, (f, (label, rep)) in enumerate(reps.items(), 2):
        w.cell(r, j, label).font = BOLD
    r += 1
    for k in keys:
        w.cell(r, 1, k)
        for j, (f, (label, rep)) in enumerate(reps.items(), 2):
            v = rep["params"].get(k, 3.0 if k == "omega_pm_ratio" else None)
            if v is not None:
                txt = f"{v:.4g}" + (" (fit)" if k in rep.get("fit_keys", []) else "")
                w.cell(r, j, txt)
        r += 1
    r += 1
    w.cell(r, 1, "Проверки без подгонки").font = BOLD; r += 1
    for f, (label, rep) in reps.items():
        ch = rep.get("checks", {})
        w.cell(r, 1, label)
        w.cell(r, 2, f'время захвата, мс: {ch.get("uptake_tau_ms_vs_Diamond2005_0.75")} (Diamond 2005: 0,75 ± 0,15 при 35 °C)')
        cm = ch.get("cleft_mean_mM_at_0.5_1_2_4_ms")
        if cm:
            w.cell(r, 8, "щель, мМ при 0,5/1/2/4 мс: " + " / ".join(f"{x:.3g}" for x in cm) + " (Clements Fig. 3C: ~0,7 / 0,5 / 0,22 при 0,6/1/2,4 мс)")
        r += 1
    w.column_dimensions["A"].width = 34
    for j in range(2, len(hdr) + 1):
        w.column_dimensions[get_column_letter(j)].width = 16
    wb.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
