# -*- coding: utf-8 -*-
"""
experiments/run_stage02.py - сдача этапов 0-2 ТЗ от 13.10.2026.

Считает три случая с полностью заданными параметрами (experiments/configs.py):
  (а) baseline_physical  - геометрия Barbour, без захвата и дрейфа;
  (б) with_uptake        - геометрия Barbour, насыщаемый захват, без дрейфа;
  (в) legacy_coursework  - исторический сценарий курсовой (для сравнения).
Для каждого сохраняет в experiments_outputs/stage02/<имя>/:
  parameters.json, mass_budget.xlsx, concentration_probes.xlsx, numerical_qc.json, analysis_summary.json
и общие: manifest.json, legacy_manifest.json (запрошенные и фактические импульсы),
geometry_validation.png, mass_budget.png, profiles.png, checks.json (таблица ожидалось/получено/PASS).

Запуск из папки diplom:  python -m experiments.run_stage02
CSV не используется: таблицы - XLSX и JSON.
"""
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import config
config.use_gpu = False
config.realtime_modeling = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl
import scipy

from experiments import configs as C
from solver.fp_solver import solve_fp_equation
from solver.geometry import barbour_area, barbour_volume, radial_cell_measures
from physics.pulses import compute_pulse_schedule
from utils.units import solver_to_uM_factor, uptake_to_solver_units
from analysis.inversion_concentration import concentration_inversion

OUT = Path("experiments_outputs/stage02")
# значения config.py курсовой на момент импорта (run_case их временно меняет)
CONFIG_DEFAULTS = {k: getattr(config, k) for k in ("pulse_max_frequency_hz", "num_pulses", "pulse_segments",
                                                    "pulse_time_window", "pulse_segment_default_mode", "t_max", "dt")}
PROBES_NM = [0.0, 50.0, 100.0, 150.0, 250.0, 400.0, 600.0, 1000.0, 2000.0]


def _set(**kw):
    for k, v in kw.items():
        setattr(config, k, v)


def run_case(p, dr=None, dt=None, R=None, T=None, frames=1500):
    """Один выброс по конфигурации p. Возвращает словарь с c (мкМ), r, t, бюджетом и диагностикой."""
    dr = p["dr_nm"] if dr is None else dr
    dt = p["dt_us"] if dt is None else dt
    R = p["R_nm"] if R is None else R
    T = p["t_max_us"] if T is None else T
    _set(geometry_mode=p["geometry"], geometry_a_nm=p["a_nm"], geometry_b_nm=p["b_nm"],
         cleft_radius_nm=p["a_nm"], cleft_height_nm=p["h_nm"], extracellular_volume_fraction=p["alpha"],
         pulse_center_nm=0.0, pulse_sigma_nm=p["sigma_nm"], pulse_max_frequency_hz=1e9,
         max_saved_frames=frames, strict_pulse_count=True)
    Km_s, Vc_s, Vp_s = uptake_to_solver_units(p["Km"], p["Vmax_cleft"], p["Vmax_pm"], p["alpha"], p["geometry"], p["h_nm"])
    t0 = time.time()
    c, r, t, D, Om, info = solve_fp_equation(
        R, dt, dr, T, p["D_cleft"], p["D_pm"], config.xi_s, p["a_nm"], p["b_nm"], p["Omega_cleft"], p["Omega_pm"],
        p["N"], (0.0, p["window_nm"]), 1, None, (0.0, 0.0), "count", "schedule", 0.0, 0.0, 0.0, False,
        config.D_transition_kind, config.D_transition_steepness, config.Omega_transition_kind, config.Omega_transition_steepness,
        outer_boundary_mode=p["boundary"], normalization_mode="concentration",
        uptake_kind=p["uptake"], Vmax_cleft=Vc_s, Vmax_pm=Vp_s, Km=Km_s,
        uptake_scheme=p["uptake_scheme"], clip_negative=p["clip_negative"], track_mass=True)
    f = solver_to_uM_factor(p["alpha"], p["geometry"], p["h_nm"])
    dV = radial_cell_measures(r, dr, p["geometry"])
    if p["geometry"] == "spherical":
        # старый режим: мера без 4π·α, но импульс нормирован на ту же меру -> масса решателя = число молекул;
        # объём жидкости ячейки = мера·4π·α (нужен только как вес при усреднении по щели)
        dV = dV * 4.0 * math.pi * p["alpha"]
    mb = info["mass_budget"]
    return {"c": c * f, "r": r, "t": t, "dV": dV, "mb": mb, "info": info, "seconds": time.time() - t0,
            "grid": {"dr_nm": dr, "dt_us": dt, "R_nm": R, "t_max_us": T}}


def probes(res, radii=PROBES_NM):
    return np.array([[np.interp(x, res["r"], res["c"][:, k]) for x in radii] for k in range(res["t"].size)])


def cleft_mean(res, a):
    m = res["r"] < a
    return (res["c"][m] * res["dV"][m, None]).sum(0) / res["dV"][m].sum()


def save_xlsx(path, sheets):
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    for name, (header, rows) in sheets.items():
        ws = wb.create_sheet(name[:31]); ws.append(header)
        for row in rows:
            ws.append([float(x) if isinstance(x, (np.floating, float, int, np.integer)) else x for x in row])
    wb.save(path)


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def legacy_pulse_table():
    """Сколько импульсов запрашивает config.py курсовой и сколько реально ставит расписание."""
    d = CONFIG_DEFAULTS
    t_values = np.arange(0.0, d["t_max"] + 0.5 * d["dt"], d["dt"])
    rows = []
    saved = config.pulse_max_frequency_hz
    try:
        for fmax in (d["pulse_max_frequency_hz"], 1e9):
            config.pulse_max_frequency_hz = fmax
            idx, _ = compute_pulse_schedule(t_values, d["num_pulses"], d["pulse_segments"], d["pulse_time_window"],
                                            segment_default_mode=d["pulse_segment_default_mode"])
            rows.append({"requested": int(d["num_pulses"]), "window_us": list(d["pulse_time_window"]),
                         "max_frequency_hz": fmax, "actual": int(len(idx)),
                         "first_times_us": [float(x) for x in t_values[idx][:10]],
                         "note": "предел частоты из config.py" if fmax == d["pulse_max_frequency_hz"] else "без предела частоты"})
    finally:
        config.pulse_max_frequency_hz = saved
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [C.get("baseline_physical"), C.get("with_uptake"), C.get("legacy_coursework")]
    results, checks, summary = {}, [], {}

    def check(name, expected, got, ok):
        checks.append({"проверка": name, "ожидалось": expected, "получено": got, "статус": "PASS" if ok else "FAIL"})

    for p in cases:
        d = OUT / p["name"]; d.mkdir(exist_ok=True)
        res = run_case(p); results[p["name"]] = res
        mb = res["mb"]
        rel_bal = float(np.max(np.abs(mb["balance_error"])) / mb["released"][-1])
        # сходимость по сетке и чувствительность к границе - на первых 1-2 мс
        Tq = min(p["t_max_us"], 2000.0)
        base = run_case(p, T=Tq, frames=200)
        fine = run_case(p, dr=p["dr_nm"] / 2, dt=p["dt_us"] / 2, T=Tq, frames=200)
        farR = run_case(p, R=p["R_nm"] * 1.5, T=Tq, frames=200)
        pr_b, pr_f, pr_R = probes(base)[-1], probes(fine)[-1], probes(farR)[-1]
        near = np.array(PROBES_NM) <= min(2000.0, 0.5 * p["R_nm"])
        grid_err = float(np.max(np.abs(pr_b[near] - pr_f[near]) / np.max(pr_f[near])))
        R_err = float(np.max(np.abs(pr_b[near] - pr_R[near]) / np.max(pr_R[near])))
        sc = res["info"]["solver_convergence"]
        qc = {"balance_rel_error_max": rel_bal, "min_c_preclip_solver_units": float(mb["min_c_preclip"]),
              "negative_values_after_step": int(np.sum(res["c"] < 0)),
              "grid_convergence_rel_diff_dr_dt_halved": grid_err, "R_sensitivity_rel_diff_R_x1.5": R_err,
              "solver_failed_steps": int(sc["failed_steps"]), "solver_checked_steps": int(sc["checked_steps"]),
              "uptake_scheme": p["uptake_scheme"], "clip_negative": p["clip_negative"],
              "pulses": int(res["info"]["count"]), "seconds": res["seconds"]}
        inv = concentration_inversion(res["c"], res["r"], res["t"], abs_threshold_uM=1e-3)
        cm = cleft_mean(res, p["a_nm"])
        k0 = int(np.argmax(cm)); below = np.nonzero(cm[k0:] < cm[k0] / math.e)[0]
        an = {"inversion": inv, "cleft_mean_peak_uM": float(cm[k0]),
              "cleft_mean_decay_1_over_e_us": float(res["t"][k0 + below[0]] - res["t"][k0]) if below.size else None,
              "fraction_uptaken": float(mb["uptaken"][-1] / mb["released"][-1]),
              "fraction_escaped": float(mb["escaped"][-1] / mb["released"][-1]),
              "fraction_free_end": float(mb["free"][-1] / mb["released"][-1])}
        summary[p["name"]] = {"qc": qc, "analysis": an}
        (d / "parameters.json").write_text(json.dumps({**p, "grid": res["grid"]}, ensure_ascii=False, indent=2), encoding="utf-8")
        (d / "numerical_qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
        (d / "analysis_summary.json").write_text(json.dumps(an, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        save_xlsx(d / "mass_budget.xlsx", {"баланс": (
            ["t, мкс", "выброшено", "свободно", "поглощено", "ушло за границу", "ошибка баланса"],
            zip(mb["t"], mb["released"], mb["free"], mb["uptaken"], mb["escaped"], mb["balance_error"]))})
        pr = probes(res)
        save_xlsx(d / "concentration_probes.xlsx", {"C(r,t), мкМ": (
            ["t, мкс"] + [f"r = {x:g} нм" for x in PROBES_NM] + ["среднее по щели r < a"],
            ([res["t"][k]] + list(pr[k]) + [cm[k]] for k in range(res["t"].size)))})
        tag = p["name"]
        check(f"{tag}: ровно один выброс", "1", str(qc["pulses"]), qc["pulses"] == 1)
        check(f"{tag}: выброшено молекул", "3000", f"{mb['released'][-1]:.6f}", abs(mb["released"][-1] - p["N"]) < 1e-6 * p["N"])
        check(f"{tag}: ошибка баланса массы", "< 1e-6", f"{rel_bal:.1e}", rel_bal < 1e-6)
        if p["name"] != "legacy_coursework":
            check(f"{tag}: отрицательные значения до обрезки", "нет", f"min = {mb['min_c_preclip']:.1e}", mb["min_c_preclip"] >= 0.0)
            check(f"{tag}: сходимость по сетке (dr, dt пополам)", "< 2%", f"{grid_err:.2%}", grid_err < 0.02)
            check(f"{tag}: влияние границы R (R x1,5)", "< 1%", f"{R_err:.2%}", R_err < 0.01)
            check(f"{tag}: инверсия концентрации", "нет (раздел 3 ТЗ)", "есть" if inv["found"] else "нет", not inv["found"])
        else:
            check(f"{tag}: инверсия концентрации (дрейф наружу)", "возможна", "есть" if inv["found"] else "нет", True)
    # исторический: запрошено vs фактически импульсов
    pulses = legacy_pulse_table()
    check("legacy: импульсов в config.py", "запрошено 100", f"фактически {pulses[0]['actual']}", True)
    (OUT / "legacy_manifest.json").write_text(json.dumps({
        "config": C.get("legacy_coursework"), "pulse_schedule_requested_vs_actual": pulses,
        "frozen_commit_main": "c05e8866fa6b89a661cd50e09ed6846140ccfa95"}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    (OUT / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({
        "git_sha": git_sha(), "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "processor": platform.processor(), "backend": "CPU (numpy/scipy)",
        "command": "python -m experiments.run_stage02", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cases": [p["name"] for p in cases]}, ensure_ascii=False, indent=2), encoding="utf-8")
    save_xlsx(OUT / "checks.xlsx", {"проверки": (["проверка", "ожидалось", "получено", "статус"],
                                                 ([c["проверка"], c["ожидалось"], c["получено"], c["статус"]] for c in checks))})
    plots(results, cases)
    for c in checks:
        print(f"{c['статус']:4s} | {c['проверка']} | ожидалось {c['ожидалось']} | получено {c['получено']}")


def plots(results, cases):
    # геометрия
    p = cases[0]
    _set(geometry_a_nm=p["a_nm"], geometry_b_nm=p["b_nm"], cleft_height_nm=p["h_nm"], extracellular_volume_fraction=p["alpha"])
    r = np.linspace(1, 800, 2000)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
    ax[0].plot(r, barbour_volume(r), "k", lw=2, label="V(r), Barbour")
    ax[0].plot(r, math.pi * p["h_nm"] * r**2, "--", label="щель πhr²")
    ax[0].plot(r, p["alpha"] * 4 / 3 * math.pi * r**3, ":", label="нейропиль α·4πr³/3")
    ax[0].set_yscale("log"); ax[0].set_xlabel("r, нм"); ax[0].set_ylabel("нм³"); ax[0].legend(fontsize=8)
    for x in (p["a_nm"], p["b_nm"]):
        ax[0].axvline(x, color="gray", lw=0.8); ax[1].axvline(x, color="gray", lw=0.8)
    ax[1].plot(r, barbour_area(r), "k", lw=2, label="A(r) = dV/dr")
    ax[1].plot(r, 2 * math.pi * p["h_nm"] * r, "--", label="2πhr")
    ax[1].plot(r, 4 * math.pi * p["alpha"] * r**2, ":", label="4παr²")
    ax[1].set_yscale("log"); ax[1].set_xlabel("r, нм"); ax[1].set_ylabel("нм²"); ax[1].legend(fontsize=8)
    fig.suptitle("Геометрия Barbour 2001: a = 200, b = 400 нм, h = 20 нм, α = 0,2"); fig.tight_layout()
    fig.savefig(OUT / "geometry_validation.png"); plt.close(fig)
    # баланс массы
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=130)
    for ax, p in zip(axes, cases):
        mb = results[p["name"]]["mb"]; t = mb["t"] / 1000
        for key, lab in (("free", "свободно"), ("uptaken", "поглощено"), ("escaped", "ушло за границу"), ("released", "выброшено")):
            ax.plot(t, mb[key], label=lab)
        ax.set_title(p["label"], fontsize=9); ax.set_xlabel("t, мс"); ax.set_ylabel("молекул"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "mass_budget.png"); plt.close(fig)
    # профили c(r) в разные моменты
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=130)
    for ax, p in zip(axes, cases):
        res = results[p["name"]]
        for tt in (10, 100, 500, 1000, 3000):
            if tt <= res["t"][-1]:
                k = int(np.argmin(np.abs(res["t"] - tt)))
                ax.plot(res["r"], np.maximum(res["c"][:, k], 1e-12), label=f"{tt / 1000:g} мс")
        ax.set_yscale("log"); ax.set_ylim(1e-4, None); ax.set_xlim(0, min(3000, res["r"][-1]))
        ax.set_title(p["label"], fontsize=9); ax.set_xlabel("r, нм"); ax.set_ylabel("c, мкМ"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "profiles.png"); plt.close(fig)


if __name__ == "__main__":
    main()
