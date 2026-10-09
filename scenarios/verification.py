# -*- coding: utf-8 -*-
"""
scenarios/verification.py - Верификация модели на экспериментальных данных.

ЧТО ДЕЛАЕТ
----------
1) Моделирует ОДИН выброс глутамата (N молекул) в режиме
   normalization_mode = "concentration" - то есть в абсолютных
   величинах, а не в вероятности одной частицы.
2) Переводит c(r,t) в мкМ (utils/units.py, учитывая долю
   внеклеточного объёма α).
3) Считает величины, которые можно прямо сравнить с литературой:
     - средняя концентрация в синапсе (r < a): пик и время спада в e раз
       -> сравнение с Clements et al., 1992 (~1 мМ, τ ~1 мс);
     - пик концентрации на разных расстояниях и длина спада λ
       -> сравнение с Matthews et al., 2022 (iGluSnFR: λ ≈ 1.2 мкм;
          NMDA-R при uncaging: λ ≈ 1.5 мкм);
     - до какого радиуса и сколько времени концентрация выше порога
       активации внесинаптических NMDA-R (~0.25 мкМ, Herman et al., 2011).
4) Считает loss - сумму квадратов логарифмов отношений модель/данные
   (метод наименьших квадратов в относительной шкале, потому что
   величины разных единиц и масштабов).
5) По флагу --scan перебирает D_pm и показывает, при каком D_pm loss
   минимален - это заготовка оптимизационной задачи, о которой говорил
   руководитель.

ЗАПУСК (из корня проекта, папка diplom)
---------------------------------------
    python -m scenarios.verification              # литературные D, отбор EAAT, без дрейфа
    python -m scenarios.verification --compare    # + те же метрики для D из курсовой
    python -m scenarios.verification --scan       # перебор D_pm + таблица loss
    python -m scenarios.verification --no-uptake  # выключить отбор (чистая диффузия)
    python -m scenarios.verification --omega      # включить дрейф Omega из config.py

Результаты: папка verification_outputs/ (графики PNG + summary.json).

ЧТО МОЖНО МЕНЯТЬ
----------------
- LITERATURE_PARAMS ниже: D, число молекул, α, Km, Vmax.
- TARGETS ниже: целевые числа из статей (и их веса в loss).
- Сетку (MAX_R, DR, DT, T_MAX): при больших D шаг dr=2 нм достаточен.

ГДЕ МОЖНО ОШИБИТЬСЯ
-------------------
- Сферическая геометрия "размазывает" щель (тонкий диск ~20 нм) по шару
  радиуса a. Поэтому средняя концентрация в синапсе и время её спада -
  оценка порядка, а не точное сравнение с Clements.
- Сам τ ~1 мс у Clements 1992 позже оспаривался (Diamond & Jahr 1997
  показывают более быструю очистку). Вес этой цели в loss поэтому ниже.
- λ из iGluSnFR завышен сенсором (он связывает глутамат и медленно
  отпускает). Это скорее верхняя граница.
- В 3D пик концентрации падает по степенному закону (~1/r³), а не по
  экспоненте. λ здесь - это длина экспоненты, подогнанной на отрезке
  400-2000 нм, ровно так же, как её считают в эксперименте.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config


from scenarios.supervisor_revision import temporary_global_settings
from solver.fp_solver import solve_fp_equation
from solver.geometry import radial_cell_measures
from utils.units import solver_to_uM_factor, uptake_to_solver_units

OUT_DIR = Path("verification_outputs")

# ---------------------------------------------------------------------------
# 1. Параметры из литературы (все числа - со ссылками в config.py и в xlsx)
# ---------------------------------------------------------------------------
LITERATURE_PARAMS = {
    # Zheng, Scimemi, Rusakov 2008: D в щели ~0.33 мкм²/мс = 330 нм²/мкс;
    # кажущийся межклеточный D ~0.32 мкм²/мс = 320 нм²/мкс.
    "D_cleft": 330.0,
    "D_pm": 320.0,
    "molecules": config.molecules_per_release,
    "alpha": config.extracellular_volume_fraction,
    "Km_uM": config.Km_uM,
    "Vmax_cleft_uM_per_us": config.Vmax_cleft_uM_per_us,
    "Vmax_pm_uM_per_us": config.Vmax_pm_uM_per_us,
}

# ---------------------------------------------------------------------------
# 2. Целевые значения (то, с чем сравниваем) и их веса в loss
# ---------------------------------------------------------------------------
TARGETS = {
    "cleft_peak_uM": {"value": 1000.0, "weight": 1.0,
                      "source": "Clements et al. 1992: пик в щели ~1 мМ"},
    "cleft_decay_us": {"value": 1000.0, "weight": 0.5,
                       "source": "Clements et al. 1992: τ ~1 мс (оспаривается, вес 0.5)"},
    "lambda_um": {"value": 1.2, "weight": 1.0,
                  "source": "Matthews et al. 2022: iGluSnFR, λ = 1.2 ± 0.05 мкм"},
}

# ---------------------------------------------------------------------------
# 3. Сетка расчёта
# ---------------------------------------------------------------------------
MAX_R = 3000.0     # нм - нужно с запасом больше 2 мкм, чтобы посчитать λ
DR = 2.0           # нм
DT = 0.5           # мкс (схема неявная, устойчива при любом dt; dt влияет на точность)
T_MAX = 5000.0     # мкс = 5 мс
RELEASE_SIGMA_NM = 20.0   # ширина источника: выброс почти точечный
PROBE_RADII = [200.0, 400.0, 500.0, 1000.0, 1500.0, 2000.0]
LAMBDA_FIT_RANGE = (400.0, 2000.0)


def run_single_release(D_cleft, D_pm, Omega_cleft=0.0, Omega_pm=0.0,
                       uptake=True, params=None, t_max=T_MAX, dt=DT):
    """Один выброс N молекул в момент t=0. Возвращает r, t, c_uM (n_r x n_t)."""
    params = dict(LITERATURE_PARAMS if params is None else params)
    alpha = params["alpha"]
    geometry = "spherical"
    Km_s, Vc_s, Vp_s = uptake_to_solver_units(
        params["Km_uM"], params["Vmax_cleft_uM_per_us"], params["Vmax_pm_uM_per_us"],
        alpha, geometry,
    )
    with temporary_global_settings(
        geometry_mode=geometry,
        realtime_modeling=False,
        pulse_center_nm=0.0,
        pulse_sigma_nm=RELEASE_SIGMA_NM,
        pulse_max_frequency_hz=1e9,
        max_saved_frames=4000,
    ):
        p_hist, r, t, D_values, _, pulse_info = solve_fp_equation(
            MAX_R, dt, DR, t_max,
            D_cleft, D_pm, config.xi_s, config.a, config.b,
            Omega_cleft, Omega_pm,
            params["molecules"],              # pulse_amount = число молекул
            (0.0, 5.0 * RELEASE_SIGMA_NM),    # окно источника
            1, None, (0.0, 0.0),              # один импульс в t=0
            "count", "schedule", 0.0, 0.0, 0.0, False,
            config.D_transition_kind, config.D_transition_steepness,
            config.Omega_transition_kind, config.Omega_transition_steepness,
            outer_boundary_mode="dirichlet",
            normalization_mode="concentration",
            uptake_kind="saturable" if uptake else "none",
            Vmax_cleft=Vc_s, Vmax_pm=Vp_s, Km=Km_s,
        )
    if pulse_info["count"] != 1:
        raise RuntimeError(f"Ожидался 1 выброс, получено {pulse_info['count']}")
    factor = solver_to_uM_factor(alpha, geometry)
    dV = radial_cell_measures(r, DR, geometry)
    molecules_left = (p_hist * dV[:, None]).sum(axis=0)   # сколько молекул осталось в области
    return {"r": r, "t": t, "c_uM": p_hist * factor, "dV": dV,
            "molecules_left": molecules_left, "D_values": D_values}


def compute_metrics(sim):
    """Метрики, которые сравниваются с экспериментом."""
    r, t, c, dV = sim["r"], sim["t"], sim["c_uM"], sim["dV"]
    m = {}

    # --- (а) Средняя концентрация в синапсе r < a (сравнение с Clements) ---
    inside = r < config.a
    cleft_mean = (c[inside] * dV[inside, None]).sum(axis=0) / dV[inside].sum()
    i_peak = int(np.argmax(cleft_mean))
    m["cleft_peak_uM"] = float(cleft_mean[i_peak])
    below = np.where(cleft_mean[i_peak:] < cleft_mean[i_peak] / math.e)[0]
    m["cleft_decay_us"] = float(t[i_peak + below[0]] - t[i_peak]) if below.size else float("nan")

    # --- (б) Пик концентрации на разных расстояниях ---
    peak_c = c.max(axis=1)
    peak_t = t[np.argmax(c, axis=1)]
    m["probes"] = []
    thr = config.extrasynaptic_nmda_threshold_uM
    dt_saved = np.diff(t, prepend=t[0])
    for rp in PROBE_RADII:
        k = int(np.argmin(np.abs(r - rp)))
        m["probes"].append({
            "r_nm": float(r[k]),
            "peak_uM": float(peak_c[k]),
            "t_peak_us": float(peak_t[k]),
            "time_above_threshold_us": float(dt_saved[c[k] >= thr].sum()),
        })

    # --- (в) Длина спада λ: экспонента на отрезке LAMBDA_FIT_RANGE ---
    sel = (r >= LAMBDA_FIT_RANGE[0]) & (r <= LAMBDA_FIT_RANGE[1]) & (peak_c > 0)
    if sel.sum() > 3:
        slope = np.polyfit(r[sel], np.log(peak_c[sel]), 1)[0]
        m["lambda_um"] = float(-1.0 / slope / 1000.0) if slope < 0 else float("inf")
    else:
        m["lambda_um"] = float("nan")

    # --- (г) Радиус, до которого пик выше порога NMDA-R ---
    above = np.where(peak_c >= thr)[0]
    m["r_above_threshold_nm"] = float(r[above[-1]]) if above.size else 0.0

    m["molecules_left_end"] = float(sim["molecules_left"][-1])
    return m


def loss(metrics, targets=TARGETS):
    """Сумма w * (ln(модель/данные))². 0 = полное совпадение."""
    total = 0.0
    for key, tgt in targets.items():
        val = metrics.get(key)
        if val is None or not np.isfinite(val) or val <= 0:
            return float("inf")
        total += tgt["weight"] * math.log(val / tgt["value"]) ** 2
    return total


def print_report(title, metrics):
    print("")
    print("=" * 78)
    print(title)
    print("=" * 78)
    for key, tgt in TARGETS.items():
        val = metrics[key]
        ratio = val / tgt["value"] if np.isfinite(val) else float("nan")
        print(f"  {key:16s} модель = {val:10.3f}   данные = {tgt['value']:8.3f}   "
              f"модель/данные = {ratio:7.3f}   [{tgt['source']}]")
    print(f"  loss = {loss(metrics):.3f}")
    print(f"  Порог NMDA-R {config.extrasynaptic_nmda_threshold_uM} мкМ превышен до r = "
          f"{metrics['r_above_threshold_nm']:.0f} нм")
    print("  r, нм   пик, мкМ   t пика, мкс   время выше порога, мкс")
    for p in metrics["probes"]:
        print(f"  {p['r_nm']:6.0f}  {p['peak_uM']:9.3f}  {p['t_peak_us']:11.1f}  "
              f"{p['time_above_threshold_us']:12.1f}")
    print(f"  Молекул осталось в области к концу: {metrics['molecules_left_end']:.0f} "
          f"из {LITERATURE_PARAMS['molecules']:.0f}")


def plot_results(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, sim in runs.items():
        r, t, c, dV = sim["r"], sim["t"], sim["c_uM"], sim["dV"]
        inside = r < config.a
        cleft_mean = (c[inside] * dV[inside, None]).sum(axis=0) / dV[inside].sum()
        axes[0].semilogy(t / 1000.0, np.maximum(cleft_mean, 1e-6), label=label)
        axes[1].semilogy(r / 1000.0, np.maximum(c.max(axis=1), 1e-6), label=label)
    # Clements: 1 мМ * exp(-t/1 мс) - ориентир
    tt = np.linspace(0, T_MAX / 1000.0, 200)
    axes[0].semilogy(tt, 1000.0 * np.exp(-tt / 1.0), "k--", label="Clements 1992: 1 мМ, τ=1 мс")
    axes[0].set_xlabel("t, мс"); axes[0].set_ylabel("средняя C в синапсе (r < a), мкМ")
    axes[0].set_title("Очистка синапса"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    rr = np.linspace(0.4, 2.0, 50)
    axes[1].semilogy(rr, 30.0 * np.exp(-(rr - 0.4) / 1.2), "k--", label="наклон λ = 1.2 мкм (Matthews 2022)")
    axes[1].axhline(config.extrasynaptic_nmda_threshold_uM, color="r", ls=":",
                    label=f"порог NMDA-R ~{config.extrasynaptic_nmda_threshold_uM} мкМ")
    axes[1].set_xlabel("r, мкм"); axes[1].set_ylabel("пик C(r), мкМ")
    axes[1].set_title("Спилловер: пик концентрации от расстояния")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Верификация модели на литературных данных")
    ap.add_argument("--compare", action="store_true", help="добавить расчёт с D из курсовой")
    ap.add_argument("--scan", action="store_true", help="перебор D_pm и таблица loss")
    ap.add_argument("--no-uptake", action="store_true", help="без отбора транспортёрами")
    ap.add_argument("--omega", action="store_true", help="включить дрейф Omega из config.py")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    uptake = not args.no_uptake
    om_c = config.Omega_cleft if args.omega else 0.0
    om_p = config.Omega_pm if args.omega else 0.0
    summary = {"params": LITERATURE_PARAMS, "targets": TARGETS, "uptake": uptake,
               "omega": [om_c, om_p], "runs": {}}
    runs = {}

    p = LITERATURE_PARAMS
    sim = run_single_release(p["D_cleft"], p["D_pm"], om_c, om_p, uptake)
    met = compute_metrics(sim)
    print_report(f"Литературные D: D_cleft={p['D_cleft']}, D_pm={p['D_pm']} нм²/мкс", met)
    runs[f"литература D={p['D_cleft']:.0f}/{p['D_pm']:.0f}"] = sim
    summary["runs"]["literature"] = {**met, "loss": loss(met)}

    if args.compare:
        sim2 = run_single_release(config.D_cleft, config.D_pm, om_c, om_p, uptake)
        met2 = compute_metrics(sim2)
        print_report(f"D из курсовой: D_cleft={config.D_cleft}, D_pm={config.D_pm} нм²/мкс", met2)
        runs[f"курсовая D={config.D_cleft:g}/{config.D_pm:g}"] = sim2
        summary["runs"]["coursework"] = {**met2, "loss": loss(met2)}

    if args.scan:
        # Сетка по двум параметрам: D_cleft (отвечает за очистку синапса)
        # и D_pm (отвечает за выход наружу). Это простейшая оптимизация
        # перебором; потом её можно заменить на scipy.optimize.minimize
        # с той же функцией loss.
        print("\nПеребор D_cleft x D_pm (нм²/мкс), метрика loss:")
        print("  D_cleft    D_pm     λ, мкм   пик в синапсе, мкМ   спад, мкс    loss")
        scan = []
        for d_cl in [10.0, 30.0, 100.0, 330.0]:
            for d_pm in [1.0, 10.0, 100.0, 320.0]:
                m = compute_metrics(run_single_release(d_cl, d_pm, om_c, om_p, uptake))
                L = loss(m)
                scan.append({"D_cleft": d_cl, "D_pm": d_pm, "loss": L, "lambda_um": m["lambda_um"],
                             "cleft_peak_uM": m["cleft_peak_uM"], "cleft_decay_us": m["cleft_decay_us"]})
                print(f"  {d_cl:7.0f}  {d_pm:6.0f}  {m['lambda_um']:8.3f}  {m['cleft_peak_uM']:18.1f}  "
                      f"{m['cleft_decay_us']:10.1f}  {L:7.3f}")
        best = min(scan, key=lambda s: s["loss"])
        print(f"  Минимум loss: D_cleft = {best['D_cleft']}, D_pm = {best['D_pm']} "
              f"(loss = {best['loss']:.3f}). Если минимум на краю сетки - расширить сетку.")
        summary["scan"] = scan

    plot_results(runs, OUT_DIR / "verification.png")
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nГотово: {(OUT_DIR / 'verification.png').resolve()}")


if __name__ == "__main__":
    main()
