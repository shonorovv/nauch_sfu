# -*- coding: utf-8 -*-
"""
scenarios/unified_fit.py - ЕДИНАЯ модель, подогнанная сразу ко всем экспериментам.

Основа - исходная модель курсовой (main.py + config.py + solver/fp_solver.py):
    ∂c/∂t = (1/A)·∂/∂r [ A(r)·( D(r)·∂c/∂r − Ω(r)·c ) ] − Vmax(r)·c/(Km + c)
то же уравнение Фоккера-Планка, та же схема (неявная, конечные объёмы, upwind
для дрейфа), те же профили D(r) и Ω(r) с переходом между a и b.
Отличия от исходной модели перечислены в docs/unified_model.md и в PARAMS ниже.

Один набор параметров считается для ВСЕХ наборов данных сразу:
    Matthews 2022 Fig. 1F, 2A, 2D  - сигнал датчика от расстояния, синаптический выброс
    Matthews 2022 Fig. 3B          - то же для анкейджинга (известный источник)
    Clements 1992 Fig. 1D (D-AA, D-CPP) - подавление NMDA-тока антагонистами
Функция потерь: chi2 = Σ по всем точкам всех наборов ((модель − данные)/σ)².

Проверки без подгонки: Diamond 2005 (время очистки), Clements 1992 Fig. 3C,
Diamond & Jahr 1997 (двухфазная оценка), Helassa 2018.

Запуск из папки diplom:
    python -m scenarios.unified_fit                 # подгонка (долго: десятки минут)
    python -m scenarios.unified_fit --eval-only     # один расчёт с PARAMS без подгонки
    python -m scenarios.unified_fit --original      # исходные параметры курсовой на тех же данных
    python -m scenarios.unified_fit --no-drift      # подгонка с Ω = 0 (сравнение моделей)
"""
import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import numpy.linalg as lin
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
config.realtime_modeling = False
from solver.fp_solver import solve_fp_equation
from solver.geometry import radial_cell_measures
from utils.units import solver_to_uM_factor, uptake_to_solver_units
from scenarios.observation import epsc_ratio, sensor_peak, a2r_peak_per_node, ampa_open_peak_per_node

DATA_FILE = Path("data/experimental_points.json")
OUT_DIR = Path("unified_outputs")

# ---------------------------------------------------------------------------
# Параметры единой модели. "fit" - подбирается, иначе фиксировано (источник в комментарии)
# ---------------------------------------------------------------------------
PARAMS = {
    # геометрия
    "cleft_radius_nm": 150.0,   # fit, границы 80-250 нм (ЭМ: 0,04 мкм² -> 113 нм, Schikorski & Stevens 1997; 150 нм - Ventura & Harris 1999 через Lotter 2021)
    "cleft_height_nm": 20.0,    # ЭМ, ширина щели ~20 нм (цит. Clements 1992)
    "transition_width_nm": 200.0,  # b − a, как в исходной модели (a = 200, b = 400)
    "alpha": 0.2,               # Nicholson & Syková 1998
    # диффузия
    "D_cleft": 10.0,            # fit, границы 1-760 нм²/мкс; старт = значение курсовой (10). Измерения: 330 (Nielsen 2004), 760 - свободный раствор
    "D_pm": 320.0,              # измерено: 0,32 мкм²/мс в CA1 (Zheng et al. 2008)
    # дрейф исходной модели: Ω_cleft = 0,05·s, Ω_pm = 0,15·s нм/мкс
    "omega_scale": 1.0,         # fit (s = 1 - значения курсовой)
    "omega_pm_ratio": 3.0,      # Ω_pm / Ω_cleft, как в config.py (Omega_pm = 3·Omega_cleft); 0 - дрейф только в щели
    # захват транспортёрами (Михаэлис-Ментен)
    "Vmax_pm": 1e-4,            # fit, мкМ/мкс (в config 0,01 - слишком сильный захват для Matthews)
    "Vmax_cleft": 0.0,          # в щели транспортёров почти нет (Zheng 2008)
    "Km": 20.0,                 # мкМ (рабочая оценка, см. config.py)
    # источник
    "molecules": 3000.0,        # fit, границы 2000-10000 (3000 - Zheng 2008/Lotter 2021; 7000-8000 - Budisantoso 2013)
    "release_sigma_nm": 20.0,
    # наблюдение
    "psf_xy_um": 0.4,           # fit, границы 0,3-0,7 мкм (в статье не указано; типичная 2P-PSF 0,3-0,5)
    "psf_z_um": 2.0,            # fit, границы 1-3,5 мкм (в статье не указано)
    "sensor_Kd_uM": 4.9,        # Matthews 2022
    "sensor_tau_off_ms": 60.0,  # Matthews 2022: спад 52-71 мс
    # анкейджинг (Matthews 2022, Методы)
    "uncaging_molecules": 36000.0,  # fit, границы 1e4-1e6 (оценка авторов 36 000: 0,2 фл × 5 мМ × ε × α × N_A, ε и реальный объём неточны)
    "uncaging_sigma_nm": 230.0, # шар того же объёма, что 0,2 фл (FCS): (2π)^1.5·σ³ = 0,2 фл
}
FIT_KEYS = ["Vmax_pm", "cleft_radius_nm", "D_cleft", "molecules", "omega_scale", "psf_xy_um", "psf_z_um", "uncaging_molecules"]
LOG_KEYS = {"Vmax_pm", "D_cleft", "uncaging_molecules"}
BOUNDS = {"Vmax_pm": (1e-7, 1.0), "cleft_radius_nm": (80.0, 250.0), "D_cleft": (1.0, 760.0),
          "molecules": (2000.0, 10000.0), "psf_xy_um": (0.3, 0.7), "omega_scale": (-60.0, 60.0), "psf_z_um": (1.0, 3.5),
          "uncaging_molecules": (1.0e4, 1.0e6)}

ORIGINAL = {  # параметры курсовой (config.py) для сравнения на тех же данных
    "geometry": "spherical", "D_cleft": 10.0, "D_pm": 0.5, "omega_scale": 1.0, "omega_pm_ratio": 3.0,
    "Vmax_pm": 0.0, "cleft_radius_nm": 200.0, "transition_width_nm": 200.0,
    "release_sigma_nm": 500.0, "release_window_nm": 500.0,   # pulse_sigma_nm = 500, pulse_r_window = (0, 500)
}
# Что пришлось добавить к исходной модели, чтобы её вообще можно было сравнить с данными:
#   N = 3000 молекул (в config.py вероятность одной частицы, а данные зависят от абсолютной концентрации),
#   сетка до 10-20 мкм и 150 мс (в config.py max_r = 1 мкм, t_max = 1 мс, а данные измерены до 6 мкм и 150 мс),
#   те же модели наблюдения (датчик + PSF, NMDA-схема), PSF 0,4×2 мкм, N анкейджинга 36 000 (оценка авторов).

GRID_FINE = {"R": 10000.0, "dr": 2.0, "dt": 0.5, "T": 10000.0}     # щель и первые 10 мс
GRID_LONG = {"R": 20000.0, "dr": 5.0, "dt": 10.0, "T": 150000.0}   # датчик, до 150 мс (при подгонке dt 10 мкс, итоговый расчёт dt 2 мкс)
T_SPLIT_US = 10000.0


@contextlib.contextmanager
def temporary_global_settings(**kwargs):
    """Временно меняет атрибуты config (в т.ч. новые, которых в config.py нет) и возвращает как было."""
    missing = object()
    old = {k: getattr(config, k, missing) for k in kwargs}
    try:
        for k, v in kwargs.items():
            setattr(config, k, v)
        yield
    finally:
        for k, v in old.items():
            if v is missing:
                delattr(config, k)
            else:
                setattr(config, k, v)


def json_clean(x):
    """NaN и ±inf недопустимы в стандартном JSON: заменяем на null / строку."""
    if isinstance(x, dict):
        return {k: json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None if math.isnan(x) else ("inf" if x > 0 else "-inf")
    return x


def load_data():
    d = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    return {ds["id"]: ds for ds in d["datasets"]}, d.get("scalar_checks", [])


def arr(ds):
    p = ds["points"]
    return (np.array([q["x"] for q in p]), np.array([q["y"] for q in p]), np.array([q["sigma"] for q in p]))


# ---------------------------------------------------------------------------
# Решатель исходной модели с параметрами единой модели
# ---------------------------------------------------------------------------
def xi_for(a, b):
    """Коэффициенты полинома перехода, как в config.py."""
    V = np.array([[1, a, a**2, a**3, a**4, a**5], [1, b, b**2, b**3, b**4, b**5],
                  [0, 1, 2*a, 3*a**2, 4*a**3, 5*a**4], [0, 1, 2*b, 3*b**2, 4*b**3, 5*b**4],
                  [0, 0, 2, 6*a, 12*a**2, 20*a**3], [0, 0, 2, 6*b, 12*b**2, 20*b**3]], float)
    return lin.solve(V, np.array([0, 1, 0, 0, 0, 0], float))


def run(p, grid, geometry="cleft_disk", molecules=None, sigma=None, omega=True):
    a = p["cleft_radius_nm"]; b = a + p["transition_width_nm"]
    N = p["molecules"] if molecules is None else molecules
    sig = p["release_sigma_nm"] if sigma is None else sigma
    s = p["omega_scale"] if omega else 0.0
    settings = dict(geometry_mode=geometry, realtime_modeling=False, pulse_center_nm=0.0,
                    pulse_sigma_nm=sig, pulse_max_frequency_hz=1e9, max_saved_frames=4000,
                    cleft_radius_nm=a, cleft_height_nm=p["cleft_height_nm"],
                    extracellular_volume_fraction=p["alpha"])
    with temporary_global_settings(**settings):
        Km_s, Vc_s, Vp_s = uptake_to_solver_units(p["Km"], p["Vmax_cleft"], p["Vmax_pm"], p["alpha"], geometry,
                                                  p["cleft_height_nm"])
        p_hist, r, t, D_values, _, info = solve_fp_equation(
            grid["R"], grid["dt"], grid["dr"], grid["T"],
            p["D_cleft"], p["D_pm"], xi_for(a, b), a, b,
            0.05 * s, 0.05 * s * p.get("omega_pm_ratio", 3.0),
            N, (0.0, p.get("release_window_nm", 5.0 * sig) if sigma is None else 5.0 * sig), 1, None, (0.0, 0.0),
            "count", "schedule", 0.0, 0.0, 0.0, False,
            config.D_transition_kind, config.D_transition_steepness,
            config.Omega_transition_kind, config.Omega_transition_steepness,
            outer_boundary_mode="dirichlet", normalization_mode="concentration",
            uptake_kind="saturable" if p["Vmax_pm"] > 0 or p["Vmax_cleft"] > 0 else "none",
            Vmax_cleft=Vc_s, Vmax_pm=Vp_s, Km=Km_s,
        )
        dV = radial_cell_measures(r, grid["dr"], geometry)
        factor = solver_to_uM_factor(p["alpha"], geometry, p["cleft_height_nm"])
    assert info["count"] == 1
    return {"r": r, "t": t, "c": p_hist * factor, "dV": dV, "molecules_left": (p_hist * dV[:, None]).sum(0)}


def blur_line(values_r, r_um, xs_um, fxy, fz, step=0.05):
    hx, hyz = float(np.max(xs_um)) + 1.5, 4.0
    x = np.arange(-hx, hx + step / 2, step); y = np.arange(-hyz, hyz + step / 2, step)
    X, Y, Z = np.meshgrid(x, y, y, indexing="ij")
    F = np.interp(np.sqrt(X**2 + Y**2 + Z**2), r_um, values_r, right=0.0)
    Fb = gaussian_filter(F, sigma=(fxy / 2.3548 / step, fxy / 2.3548 / step, fz / 2.3548 / step), mode="constant")
    iy = len(y) // 2
    return np.interp(xs_um, x, Fb[:, iy, iy])


def sensor_profile(fine, long, p, xs_um, cleft=True):
    """Пик датчика B(r): до 10 мс по мелкой сетке, дальше по крупной; затем PSF и нормировка."""
    koff = 1.0 / (p["sensor_tau_off_ms"] * 1000.0); kon = koff / p["sensor_Kd_uM"]
    r = long["r"]
    B = np.zeros(r.size); Bmax = np.zeros(r.size)

    def step(C0, C1, h):
        nonlocal B
        C = 0.5 * (C0 + C1); a = kon * C + koff; Binf = kon * C / a
        B = Binf + (B - Binf) * np.exp(-a * h); np.maximum(Bmax, B, out=Bmax)

    if fine is not None:
        cf = np.array([np.interp(r, fine["r"], fine["c"][:, k], right=0.0) for k in range(fine["t"].size)]).T
        for k in range(1, fine["t"].size):
            step(cf[:, k - 1], cf[:, k], fine["t"][k] - fine["t"][k - 1])
        t0 = fine["t"][-1]
    else:
        t0 = 0.0
    tl, cl = long["t"], long["c"]
    ks = np.where(tl > t0)[0]
    prev_t, prev_c = t0, (np.interp(r, fine["r"], fine["c"][:, -1], right=0.0) if fine is not None else cl[:, 0])
    for k in ks:
        step(prev_c, cl[:, k], tl[k] - prev_t); prev_t, prev_c = tl[k], cl[:, k]
    # Яркость пропорциональна объёму жидкости с датчиком в элементе ткани:
    # снаружи щели это доля α, внутри шара r < a реальна только щель-диск π·a²·h,
    # поэтому вес 3h/(4a) (объём диска / объём шара). Для сферической геометрии вес α везде.
    wfl = np.full(r.size, p["alpha"])
    if cleft:
        wfl[r < p["cleft_radius_nm"]] = 3.0 * p["cleft_height_nm"] / (4.0 * p["cleft_radius_nm"])
    prof = blur_line(Bmax * wfl, r / 1000.0, np.concatenate([[0.0], xs_um]), p["psf_xy_um"], p["psf_z_um"])
    return prof[1:] / prof[0]


def cleft_receptor_inputs(fine, p):
    """Глутамат у рецепторов постсинаптической плотности: узлы r < a, веса = площадь 2πr·dr."""
    r = fine["r"]; m = r < p["cleft_radius_nm"]
    w = 2 * np.pi * r[m]
    return fine["c"][m], fine["t"], w


def uptake_tau_ms(sim, p):
    """Время спада суммарного потока захвата (аналог времени очистки Diamond 2005)."""
    r, t, c, dV = sim["r"], sim["t"], sim["c"], sim["dV"]
    a = p["cleft_radius_nm"]
    Vm = np.where(r > a + p["transition_width_nm"] / 2, p["Vmax_pm"], p["Vmax_cleft"])
    J = (Vm[:, None] * c / (p["Km"] + c) * dV[:, None]).sum(0)
    k0 = int(np.argmax(J)); Jp = J[k0]
    if Jp <= 0:
        return float("nan")
    below = np.nonzero(J[k0:] < Jp / math.e)[0]
    if below.size == 0:
        return float("inf")   # поток захвата не упал в e раз за окно расчёта (захват слишком слабый)
    return (t[k0 + below[0]] - t[k0]) / 1000.0


# ---------------------------------------------------------------------------
# Все наблюдаемые для одного набора параметров
# ---------------------------------------------------------------------------
def finish(out, curves, fine, p, want_curves, data):
    """Clements (рецепторы в щели) и служебные кривые; вызывается в конце evaluate."""
    # Clements: рецепторы в щели
    A, tA, w = cleft_receptor_inputs(fine, p)
    for i, ant in (("clements1992_fig1D_DAA", "D-AA"), ("clements1992_fig1D_DCPP", "D-CPP")):
        x, y, s = arr(data[i])
        out[i] = epsc_ratio(A, tA, x, ant, weights=w, dt_us=2.5)
        if want_curves:
            xc = np.logspace(-2 if ant == "D-CPP" else -1, 1.3 if ant == "D-CPP" else 3.6, 25)
            curves[i] = (xc, epsc_ratio(A, tA, xc, ant, weights=w, dt_us=2.5))
    if want_curves:
        r = fine["r"]; m = r < p["cleft_radius_nm"]
        curves["cleft_mean_mM"] = (fine["t"] / 1000.0, (fine["c"][m] * fine["dV"][m, None]).sum(0) / fine["dV"][m].sum() / 1000.0)
        curves["uptake_tau_ms"] = uptake_tau_ms(fine, p)
        curves["mass_fine"] = (fine["t"], fine["molecules_left"])
    return out, curves


def evaluate(p, data, want_curves=False, original=False, with_validation=False):
    geom = "spherical" if original else "cleft_disk"
    fine = run(p, GRID_FINE, geometry=geom)
    long = run(p, GRID_LONG, geometry=geom)
    out, curves = {}, {}
    syn_ids = ["matthews2022_fig1F", "matthews2022_fig2A_small", "matthews2022_fig2D_spont"]
    xs_all = np.linspace(0, 6, 61)
    prof_fine = sensor_profile(fine, long, p, cleft=(geom == "cleft_disk"), xs_um=np.concatenate([xs_all, np.unique(np.concatenate([arr(data[i])[0] for i in syn_ids]))]))
    prof_curve, prof_pts = prof_fine[:xs_all.size], prof_fine[xs_all.size:]
    xs_pts = np.unique(np.concatenate([arr(data[i])[0] for i in syn_ids]))
    for i in syn_ids:
        x, y, s = arr(data[i]); out[i] = np.interp(x, xs_pts, prof_pts)
    curves["synaptic"] = (xs_all, prof_curve)
    # анкейджинг: источник в нейропиле, без щели и без дрейфа
    pu = dict(p); pu["Vmax_cleft"] = p["Vmax_pm"]; pu["D_cleft"] = p["D_pm"]   # в нейропиле щели нет: D и захват всюду как снаружи
    unc = run(pu, GRID_LONG, geometry="spherical", molecules=p["uncaging_molecules"],
              sigma=p["uncaging_sigma_nm"], omega=False)
    x3, y3, s3 = arr(data["matthews2022_fig3B_uncaging"])
    pr = sensor_profile(None, unc, pu, np.concatenate([xs_all, x3]), cleft=False)
    out["matthews2022_fig3B_uncaging"] = pr[xs_all.size:]; curves["uncaging"] = (xs_all, pr[:xs_all.size])
    # NMDA-рецепторы шипика на расстоянии d от точки анкейджинга (Matthews Fig. 3C):
    # глутамат c(r = d, t) из того же расчёта анкейджинга -> пик A2R -> нормировка на d = 0
    xn, yn, sn = arr(data["matthews2022_fig3C_NMDA"])
    dn = np.linspace(0.0, 2.5, 11) if want_curves else xn
    An = np.array([np.interp(d * 1000.0, unc["r"], unc["c"][:, k]) for k in range(unc["t"].size) for d in dn]).reshape(unc["t"].size, dn.size).T
    pk = a2r_peak_per_node(An, unc["t"], dt_us=5.0); nm = pk / pk[0]
    out["matthews2022_fig3C_NMDA"] = np.interp(xn, dn, nm) if want_curves else nm
    curves["nmda"] = (dn, nm)
    # AMPA-рецепторы шипика (Matthews Fig. 4C) и то же при блокаде транспортёров TBOA (Fig. 4D).
    # В подгонку НЕ входят (см. VALIDATION_SETS и docs/unified_model.md, п. 7.5): форма профиля
    # в этой модели не воспроизводится ни при каких N и константе захвата; считаются как проверка.
    if not with_validation:
        return finish(out, curves, fine, p, want_curves, data)
    # тот же источник анкейджинга; при TBOA захват выключен (Vmax = 0), всё остальное то же
    pt = dict(pu); pt["Vmax_pm"] = 0.0; pt["Vmax_cleft"] = 0.0
    unc_tboa = run(pt, GRID_LONG, geometry="spherical", molecules=p["uncaging_molecules"],
                   sigma=p["uncaging_sigma_nm"], omega=False)
    for key, sim, ds_id in (("ampa", unc, "matthews2022_fig4C_AMPA"), ("ampa_tboa", unc_tboa, "matthews2022_fig4D_AMPA_TBOA")):
        xa_, ya_, sa_ = arr(data[ds_id])
        da = np.linspace(0.0, 1.5, 16) if want_curves else xa_
        Aa = np.array([[np.interp(d * 1000.0, sim["r"], sim["c"][:, k]) for k in range(sim["t"].size)] for d in da])
        pa = ampa_open_peak_per_node(Aa, sim["t"], dt_us=2.0); na = pa / pa[0]
        out[ds_id] = np.interp(xa_, da, na) if want_curves else na
        curves[key] = (da, na)
    return finish(out, curves, fine, p, want_curves, data)


FIT_SETS = ["matthews2022_fig1F", "matthews2022_fig2A_small", "matthews2022_fig2D_spont",
            "matthews2022_fig3B_uncaging", "matthews2022_fig3C_NMDA",
            "clements1992_fig1D_DAA", "clements1992_fig1D_DCPP"]


# Наборы, которые считаются для проверки, но в χ² подгонки не входят (п. 7.5 документа)
VALIDATION_SETS = ["matthews2022_fig4C_AMPA", "matthews2022_fig4D_AMPA_TBOA"]


def residuals_from(out, data):
    res, per = [], {}
    for i in FIT_SETS:
        x, y, s = arr(data[i]); r_ = (out[i] - y) / s
        res.append(r_); per[i] = float(np.sum(r_**2))
    return np.concatenate(res), per


def to_vec(p, keys):
    return np.array([math.log10(p[k]) if k in LOG_KEYS else p[k] for k in keys])


def from_vec(v, keys, base):
    p = dict(base)
    for k, x in zip(keys, v):
        p[k] = 10**x if k in LOG_KEYS else float(x)
    return p


def plot_all(p, data, out, curves, title, path):
    fig, axes = plt.subplots(3, 4, figsize=(19, 12.5), dpi=110)
    def dist_panel(ax, i, curve, ttl, xmax=6.2, color="#2a78d6", note=""):
        x, y, s = arr(data[i])
        ax.errorbar(x, y, yerr=s, fmt="o", color="k", ms=4, capsize=2, label=data[i]["figure"].split(",")[0] + " (измерение)")
        if curve is not None:
            r_ = np.sum(((out[i] - y) / s) ** 2) if i in out else float("nan")
            ax.plot(*curve, color=color, lw=2, label=f"модель{note}, χ² = {r_:.1f} ({len(x)} точек)")
        ax.set_title(ttl, fontsize=10); ax.set_xlim(0, xmax); ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("расстояние, мкм"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    dist_panel(axes[0][0], "matthews2022_fig1F", curves["synaptic"], "Matthews 2022 Fig. 1F: синапс, iGluSnFR")
    dist_panel(axes[0][1], "matthews2022_fig2A_small", curves["synaptic"], "Matthews 2022 Fig. 2A: малые ответы")
    dist_panel(axes[0][2], "matthews2022_fig2D_spont", curves["synaptic"], "Matthews 2022 Fig. 2D: спонтанные")
    dist_panel(axes[0][3], "matthews2022_fig3B_uncaging", curves["uncaging"], "Matthews 2022 Fig. 3B: анкейджинг, iGluSnFR")
    dist_panel(axes[1][0], "matthews2022_fig3C_NMDA", curves["nmda"], "Matthews 2022 Fig. 3C: анкейджинг, NMDA-ток", xmax=2.6)
    for ax, i in zip(axes[1][1:3], ["clements1992_fig1D_DAA", "clements1992_fig1D_DCPP"]):
        x, y, s = arr(data[i])
        ax.errorbar(x, y, yerr=s, fmt="o", color="k", ms=4, capsize=2, label="Clements 1992 Fig. 1D (измерение)")
        if i in curves:
            ax.plot(*curves[i], color="#2a78d6", lw=2, label=f"модель, χ² = {np.sum(((out[i] - y) / s) ** 2):.1f} ({len(x)} точек)")
        ax.set_xscale("log"); ax.set_ylim(-0.05, 1.1); ax.set_xlabel(("D-AA" if "DAA" in i else "D-CPP") + ", мкМ")
        ax.set_title("Clements 1992 Fig. 1D, " + ("D-AA" if "DAA" in i else "D-CPP (контроль)"), fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax = axes[1][3]
    t, cm = curves["cleft_mean_mM"]
    ax.plot(t, cm, color="#2a78d6", lw=2, label="модель: среднее по щели")
    x3, y3, s3 = arr(data["clements1992_fig3C"])
    ax.plot(x3, y3, "o", color="k", ms=3, label="Clements Fig. 3C (оценка, не в подгонке)")
    tt = np.linspace(0, 10, 400)
    ax.plot(tt, 3.2 * np.exp(-tt / 0.1) + 0.525 * np.exp(-tt / 1.0), "--", color="#eb6834", label="Diamond & Jahr 1997 (оценка)")
    ax.set_yscale("log"); ax.set_ylim(1e-4, 30); ax.set_xlim(0, 10); ax.set_xlabel("время, мс"); ax.set_ylabel("мМ")
    ax.set_title("Проверка: глутамат в щели (не в подгонке)", fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    if "ampa" in curves:
        dist_panel(axes[2][0], "matthews2022_fig4C_AMPA", curves["ampa"], "Matthews 2022 Fig. 4C: AMPA-ток (проверка, не в подгонке)", xmax=1.5, color="#eb6834")
        dist_panel(axes[2][1], "matthews2022_fig4D_AMPA_TBOA", curves["ampa_tboa"], "Matthews 2022 Fig. 4D: AMPA + TBOA (проверка)", xmax=1.5, color="#eb6834", note=" (захват = 0)")
    else:
        axes[2][0].axis("off"); axes[2][1].axis("off")
    for ax in axes[2][2:]:
        ax.axis("off")
    keys = ["cleft_radius_nm", "D_cleft", "D_pm", "omega_scale", "Vmax_pm", "molecules", "psf_xy_um", "psf_z_um", "uncaging_molecules"]
    txt = "Параметры модели:\n" + "\n".join(f"  {k} = {p[k]:.4g}" for k in keys if k in p)
    axes[2][2].text(0.0, 1.0, txt, va="top", fontsize=10, family="monospace", transform=axes[2][2].transAxes)
    fig.suptitle(title, fontsize=12); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--original", action="store_true")
    ap.add_argument("--no-drift", action="store_true")
    ap.add_argument("--max-nfev", type=int, default=60)
    ap.add_argument("--from-report", default=None, help="пересчитать и перерисовать по параметрам из *_report.json без подгонки")
    ap.add_argument("--tag", default=None, help="имя выходных файлов")
    ap.add_argument("--fix", nargs="*", default=[], help="зафиксировать параметры: имя=значение (убираются из подгонки)")
    ap.add_argument("--with-ampa", action="store_true", help="включить AMPA (Fig. 4C, 4D) в χ² подгонки")
    ap.add_argument("--start-report", default=None, help="начальная точка подгонки из *_report.json")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data, scalars = load_data()
    base = dict(PARAMS)
    keys = list(FIT_KEYS)
    tag = "unified"
    if args.original:
        base.update(ORIGINAL); tag = "original"
    if args.no_drift:
        base["omega_scale"] = 0.0; keys.remove("omega_scale"); tag = "unified_nodrift"
    if args.start_report:
        st = json.loads(Path(args.start_report).read_text(encoding="utf-8"))["params"]
        for kk in keys:
            base[kk] = st[kk]
    for item in args.fix:
        kk, vv = item.split("="); base[kk] = float(vv)
        if kk in keys:
            keys.remove(kk)
    if args.with_ampa:
        FIT_SETS.extend(VALIDATION_SETS); VALIDATION_SETS.clear()
    if args.tag:
        tag = args.tag
    log = open(OUT_DIR / f"{tag}_log.txt", "a", encoding="utf-8")

    def F(v):
        p = from_vec(v, keys, base); t0 = time.time()
        out, _ = evaluate(p, data, with_validation=args.with_ampa)
        res, per = residuals_from(out, data)
        line = (f"chi2={np.sum(res**2):9.2f} | " + " ".join(f"{k}={p[k]:.4g}" for k in keys)
                + " | " + " ".join(f"{k.split('_',1)[1]}={v:.1f}" for k, v in per.items()) + f" | {time.time()-t0:.0f}s")
        print(line, flush=True); log.write(line + "\n"); log.flush()
        return res

    if args.from_report:
        rep = json.loads(Path(args.from_report).read_text(encoding="utf-8"))
        base.update(rep["params"]); keys = rep.get("fit_keys") or keys
        tag = args.tag or (rep["tag"] + "_replot")
    if args.eval_only or args.original or args.from_report:
        p = base
    else:
        lo = [math.log10(BOUNDS[k][0]) if k in LOG_KEYS else BOUNDS[k][0] for k in keys]
        hi = [math.log10(BOUNDS[k][1]) if k in LOG_KEYS else BOUNDS[k][1] for k in keys]
        v0 = np.clip(to_vec(base, keys), np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        sol = least_squares(F, v0, bounds=(lo, hi), diff_step=0.05, x_scale="jac", max_nfev=args.max_nfev,
                            ftol=1e-3, xtol=1e-3)
        p = from_vec(sol.x, keys, base)
        J = sol.jac; dof = max(sol.fun.size - len(keys), 1)
        try:
            cov = np.linalg.inv(J.T @ J) * max(np.sum(sol.fun**2) / dof, 1.0)
            err = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            err = np.full(len(keys), np.nan)
    GRID_LONG["dt"] = 2.0   # итоговый расчёт на мелком шаге по времени
    out, curves = evaluate(p, data, want_curves=True, original=args.original, with_validation=True)
    res, per = residuals_from(out, data)
    chi2 = float(np.sum(res**2)); n = res.size; k = 0 if (args.eval_only or args.original) else len(keys)
    if args.from_report:
        err = np.full(len(keys), np.nan)
        if "param_errors_1sigma" in rep:
            err = np.array([rep["param_errors_1sigma"].get(kk, [np.nan])[0] for kk in keys])
    report = {"tag": tag, "params": p, "fit_keys": [] if k == 0 else keys,
              "chi2_total": chi2, "n_points": int(n), "n_free": k, "chi2_per_dof": chi2 / max(n - k, 1),
              "aic": chi2 + 2 * k, "chi2_by_dataset": per,
              "checks": {"uptake_tau_ms_vs_Diamond2005_0.75": curves.get("uptake_tau_ms"),
                         "cleft_mean_mM_at_0.5_1_2_4_ms": [float(np.interp(tt, *curves["cleft_mean_mM"])) for tt in (0.5, 1, 2, 4)]},
              "model_points": {i: [float(v) for v in out[i]] for i in FIT_SETS + VALIDATION_SETS if i in out},
              "validation_chi2_not_in_fit": {i: float(np.sum(((out[i] - arr(data[i])[1]) / arr(data[i])[2]) ** 2))
                                             for i in VALIDATION_SETS if i in out}}
    if k:
        report["param_errors_1sigma"] = {kk: (float(e), "в log10" if kk in LOG_KEYS else "") for kk, e in zip(keys, err)}
    (OUT_DIR / f"{tag}_report.json").write_text(json.dumps(json_clean(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    plot_all(p, data, out, curves, f"{tag}: χ² = {chi2:.1f} на {n} точках, свободных параметров {k}", OUT_DIR / f"{tag}_fit.png")
    # проверка: средняя концентрация в щели против Clements Fig. 3C и Diamond & Jahr
    t, cm = curves["cleft_mean_mM"]
    x3, y3, s3 = arr(data["clements1992_fig3C"])
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    ax.plot(t, cm, color="#2a78d6", lw=2, label="единая модель: среднее по щели r < a")
    ax.plot(x3, y3, "o", color="k", ms=4, label="Clements 1992 Fig. 3C (оценка через модель)")
    tt = np.linspace(0, 10, 400)
    ax.plot(tt, 3.2 * np.exp(-tt / 0.1) + 0.525 * np.exp(-tt / 1.0), "--", color="#eb6834", label="Diamond & Jahr 1997 (оценка через модель)")
    ax.set_yscale("log"); ax.set_ylim(1e-4, 30); ax.set_xlim(0, 10); ax.set_xlabel("время, мс"); ax.set_ylabel("мМ")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_title("Проверка без подгонки: концентрация в щели", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT_DIR / f"{tag}_cleft_check.png"); plt.close(fig)
    print(json.dumps({k_: report[k_] for k_ in ("chi2_total", "n_points", "n_free", "chi2_per_dof", "chi2_by_dataset", "checks")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
