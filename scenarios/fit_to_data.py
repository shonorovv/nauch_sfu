# -*- coding: utf-8 -*-
"""
scenarios/fit_to_data.py - подгонка модели к экспериментальным точкам (МНК).

Что делает
----------
1. Читает экспериментальные точки из data/experimental_points.json
   (единственный источник данных; таблица для чтения - data/experimental_points.xlsx).
2. Запускает решатель Фоккера-Планка (solver/fp_solver.py) через
   scenarios/verification.run_single_release с заданными параметрами.
3. Из решения C(r, t) считает ту же величину, которую измеряли в опыте.
4. Считает функцию потерь
       chi2 = sum_i ((модель_i - данные_i) / sigma_i)^2
   и ищет её минимум по свободному параметру. Параметр в минимуме и есть результат.
5. Строит графики «модель поверх точек» и пишет отчёт в fit_outputs/.

Две подгонки
------------
A. Matthews 2022, Fig. 1F (ИЗМЕРЕНИЕ, сигнал датчика iGluSnFr от расстояния).
   Свободный параметр: Vmax захвата транспортёрами вне синапса (Vmax_pm).
   D_pm = 320 нм²/мкс фиксирован: это измеренное значение (Zheng et al. 2008).
   Модель наблюдения: датчик связывает глутамат
       dB/dt = kon*C*(1-B) - koff*B,   Kd = koff/kon = 4,9 мкМ,  1/koff = 60 мс
   (оба числа из Matthews 2022), затем сигнал размывается оптикой микроскопа
   (гауссова PSF; её размер в статье НЕ указан - это допущение, см. PSF ниже),
   и нормируется на значение у бутона, как у авторов.

B. Clements 1992, Fig. 3C (УРАВНЕНИЕ, которое авторы подогнали к своим измерениям).
   Свободный параметр: D_cleft. Сравнивается средняя концентрация в синапсе r < a.

Запуск (из папки diplom):
    python -m scenarios.fit_to_data            # обе подгонки
    python -m scenarios.fit_to_data --only A   # только Matthews
    python -m scenarios.fit_to_data --psf 0.5 2.5   # другая PSF (проверка чувствительности)

Что можно менять: SENSOR, PSF, GRID_*, границы параметров в BOUNDS_*.
Где можно ошибиться: размер PSF и число молекул влияют на форму профиля;
результат подгонки A действителен только при указанных допущениях.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from scenarios import verification as V

DATA_FILE = Path("data/experimental_points.json")
OUT_DIR = Path("fit_outputs")

# ---------------------------------------------------------------------------
# Параметры модели наблюдения (датчик и оптика)
# ---------------------------------------------------------------------------
SENSOR = {"Kd_uM": 4.9, "tau_off_ms": 60.0}       # Matthews 2022
PSF = {"fwhm_xy_um": 0.35, "fwhm_z_um": 1.5}       # ДОПУЩЕНИЕ: в статье не указано

# ---------------------------------------------------------------------------
# Сетки расчёта
# ---------------------------------------------------------------------------
# A: данные до 6 мкм -> область 10 мкм, чтобы граница (C = 0) не искажала профиль;
#    30 мс - время, за которое датчик успевает «собрать» сигнал.
GRID_A = {"MAX_R": 10000.0, "DR": 4.0, "DT": 1.0, "T_MAX": 30000.0}
# B: данные до 10 мс внутри синапса -> область 3 мкм, мелкая сетка.
GRID_B = {"MAX_R": 3000.0, "DR": 2.0, "DT": 0.5, "T_MAX": 10000.0}

# Границы поиска (в логарифмах: параметры меняются на порядки)
BOUNDS_A = (-6.0, 0.0)     # log10(Vmax_pm, мкМ/мкс)
BOUNDS_B = (0.0, 3.0)      # log10(D_cleft, нм²/мкс)

UM3_TO_UM = 1.0 / 6.02214076e23 / 1e-15 * 1e6   # 1 молекула/мкм³ в мкМ (для проверок)


# ===========================================================================
# Данные
# ===========================================================================
def load_dataset(dataset_id):
    """Возвращает словарь набора и массивы x, y, sigma."""
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    for ds in data["datasets"]:
        if ds["id"] == dataset_id:
            pts = ds["points"]
            x = np.array([p["x"] for p in pts], dtype=float)
            y = np.array([p["y"] for p in pts], dtype=float)
            s = np.array([p["sigma"] for p in pts], dtype=float)
            return ds, x, y, s
    raise KeyError(dataset_id)


# ===========================================================================
# Запуск решателя
# ===========================================================================
def simulate(grid, D_cleft, D_pm, Vmax_pm):
    """Один выброс в t = 0. Возвращает r (нм), t (мкс), C (мкМ)."""
    V.MAX_R, V.DR = grid["MAX_R"], grid["DR"]
    params = dict(V.LITERATURE_PARAMS)
    params["Vmax_pm_uM_per_us"] = float(Vmax_pm)
    sim = V.run_single_release(D_cleft, D_pm, uptake=True, params=params,
                               t_max=grid["T_MAX"], dt=grid["DT"])
    return sim


# ===========================================================================
# Модель наблюдения A: датчик + оптика
# ===========================================================================
def sensor_peak_occupancy(c_uM, t_us, Kd_uM, tau_off_ms):
    """
    Пиковая занятость датчика B(r) за всё время расчёта.
    На каждом шаге между сохранёнными кадрами C считается постоянной,
    и линейное уравнение для B решается точно (без ограничений на шаг).
    """
    koff = 1.0 / (tau_off_ms * 1000.0)        # 1/мкс
    kon = koff / Kd_uM                          # 1/(мкМ*мкс)
    B = np.zeros(c_uM.shape[0]); Bmax = np.zeros_like(B)
    for k in range(1, t_us.size):
        h = t_us[k] - t_us[k - 1]
        C = 0.5 * (c_uM[:, k] + c_uM[:, k - 1])
        a = kon * C + koff
        Binf = kon * C / a
        B = Binf + (B - Binf) * np.exp(-a * h)
        np.maximum(Bmax, B, out=Bmax)
    return Bmax


def blur_along_line(values_r, r_um, xs_um, fwhm_xy, fwhm_z, step=0.05):
    """
    Радиальную функцию f(r) размещаем в 3D, свёртываем с гауссовой PSF
    и берём значения вдоль линии сканирования (ось x), проходящей через источник.
    """
    hx, hyz = float(max(xs_um)) + 1.5, 3.5
    x = np.arange(-hx, hx + step / 2, step)
    y = np.arange(-hyz, hyz + step / 2, step)
    X, Y, Z = np.meshgrid(x, y, y, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    F = np.interp(R, r_um, values_r, right=0.0)
    s_xy = fwhm_xy / 2.3548 / step
    s_z = fwhm_z / 2.3548 / step
    Fb = gaussian_filter(F, sigma=(s_xy, s_xy, s_z), mode="constant")
    iy = len(y) // 2
    return np.interp(xs_um, x, Fb[:, iy, iy])


def model_matthews(Vmax_pm, xs_um, psf=PSF, D_cleft=330.0, D_pm=320.0, observable="sensor"):
    """Нормированный профиль сигнала в точках xs_um. observable='peakC' - пик концентрации."""
    sim = simulate(GRID_A, D_cleft, D_pm, Vmax_pm)
    r_um = sim["r"] / 1000.0
    if observable == "sensor":
        f = sensor_peak_occupancy(sim["c_uM"], sim["t"], SENSOR["Kd_uM"], SENSOR["tau_off_ms"])
    else:
        f = sim["c_uM"].max(axis=1)
    prof = blur_along_line(f, r_um, np.concatenate([[0.0], xs_um]), psf["fwhm_xy_um"], psf["fwhm_z_um"])
    return prof[1:] / prof[0]


# ===========================================================================
# Модель наблюдения B: средняя концентрация в синапсе
# ===========================================================================
def model_clements(D_cleft, ts_ms, D_pm=320.0, Vmax_pm=None):
    Vmax_pm = V.LITERATURE_PARAMS["Vmax_pm_uM_per_us"] if Vmax_pm is None else Vmax_pm
    sim = simulate(GRID_B, D_cleft, D_pm, Vmax_pm)
    r, t, c, dV = sim["r"], sim["t"], sim["c_uM"], sim["dV"]
    inside = r < config.a
    cleft_mM = (c[inside] * dV[inside, None]).sum(axis=0) / dV[inside].sum() / 1000.0
    return np.interp(ts_ms * 1000.0, t, cleft_mM), t / 1000.0, cleft_mM


# ===========================================================================
# Подгонка по одному параметру: минимум chi2 + интервал по профилю (Δchi2 = 1)
# ===========================================================================
def fit_1d(chi2_of_logp, bounds, n_scan=13):
    """Сначала грубый перебор по сетке, затем уточнение минимума."""
    grid = np.linspace(bounds[0], bounds[1], n_scan)
    scan = [(g, chi2_of_logp(g)) for g in grid]
    i = int(np.argmin([s[1] for s in scan]))
    lo = grid[max(i - 1, 0)]; hi = grid[min(i + 1, n_scan - 1)]
    res = minimize_scalar(chi2_of_logp, bounds=(lo, hi), method="bounded",
                          options={"xatol": 0.02})
    best = (res.x, res.fun) if res.fun <= scan[i][1] else scan[i]
    scan.append(best)
    # интервал 1σ по профилю chi2: идём от минимума в обе стороны мелким шагом,
    # пока chi2 не превысит chi2_min + 1 (или не упрёмся в границу поиска)
    edges = []
    for direction in (-1, 1):
        g_prev = best[0]
        for k in range(1, 61):
            g = best[0] + direction * 0.005 * k
            if g < bounds[0] or g > bounds[1]:
                edges.append(bounds[0] if direction < 0 else bounds[1]); break
            c = chi2_of_logp(g); scan.append((g, c))
            if c > best[1] + 1.0:
                edges.append(g_prev); break
            g_prev = g
        else:
            edges.append(g_prev)
    scan.sort()
    return best, (edges[0], edges[1]), scan


# ===========================================================================
# Основная часть
# ===========================================================================
def run_A(psf):
    ds, x, y, s = load_dataset("matthews2022_fig1F")
    cache = {}

    def chi2(logV):
        key = round(float(logV), 4)
        if key not in cache:
            m = model_matthews(10 ** logV, x, psf=psf)
            cache[key] = float(np.sum(((m - y) / s) ** 2))
            print(f"  A: Vmax_pm = {10**logV:.3g} мкМ/мкс  chi2 = {cache[key]:.2f}", flush=True)
        return cache[key]

    (bestlog, chi2min), (lo, hi), scan = fit_1d(chi2, BOUNDS_A)
    best_profile = model_matthews(10 ** bestlog, x, psf=psf)
    xf = np.linspace(0, 6, 61)
    curves = {
        "best": model_matthews(10 ** bestlog, xf, psf=psf),
        "config": model_matthews(V.LITERATURE_PARAMS["Vmax_pm_uM_per_us"], xf, psf=psf),
        "peakC": model_matthews(10 ** bestlog, xf, psf=psf, observable="peakC"),
    }
    dof = len(x) - 1
    res = {
        "dataset": ds["id"], "free_parameter": "Vmax_pm, мкМ/мкс",
        "fixed": {"D_pm_nm2_per_us": 320.0, "D_cleft_nm2_per_us": 330.0,
                  "Km_uM": V.LITERATURE_PARAMS["Km_uM"],
                  "molecules": V.LITERATURE_PARAMS["molecules"], "alpha": V.LITERATURE_PARAMS["alpha"],
                  "sensor": SENSOR, "psf_assumption": psf},
        "best_Vmax_pm_uM_per_us": 10 ** bestlog,
        "interval_1sigma_uM_per_us": [10 ** lo, 10 ** hi],
        "at_lower_bound": bool(abs(lo - BOUNDS_A[0]) < 1e-9),
        "chi2_min": chi2min, "dof": dof, "chi2_per_dof": chi2min / dof,
        "chi2_config_Vmax": float(np.sum(((model_matthews(V.LITERATURE_PARAMS["Vmax_pm_uM_per_us"], x, psf=psf) - y) / s) ** 2)),
        "points": [{"x": float(a), "data": float(b), "sigma": float(c), "model": float(d)}
                   for a, b, c, d in zip(x, y, s, best_profile)],
        "scan": [{"log10_Vmax": float(g), "chi2": float(c)} for g, c in scan],
    }
    return ds, (x, y, s), xf, curves, res


def run_B():
    ds, x, y, s = load_dataset("clements1992_fig3C")
    cache = {}

    def chi2(logD):
        key = round(float(logD), 4)
        if key not in cache:
            m, _, _ = model_clements(10 ** logD, x)
            cache[key] = float(np.sum(((m - y) / s) ** 2))
            print(f"  B: D_cleft = {10**logD:.3g} нм²/мкс  chi2 = {cache[key]:.2f}", flush=True)
        return cache[key]

    (bestlog, chi2min), (lo, hi), scan = fit_1d(chi2, BOUNDS_B)
    m_best, tf, cf_best = model_clements(10 ** bestlog, x)
    _, tf_lit, cf_lit = model_clements(330.0, x)
    dof = len(x) - 1
    res = {
        "dataset": ds["id"], "free_parameter": "D_cleft, нм²/мкс",
        "fixed": {"D_pm_nm2_per_us": 320.0, "Vmax_pm_uM_per_us": V.LITERATURE_PARAMS["Vmax_pm_uM_per_us"],
                  "a_nm": config.a, "geometry": "spherical"},
        "best_D_cleft_nm2_per_us": 10 ** bestlog,
        "interval_1sigma_nm2_per_us": [10 ** lo, 10 ** hi],
        "measured_D_cleft_reference_nm2_per_us": 330.0,
        "chi2_min": chi2min, "dof": dof, "chi2_per_dof": chi2min / dof,
        "chi2_at_330": chi2(math.log10(330.0)),
        "points": [{"x": float(a), "data": float(b), "sigma": float(c), "model": float(d)}
                   for a, b, c, d in zip(x, y, s, m_best)],
        "scan": [{"log10_D_cleft": float(g), "chi2": float(c)} for g, c in scan],
    }
    return ds, (x, y, s), (tf, cf_best, tf_lit, cf_lit), res


def plot_A(data, xf, curves, res, path):
    x, y, s = data
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    for ax, logy in zip(axes, (False, True)):
        ax.errorbar(x, y, yerr=s, fmt="o", color="k", ms=5, capsize=2,
                    label="Matthews 2022, Fig. 1F (измерение, n = 6)")
        ax.plot(xf, curves["best"], color="#2a78d6", lw=2,
                label=f"модель, подгонка: Vmax_pm = {res['best_Vmax_pm_uM_per_us']:.2g} мкМ/мкс")
        ax.plot(xf, curves["config"], color="#1baf7a", lw=2, ls="-.",
                label=f"модель, Vmax_pm из config ({V.LITERATURE_PARAMS['Vmax_pm_uM_per_us']:.2g})")
        ax.plot(xf, curves["peakC"], color="#eb6834", lw=2, ls="--",
                label="пик концентрации (без датчика) - для сравнения")
        ax.set_xlabel("расстояние от бутона, мкм"); ax.grid(alpha=0.3)
        if logy:
            ax.set_yscale("log"); ax.set_ylim(1e-3, 1.5)
        else:
            ax.set_ylim(-0.05, 1.1); ax.set_ylabel("нормированный сигнал")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Подгонка A: χ² = {res['chi2_min']:.1f} при {res['dof']} степенях свободы "
                 f"(PSF {res['fixed']['psf_assumption']['fwhm_xy_um']}×{res['fixed']['psf_assumption']['fwhm_z_um']} мкм - допущение)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_B(data, curves, res, path):
    x, y, s = data
    tf, cf_best, tf_lit, cf_lit = curves
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=150)
    ax.errorbar(x, y, yerr=s, fmt="o", color="k", ms=5, capsize=2,
                label="Clements 1992, Fig. 3C (уравнение, подогнанное к измерениям)")
    ax.plot(tf, cf_best, color="#2a78d6", lw=2,
            label=f"модель, подгонка: D_cleft = {res['best_D_cleft_nm2_per_us']:.3g} нм²/мкс")
    ax.plot(tf_lit, cf_lit, color="#eb6834", lw=2, ls="--",
            label="модель, D_cleft = 330 нм²/мкс (измерено)")
    ax.set_yscale("log"); ax.set_ylim(1e-4, 2); ax.set_xlim(0, 10)
    ax.set_xlabel("время после выброса, мс"); ax.set_ylabel("средняя концентрация в синапсе, мМ")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"Подгонка B: χ² = {res['chi2_min']:.1f} при {res['dof']} степенях свободы", fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="МНК-подгонка модели к экспериментальным точкам")
    ap.add_argument("--only", choices=["A", "B"], default=None)
    ap.add_argument("--psf", nargs=2, type=float, default=None, metavar=("FWHM_XY", "FWHM_Z"))
    args = ap.parse_args()
    psf = dict(PSF) if args.psf is None else {"fwhm_xy_um": args.psf[0], "fwhm_z_um": args.psf[1]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    t0 = time.time()
    if args.only in (None, "A"):
        print("Подгонка A: Matthews 2022 Fig. 1F, параметр Vmax_pm")
        ds, data, xf, curves, res = run_A(psf)
        tag = f"_psf{psf['fwhm_xy_um']}x{psf['fwhm_z_um']}"
        plot_A(data, xf, curves, res, OUT_DIR / f"fit_A_matthews{tag}.png")
        report["A" + tag] = res
    if args.only in (None, "B"):
        print("Подгонка B: Clements 1992 Fig. 3C, параметр D_cleft")
        ds, data, curves, res = run_B()
        plot_B(data, curves, res, OUT_DIR / "fit_B_clements.png")
        report["B"] = res
    out = OUT_DIR / "fit_report.json"
    old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    old.update(report)
    out.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово за {time.time() - t0:.0f} с: {out.resolve()}")


if __name__ == "__main__":
    main()
