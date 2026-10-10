# -*- coding: utf-8 -*-
"""
Рисунок: данные Matthews 2022 (Fig. 1F) и три наблюдаемые одной и той же
аналитической модели диффузии. Показывает, что сравнивать λ надо по той
величине, которую видит датчик, а не по пиковой концентрации.
Запуск: python plot_observables.py  (нужны numpy, scipy, matplotlib)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from observable_check import R_DATA, Y_DATA, sensor_peak, blur_line, lam_fit

# --- параметры (те же, что в проверке; PSF визуализации - допущение) ---
ALPHA, DSTAR, SIGMA0, M = 0.2, 0.30, 0.2, 4000
FXY, FZ = 0.35, 1.5
r_grid = np.linspace(0, 10, 2001)
xf = np.linspace(0, 6, 121)

def profile(k, which, xs=xf):
    Bmax, Cmax, E = sensor_peak(r_grid, M, ALPHA, DSTAR, SIGMA0, k)
    v = {"B": Bmax, "C": Cmax}[which]
    p = blur_line(v, r_grid, FXY, FZ, xs)
    return p / p[0]

curves = [
    ("датчик, k = 0",        profile(0.0, "B"), "#2a78d6", "-"),
    ("датчик, k = 0,1 1/мс", profile(0.1, "B"), "#1baf7a", "-."),
    ("пик концентрации",     profile(0.0, "C"), "#eb6834", "--"),
]
# λ считаем в тех же точках, что и у данных (0; 0,5; ...; 3 мкм), чтобы процедура совпадала
lams = [lam_fit(R_DATA, profile(k, w, R_DATA)) for k, w in ((0.0, "B"), (0.1, "B"), (0.0, "C"))]

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), dpi=160, facecolor=SURF)
for ax in axes:
    ax.set_facecolor(SURF)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("расстояние от источника, мкм", color=INK2)
    for (lab, y, col, ls), lam in zip(curves, lams):
        ax.plot(xf, y, color=col, ls=ls, lw=2, label=f"{lab} (λ = {lam:.2f} мкм)".replace(".", ","))
    ax.plot(R_DATA, Y_DATA, "o", ms=7, color=INK, mec=SURF, mew=1.5,
            label="данные Matthews 2022, Fig. 1F (λ = 1,15 мкм)", zorder=5)
    ax.set_xlim(0, 6.2)

axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("нормированный сигнал", color=INK2)
axes[0].set_title("Линейная шкала", color=INK, loc="left", fontsize=11)
axes[1].set_yscale("log"); axes[1].set_ylim(5e-4, 1.3)
axes[1].set_title("Логарифмическая шкала (экспонента = прямая)", color=INK, loc="left", fontsize=11)

# прямые подписи на логарифмической панели: в свободных местах рядом со своей кривой
for (lab, y, col, ls), (xl, yl) in zip(curves, ((3.7, 0.11), (1.85, 0.0105), (1.15, 0.0011))):
    axes[1].text(xl, yl, lab, color=INK2, fontsize=9.5)
from matplotlib.ticker import FuncFormatter
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))

fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
           frameon=False, fontsize=9.5, labelcolor=INK2, bbox_to_anchor=(0.5, 1.0))
fig.text(0.01, 0.01,
         "Модель: аналитическое решение для точечного источника (не основной решатель курсовой). "
         "D* = 0,30 мкм²/мс, α = 0,2, 4000 молекул, σ0 = 0,2 мкм.\n"
         "Датчик: Kd = 4,9 мкМ, τ_off = 60 мс (по Matthews 2022). PSF микроскопа 0,35 × 1,5 мкм (FWHM) - допущение, "
         "в статье для этих опытов не указан.\n"
         "λ у всех кривых и у данных посчитан одинаково: экспонента по точкам 0-3 мкм с шагом 0,5 мкм.",
         fontsize=8.3, color=MUTED, va="bottom")
fig.tight_layout(rect=(0, 0.1, 1, 0.9))
fig.savefig("observables_vs_matthews.png", facecolor=SURF)
print("λ:", [round(l, 2) for l in lams])
