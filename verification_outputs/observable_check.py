# -*- coding: utf-8 -*-
"""
Быстрая проверка гипотезы: "λ в Matthews 2022 измерен датчиком-интегратором (iGluSnFr),
поэтому сравнивать его надо с профилем ЭКСПОЗИЦИИ (интеграл C по времени) или с занятостью
датчика, а не с профилем ПИКОВОЙ концентрации".

Это упрощённая аналитическая модель (не основной решатель курсовой):
  - точечный источник с гауссовым начальным размером sigma0 во внеклеточной среде,
  - эффективная диффузия D* = D_free / tortuosity^2, доля объёма alpha,
  - линейный захват с константой k (1/мс)  [k = 0 - без захвата],
  - датчик: dB/dt = kon*C*(1-B) - koff*B, Kd = koff/kon,
  - размытие PSF микроскопа (гауссова, анизотропная) ПОСЛЕ нелинейности датчика.
Все допущения помечены в тексте ответа.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

UM3_TO_UM = 1.0 / 6.02214076e23 / 1e-15 * 1e6  # 1 молекула/мкм^3 в мкМ (~1.66e-3)

# Данные Matthews 2022, Fig. 1F (оцифровка, нормированный ΔF/F)
R_DATA = np.array([0, .49, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.51, 6])
Y_DATA = np.array([1.0, .666, .406, .280, .185, .105, .071, .048, .048, .044, .037, .044, .035])

def field_C(r, t, M, alpha, Dstar, sigma0, k):
    """C(r,t) в мкМ для гауссова источника M молекул, линейный захват k."""
    s2 = sigma0**2 + 2.0 * Dstar * t                 # дисперсия по каждой оси, мкм^2
    c = M / (alpha * (2*np.pi*s2)**1.5) * np.exp(-r**2 / (2*s2) - k*t)  # молекул/мкм^3 жидкости
    return c * UM3_TO_UM

def sensor_peak(r_grid, M, alpha, Dstar, sigma0, k, Kd=4.9, tau_off=60.0,
                t_max=60.0, dt=0.002):
    """Пиковая занятость датчика B(r) и пиковая концентрация, экспозиция."""
    koff = 1.0 / tau_off          # 1/мс
    kon = koff / Kd               # 1/(мкМ*мс)
    B = np.zeros_like(r_grid); Bmax = np.zeros_like(r_grid)
    Cmax = np.zeros_like(r_grid); E = np.zeros_like(r_grid)
    # неравномерный шаг: мелкий в начале, крупный в конце
    t = 0.0
    while t < t_max:
        h = dt if t < 2 else (0.02 if t < 20 else 0.1)
        C = field_C(r_grid, t + 0.5*h, M, alpha, Dstar, sigma0, k)
        # точное решение линейного ОДУ на шаге при постоянной C
        a = kon*C + koff
        Binf = kon*C / a
        B = Binf + (B - Binf) * np.exp(-a*h)
        Bmax = np.maximum(Bmax, B); Cmax = np.maximum(Cmax, C); E += C*h
        t += h
    return Bmax, Cmax, E

def blur_line(r_fun_vals, r_grid, fwhm_xy, fwhm_z, xs, step=0.05, half=(7.5, 3.5)):
    """Размытие радиальной функции 3D-PSF и выборка вдоль линии сканирования (ось x)."""
    hx, hyz = half
    x = np.arange(-hx, hx + step/2, step); y = np.arange(-hyz, hyz + step/2, step)
    X, Y, Z = np.meshgrid(x, y, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)
    V = np.interp(R, r_grid, r_fun_vals)
    sxy = fwhm_xy / 2.3548 / step; sz = fwhm_z / 2.3548 / step
    Vb = gaussian_filter(V, sigma=(sxy, sxy, sz), mode='constant')
    iy = len(y)//2
    return np.interp(xs, x, Vb[:, iy, iy])

def lam_fit(xs, ys):
    f = lambda x, A, l: A*np.exp(-x/l)
    m = xs <= 3.0
    p, _ = curve_fit(f, xs[m], ys[m], p0=[1, 1])
    return p[1]

if __name__ == "__main__":
    r_grid = np.linspace(0, 10, 2001)
    xs = R_DATA
    alpha, Dstar = 0.2, 0.30          # мкм^2/мс (= 300 нм^2/мкс)
    sigma0 = 0.2                      # мкм, размер области выброса (порядок радиуса бутона/щели)
    fwhm_xy, fwhm_z = 0.35, 1.5       # мкм, PSF визуализации - ДОПУЩЕНИЕ
    print(f"{'M':>6} {'k,1/мс':>7} | λ пик C | λ экспоз. | λ датчик | B(0) | χ² датчик vs Fig1F")
    for M in (4000, 8000):
        for k in (0.0, 0.1, 0.3, 1.0, 3.0):
            Bmax, Cmax, E = sensor_peak(r_grid, M, alpha, Dstar, sigma0, k)
            out = {}
            for name, v in (('C', Cmax), ('E', E), ('B', Bmax)):
                prof = blur_line(v, r_grid, fwhm_xy, fwhm_z, xs)
                prof = prof / prof[0]
                out[name] = (lam_fit(xs, prof), prof)
            chi = np.sum((out['B'][1] - Y_DATA)**2 / (0.02**2 + (0.05*Y_DATA)**2))
            print(f"{M:6d} {k:7.1f} | {out['C'][0]:7.2f} | {out['E'][0]:8.2f} | {out['B'][0]:8.2f} | {Bmax[0]:.2f} | {chi:8.1f}")
    print("Matthews Fig.1F: λ (та же процедура, точки 0-3 мкм) = 1.15 мкм")
