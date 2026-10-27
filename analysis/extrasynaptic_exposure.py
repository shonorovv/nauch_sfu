# -*- coding: utf-8 -*-
"""
analysis/extrasynaptic_exposure.py - внесинаптическая экспозиция глутамата (P0.5 ТЗ от 13.10.2026).

Заменяет по смыслу analysis/excitotoxicity.py. Величины здесь описывают, сколько свободного
глутамата и как долго находится за краем синапса. Рецепторной активацией и тем более
эксайтотоксичностью они НЕ являются: для этого нужны кинетика рецепторов и модель Ca2+.

Три независимые величины с разными единицами:
  compute_mass_exposure          - M_extra(t) в молекулах, ∫M_extra dt в молекула·мкс
  compute_local_concentration_exposure
                                 - C(r_receptor, t) в мкМ, время выше порога (мкс),
                                   интеграл превышения (мкМ·мкс)
Порог в мкМ применяется только ко второй величине, к локальной концентрации.
"""
import numpy as np


def _trapezoid_cumulative(y, t):
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    if t.size < 2:
        return np.zeros_like(y)
    seg = 0.5 * (y[1:] + y[:-1]) * np.diff(t)
    return np.concatenate(([0.0], np.cumsum(seg)))


def compute_mass_exposure(r_nm, t_us, shell_molecules, r_edge_nm, mass_threshold=None):
    """
    shell_molecules[i, k] - число молекул в оболочке i в момент k (c·ΔV; в режиме
    "probability" - вероятность, тогда единицы величин ниже безразмерные).
    Возвращает словарь либо None, если зона пуста:
      mass_extra, exposure_cumulative, exposure_total (∫M_extra dt, молекула·мкс),
      peak_mass_extra, peak_time; при mass_threshold (в тех же единицах, что M_extra) -
      time_above_mass_threshold, first_time_above_mass_threshold.
    """
    r = np.asarray(r_nm, float)
    t = np.asarray(t_us, float)
    q = np.asarray(shell_molecules, float)
    if q.size == 0 or r.size == 0:
        return None
    mask = r >= float(r_edge_nm)
    if not np.any(mask):
        return None
    mass_extra = q[mask, :].sum(axis=0)
    cum = _trapezoid_cumulative(mass_extra, t)
    k = int(np.argmax(mass_extra))
    out = {
        "r_edge_nm": float(r_edge_nm),
        "mass_extra": mass_extra,
        "exposure_cumulative": cum,
        "exposure_total": float(cum[-1]) if cum.size else 0.0,
        "peak_mass_extra": float(mass_extra[k]),
        "peak_time": float(t[k]),
    }
    if mass_threshold is not None:
        above = mass_extra >= float(mass_threshold)
        both = above[:-1] & above[1:]
        out["mass_threshold"] = float(mass_threshold)
        out["time_above_mass_threshold"] = float(np.sum(np.diff(t)[both])) if t.size > 1 else 0.0
        out["first_time_above_mass_threshold"] = float(t[int(np.argmax(above))]) if np.any(above) else None
    return out


def compute_local_concentration_exposure(c_uM, r_nm, t_us, r_receptor_nm, threshold_uM):
    """
    Локальная концентрация в точке рецепторной зоны (ближайшая ячейка к r_receptor_nm).
    c_uM[i, k] - концентрация в мкМ. Возвращает словарь:
      c_receptor_uM(t), peak_uM, peak_time_us,
      time_above_threshold_us - суммарное время c >= threshold (линейная интерполяция пересечений),
      excess_integral_uM_us   - ∫ max(c − threshold, 0) dt,
      first_time_above_threshold_us.
    """
    c = np.asarray(c_uM, float)
    r = np.asarray(r_nm, float)
    t = np.asarray(t_us, float)
    i = int(np.argmin(np.abs(r - float(r_receptor_nm))))
    y = c[i, :]
    thr = float(threshold_uM)
    time_above = 0.0
    excess = 0.0
    for k in range(t.size - 1):
        y0, y1, dt = y[k], y[k + 1], t[k + 1] - t[k]
        if y0 >= thr and y1 >= thr:
            time_above += dt
            excess += 0.5 * (y0 + y1 - 2.0 * thr) * dt
        elif y0 >= thr or y1 >= thr:
            hi, lo = (y0, y1) if y0 >= y1 else (y1, y0)
            frac = (hi - thr) / (hi - lo)          # доля шага, где c выше порога
            time_above += frac * dt
            excess += 0.5 * (hi - thr) * frac * dt
    kp = int(np.argmax(y))
    above = y >= thr
    return {
        "r_receptor_nm": float(r[i]),
        "c_receptor_uM": y,
        "peak_uM": float(y[kp]),
        "peak_time_us": float(t[kp]),
        "threshold_uM": thr,
        "time_above_threshold_us": float(time_above),
        "excess_integral_uM_us": float(excess),
        "first_time_above_threshold_us": float(t[int(np.argmax(above))]) if np.any(above) else None,
    }
