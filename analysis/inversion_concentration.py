# -*- coding: utf-8 -*-
"""
analysis/inversion_concentration.py - инверсия ЛОКАЛЬНОЙ КОНЦЕНТРАЦИИ (этап 3 ТЗ от 13.10.2026).

Инверсия = в какой-то момент t есть r2 > r1 с c(r2, t) > c(r1, t), то есть dc/dr > 0,
и разница ΔC = c(r2) − c(r1) больше абсолютного порога (в мкМ). Это отдельная величина:
  - смещение максимума профиля с r = 0 (peak_shift) и
  - перераспределение МАССЫ между областями (рост молекул в наружных оболочках
    из-за роста их объёма) инверсией концентрации не считаются.

Функция работает с c в мкМ (после перевода единиц), r в нм, t в мкс.
"""
import numpy as np


def concentration_inversion(c_uM, r_nm, t_us, abs_threshold_uM=1e-3, rel_threshold=1e-9):
    """
    Возвращает словарь:
      found            - есть ли положительный градиент выше порогов
      t_first_us       - первый момент, когда он появился
      r_left_nm, r_right_nm - концы самого большого участка роста c(r) в этот момент
      delta_c_uM       - c(r_right) − c(r_left) на этом участке (максимум по времени)
      ratio_c          - c(r_right)/c(r_left)
      duration_us      - сколько времени инверсия существует (сумма шагов сохранения)
      max_positive_gradient_uM_per_nm
      peak_shift_nm    - максимум по времени положения пика профиля (отдельная характеристика)
    abs_threshold_uM: ΔC меньше этого считается неразличимым (физический порог);
    rel_threshold: доля от max c в момент t (численный порог).
    """
    c = np.asarray(c_uM, float); r = np.asarray(r_nm, float); t = np.asarray(t_us, float)
    out = {"found": False, "t_first_us": None, "r_left_nm": None, "r_right_nm": None,
           "delta_c_uM": 0.0, "ratio_c": None, "duration_us": 0.0,
           "max_positive_gradient_uM_per_nm": 0.0,
           "peak_shift_nm": float(np.max(r[np.argmax(c, axis=0)])) if c.size else 0.0}
    dt = np.diff(t, prepend=t[0])
    for k in range(c.shape[1]):
        col = c[:, k]
        thr = max(abs_threshold_uM, rel_threshold * float(col.max()))
        # самый большой рост: c(j) - min_{i<j} c(i)
        run_min = np.minimum.accumulate(col)
        rise = col - run_min
        j = int(np.argmax(rise))
        if rise[j] > thr:
            i = int(np.argmax(col[:j + 1] == run_min[j]))
            grad = np.max(np.diff(col) / np.diff(r))
            if not out["found"]:
                out.update(found=True, t_first_us=float(t[k]))
            out["duration_us"] += float(dt[k])
            if rise[j] > out["delta_c_uM"]:
                out.update(r_left_nm=float(r[i]), r_right_nm=float(r[j]), delta_c_uM=float(rise[j]),
                           ratio_c=float(col[j] / max(col[i], 1e-300)))
            out["max_positive_gradient_uM_per_nm"] = max(out["max_positive_gradient_uM_per_nm"], float(grad))
    return out
