# -*- coding: utf-8 -*-
"""
analysis/excitotoxicity.py - Метрика эксайтотоксической нагрузки.

Курсовая (Шоноров, 2026) обосновывает актуальность модели через спилловер
глутамата к внесинаптическим NMDA-рецепторам и последующую эксайтотоксичность
(избыточный вход Ca2+ -> каскад гибели нейрона). До сих пор в коде эта связь
никак не измерялась: индекс инверсии I(t) = max(p_вне)/max(p_внутри) -
чисто геометрическая характеристика формы профиля, не привязанная к тому,
сколько медиатора и как долго реально присутствует за краем синапса.

compute_extrasynaptic_exposure добавляет физиологически более прямую
величину: массу медиатора за заданным радиусом (краем синапса) во времени
и её интеграл по времени - "доза x время". Это ближе к механизму, которым
объясняют эксайтотоксичность: устойчивая, а не обязательно пиковая,
активация внесинаптических NMDA-рецепторов запускает каскад гибели клетки
(Hardingham & Bading, Nat Rev Neurosci, 2010) - поэтому накопленная
экспозиция информативнее одномоментного максимума.
"""

import numpy as np


def compute_extrasynaptic_exposure(
    r_values,
    t_values,
    shell_probabilities,
    extrasynaptic_radius,
    concentration_threshold=None,
):
    """
    Возвращает dict с временным рядом массы за extrasynaptic_radius и её
    накопленной во времени экспозицией, либо None, если зона/данные пусты.

    - mass_extra(t)         - масса (сумма p*dV по оболочкам) при r >= extrasynaptic_radius
    - exposure_cumulative(t) - интеграл mass_extra по времени (метод трапеций), нарастающим итогом
    - exposure_total        - полная накопленная экспозиция за весь расчёт
    - peak_mass_extra, peak_time - момент и величина максимума mass_extra
    - concentration_threshold, time_above_threshold, first_time_above_threshold -
      заполняются только если передан порог (физиологический порог активации
      внесинаптических NMDA-R сейчас не откалиброван по литературе - см. TODO
      в config.py при excitotoxicity_concentration_threshold).
    """
    r_values = np.asarray(r_values, dtype=float)
    t_values = np.asarray(t_values, dtype=float)
    shell_probabilities = np.asarray(shell_probabilities, dtype=float)
    if shell_probabilities.size == 0 or r_values.size == 0:
        return None
    mask = r_values >= float(extrasynaptic_radius)
    if not np.any(mask):
        return None

    mass_extra = np.sum(shell_probabilities[mask, :], axis=0)

    if t_values.size > 1:
        segment_integrals = 0.5 * (mass_extra[1:] + mass_extra[:-1]) * np.diff(t_values)
        exposure_cumulative = np.concatenate(([0.0], np.cumsum(segment_integrals)))
    else:
        exposure_cumulative = np.zeros_like(mass_extra)

    peak_idx = int(np.argmax(mass_extra))
    result = {
        "extrasynaptic_radius": float(extrasynaptic_radius),
        "mass_extra": mass_extra,
        "exposure_cumulative": exposure_cumulative,
        "exposure_total": float(exposure_cumulative[-1]) if exposure_cumulative.size else 0.0,
        "peak_mass_extra": float(mass_extra[peak_idx]),
        "peak_time": float(t_values[peak_idx]),
    }

    if concentration_threshold is not None:
        above = mass_extra >= float(concentration_threshold)
        if t_values.size > 1:
            dt_local = np.diff(t_values)
            both_above = above[:-1] & above[1:]
            time_above = float(np.sum(dt_local[both_above]))
        else:
            time_above = 0.0
        result["concentration_threshold"] = float(concentration_threshold)
        result["time_above_threshold"] = time_above
        result["first_time_above_threshold"] = (
            float(t_values[int(np.argmax(above))]) if np.any(above) else None
        )

    return result
