# -*- coding: utf-8 -*-
"""
solver/geometry.py - Геометрические меры для радиальных сеток.

Поддерживаются: spherical (3D), cylindrical (2D) и cleft_disk (щель + 3D снаружи).

cleft_disk (добавлено 12.10.2026, единая модель):
    при r <= a_c - синаптическая щель, плоский слой высотой h:
        объём слоя dV = π·h·(r_out² − r_in²), площадь грани A = 2π·h·r
    при r > a_c - внеклеточное пространство вокруг синапса, доля объёма α:
        dV = α·(4π/3)·(r_out³ − r_in³), A = 4π·α·r²
    a_c = config.cleft_radius_nm, h = config.cleft_height_nm,
    α = config.extracellular_volume_fraction.
    В этом режиме меры - настоящие объёмы жидкости (нм³) и площади (нм²),
    поэтому c решателя - молекулы на нм³ жидкости, без множителей 4π и α.
    Режимы spherical и cylindrical работают как раньше.
"""

import numpy as np
import config


def normalize_geometry_mode(mode=None):
    """Приводит строку геометрии к каноническому виду."""
    mode = config.geometry_mode if mode is None else mode
    mode = str(mode).lower()
    aliases = {
        "radial": "cylindrical",
        "polar": "cylindrical",
        "cylindrical": "cylindrical",
        "2d": "cylindrical",
        "sphere": "spherical",
        "spherical": "spherical",
        "3d": "spherical",
        "cleft_disk": "cleft_disk",
        "disk": "cleft_disk",
        "cleft": "cleft_disk",
    }
    if mode not in aliases:
        raise ValueError(f"Неизвестный тип геометрии: {mode}")
    return aliases[mode]


def _cleft_params():
    """Радиус щели, её высота и доля внеклеточного объёма из config."""
    a_c = float(getattr(config, "cleft_radius_nm", config.a))
    h = float(getattr(config, "cleft_height_nm", 20.0))
    alpha = float(getattr(config, "extracellular_volume_fraction", 0.2))
    return a_c, h, alpha


def radial_cell_measures(r_values, dr, mode=None):
    """
    Объём радиальных ячеек (без 4pi/2pi).

    spherical:    (r_outer^3 - r_inner^3) / 3
    cylindrical:  (r_outer^2 - r_inner^2) / 2
    """
    kind = normalize_geometry_mode(mode)
    r_values = np.asarray(r_values, dtype=float)
    dr = float(dr)
    r_inner = np.maximum(r_values - 0.5 * dr, 0.0)
    r_outer = np.maximum(r_values + 0.5 * dr, 0.0)
    if kind == "spherical":
        return (r_outer**3 - r_inner**3) / 3.0
    if kind == "cleft_disk":
        a_c, h, alpha = _cleft_params()
        r1 = np.minimum(r_inner, a_c); r2 = np.minimum(r_outer, a_c)
        inside = np.pi * h * (r2**2 - r1**2)
        r1o = np.maximum(r_inner, a_c); r2o = np.maximum(r_outer, a_c)
        outside = alpha * 4.0 * np.pi / 3.0 * (r2o**3 - r1o**3)
        return inside + outside
    return 0.5 * (r_outer**2 - r_inner**2)


def radial_face_measures(r_faces, mode=None):
    """
    Площадь радиальных граней (без 4pi/2pi).

    spherical:    r^2
    cylindrical:  r
    """
    kind = normalize_geometry_mode(mode)
    r_faces = np.maximum(np.asarray(r_faces, dtype=float), 0.0)
    if kind == "spherical":
        return r_faces**2
    if kind == "cleft_disk":
        a_c, h, alpha = _cleft_params()
        return np.where(r_faces <= a_c + 1e-9, 2.0 * np.pi * h * r_faces,
                        4.0 * np.pi * alpha * r_faces**2)
    return r_faces


def geometry_report_label(mode=None):
    kind = normalize_geometry_mode(mode)
    if kind == "cleft_disk":
        return "щель-диск (dV = π·h·d(r²)) внутри r <= a_c, снаружи сфера с долей объёма α"
    if kind == "spherical":
        return "сферическая симметрия: dV ~ r^2 dr, поток через грань ~ r^2"
    return "радиальная/цилиндрическая симметрия: dV ~ r dr, поток через грань ~ r"


def normalization_report_label(mode=None):
    kind = normalize_geometry_mode(mode)
    if kind == "spherical":
        return "Интеграл p(r,t) * dV_сф, dV_сф=интеграл по ячейке r^2 dr"
    return "Интеграл p(r,t) * dV_рад, dV_рад=интеграл по ячейке r dr"


def shell_probability_report_formula(mode=None):
    kind = normalize_geometry_mode(mode)
    if kind == "spherical":
        return "q = p * интеграл по ячейке r^2 dr"
    return "q = p * интеграл по ячейке r dr"
