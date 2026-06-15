# -*- coding: utf-8 -*-
"""
solver/geometry.py - Геометрические меры для радиальных сеток.

Поддерживаются: spherical (3D) и cylindrical (2D).
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
    }
    if mode not in aliases:
        raise ValueError(f"Неизвестный тип геометрии: {mode}")
    return aliases[mode]


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
    return r_faces


def geometry_report_label(mode=None):
    kind = normalize_geometry_mode(mode)
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
