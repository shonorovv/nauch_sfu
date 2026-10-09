# -*- coding: utf-8 -*-
"""
tests/test_units.py - Проверка перевода единиц решателя в мкМ.

Запуск: python -m unittest tests.test_units
"""
import math
import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.units import (  # noqa: E402
    MOLECULES_PER_NM3_TO_UM, solver_to_uM, uM_to_solver, uptake_to_solver_units,
)
from solver.geometry import radial_cell_measures  # noqa: E402


class UnitsTest(unittest.TestCase):
    def test_one_molecule_per_nm3_is_1p66_M(self):
        # 1 молекула/нм³ = 1e24/N_A моль/л ≈ 1.66 М = 1.66e6 мкМ
        self.assertAlmostEqual(MOLECULES_PER_NM3_TO_UM / 1e6, 1.6605, places=3)

    def test_roundtrip(self):
        c = np.array([0.0, 0.25, 1000.0])
        back = solver_to_uM(uM_to_solver(c, 0.2), 0.2)
        np.testing.assert_allclose(back, c)

    def test_uniform_ball_concentration(self):
        # N молекул равномерно в шаре радиуса R при доле α:
        # C = N / (α * 4/3 π R³) молекул/нм³ -> в мкМ.
        N, R, alpha, dr = 3000.0, 200.0, 0.2, 1.0
        r = (np.arange(int(R / dr)) + 0.5) * dr
        dV = radial_cell_measures(r, dr, "spherical")      # без 4π
        c_solver = np.full_like(r, N / dV.sum())            # масса решателя = N
        expected_uM = N / (alpha * 4.0 / 3.0 * math.pi * R**3) * MOLECULES_PER_NM3_TO_UM
        np.testing.assert_allclose(solver_to_uM(c_solver, alpha), expected_uM, rtol=1e-9)

    def test_uptake_units_scale_together(self):
        # Km и Vmax делятся на один множитель -> отношение Vmax/Km сохраняется
        Km_s, _, Vp_s = uptake_to_solver_units(20.0, 0.0, 0.01, 0.2)
        self.assertAlmostEqual(Vp_s / Km_s, 0.01 / 20.0)


if __name__ == "__main__":
    unittest.main()
