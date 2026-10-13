# -*- coding: utf-8 -*-
"""
tests/test_inversion_physics.py - этап 3: контрольные режимы монотонности (docs/monotonicity_note.md).

Утверждение: при Ω = 0, гладких D(r) > 0 и A(r) > 0, захвате, не убывающем по r,
и убывающем начальном профиле c(r, t) остаётся невозрастающей по r - инверсии нет.
Контроль 4: дрейф наружу (Ω > 0, параметры курсовой) - инверсия математически возможна.

Запуск: python -m unittest tests.test_inversion_physics -v
"""
import unittest

import numpy as np

from tests._physical_helpers import run_physical
from analysis.inversion_concentration import concentration_inversion
from utils.units import solver_to_uM_factor


def _inv(geometry="barbour_mixed", **kw):
    c, r, t, info = run_physical(geometry=geometry, **kw)
    c_uM = c * solver_to_uM_factor(0.2, geometry, 20.0)
    return concentration_inversion(c_uM, r, t, abs_threshold_uM=1e-6), info


class TestMonotonicityControls(unittest.TestCase):
    def test_1_uniform_diffusion_no_uptake(self):
        res, _ = _inv(D_cleft=320.0, D_pm=320.0)
        self.assertFalse(res["found"], res)

    def test_2_variable_D_and_barbour_geometry(self):
        for Dc in (10.0, 760.0):              # D в щели ниже и выше, чем снаружи
            res, _ = _inv(D_cleft=Dc, D_pm=320.0)
            self.assertFalse(res["found"], (Dc, res))

    def test_3_uptake_nondecreasing_with_r(self):
        res, info = _inv(D_cleft=10.0, D_pm=320.0, uptake="saturable", Vmax_pm_uM=0.01)
        self.assertFalse(res["found"], res)
        self.assertGreater(info["mass_budget"]["uptaken"][-1], 0.0)

    def test_4_outward_drift_can_invert(self):
        """Курсовая: сфера, D 10/0,5, Ω 0,05/0,15 наружу, источник σ = 500 нм в окне 0-500 нм."""
        res, _ = _inv(geometry="spherical", D_cleft=10.0, D_pm=0.5, Omega_cleft=0.05, Omega_pm=0.15,
                      sigma=500.0, window=500.0, R=1000.0, dr=0.5, dt=0.5, T=1000.0)
        self.assertTrue(res["found"], res)

    def test_4b_same_without_drift_has_no_inversion(self):
        res, _ = _inv(geometry="spherical", D_cleft=10.0, D_pm=0.5,
                      sigma=500.0, window=500.0, R=1000.0, dr=0.5, dt=0.5, T=1000.0)
        self.assertFalse(res["found"], res)

    def test_5_inward_drift_from_unified_fit_has_no_inversion(self):
        res, _ = _inv(D_cleft=16.7, D_pm=320.0, Omega_cleft=-0.084, Omega_pm=-0.25)
        self.assertFalse(res["found"], res)


if __name__ == "__main__":
    unittest.main()
