# -*- coding: utf-8 -*-
"""
tests/test_uptake_physical.py - этап 2: насыщаемый захват -Vmax·c/(Km+c) без обрезки отрицательных значений.

Запуск: python -m unittest tests.test_uptake_physical -v
"""
import math
import unittest

import numpy as np

from tests._physical_helpers import run_physical
from solver.fp_solver import implicit_saturable_uptake
from utils.units import MOLECULES_PER_NM3_TO_UM


class TestImplicitReactionStep(unittest.TestCase):
    def test_identity_holds_cell_by_cell(self):
        rng = np.random.default_rng(0)
        c_star = np.concatenate([[0.0], rng.uniform(0, 5e-3, 200), rng.uniform(0, 1e-6, 50)])
        tau = rng.uniform(0, 2e-3, c_star.size)
        Km = 1.2e-5
        c = implicit_saturable_uptake(c_star, tau, Km)
        np.testing.assert_allclose(c + tau * c / (Km + c), c_star, rtol=1e-10, atol=1e-18)
        self.assertTrue(np.all(c >= 0.0))
        self.assertTrue(np.all(c <= c_star + 1e-18))

    def test_zero_rate_returns_input(self):
        c_star = np.array([0.0, 1e-7, 3e-3])
        np.testing.assert_allclose(implicit_saturable_uptake(c_star, 0.0, 1e-5), c_star, rtol=1e-14)

    def test_positivity_for_huge_rate(self):
        # dt·Vmax/Km = 1e4: явная схема дала бы отрицательные значения, неявная - нет
        c = implicit_saturable_uptake(np.array([1e-9, 1e-5, 1.0]), 1e-1, 1e-5)
        self.assertTrue(np.all(c >= 0.0))


class TestUptakeLimits(unittest.TestCase):
    """Перемешанный объём: маленькая область с no-flux и быстрой диффузией, захват всюду одинаковый."""

    def _well_mixed(self, N, Vmax, T, dt=0.5):
        # a, b малы -> 3D-нейропиль; R = 150 нм, D = 2000 нм²/мкс -> перемешивание за ~10 мкс
        return run_physical(D_cleft=2000.0, D_pm=2000.0, a=0.5, b=1.0, R=150.0, dr=1.0, dt=dt, T=T,
                            N=N, sigma=40.0, window=150.0, boundary="no_flux", uptake="saturable",
                            Vmax_cleft_uM=Vmax, Vmax_pm_uM=Vmax, frames=100)

    def test_linear_limit_c_much_less_than_Km(self):
        Vmax, Km = 0.01, 20.0                 # мкМ/мкс, мкМ -> k = Vmax/Km = 5e-4 1/мкс
        c, r, t, info = self._well_mixed(N=0.1, Vmax=Vmax, T=2000.0)   # континуальная модель: N - среднее число молекул
        mb = info["mass_budget"]
        c_uM0 = mb["released"][0] / (0.2 * 4 / 3 * math.pi * 150.0**3) * MOLECULES_PER_NM3_TO_UM
        self.assertLess(c_uM0, 0.01 * Km)
        k_fit = -np.polyfit(mb["t"][20:], np.log(mb["free"][20:]), 1)[0]
        self.assertAlmostEqual(k_fit / (Vmax / Km), 1.0, delta=0.01)

    def test_saturated_limit_c_much_greater_than_Km(self):
        Vmax = 0.01
        V = 0.2 * 4 / 3 * math.pi * 150.0**3             # объём жидкости, нм³
        N = 1e3 * 20.0 / MOLECULES_PER_NM3_TO_UM * V      # c0 = 1000·Km
        c, r, t, info = self._well_mixed(N=N, Vmax=Vmax, T=400.0)
        mb = info["mass_budget"]
        rate = -np.polyfit(mb["t"][10:], mb["free"][10:], 1)[0]           # молекул/мкс
        expected = Vmax / MOLECULES_PER_NM3_TO_UM * V                     # Vmax·V в молекулах/мкс
        self.assertAlmostEqual(rate / expected, 1.0, delta=0.01)

    def test_dt_convergence_first_order(self):
        Vmax = 0.05
        V = 0.2 * 4 / 3 * math.pi * 150.0**3
        N = 20.0 / MOLECULES_PER_NM3_TO_UM * V             # c0 = Km: самый нелинейный режим
        ends = [self._well_mixed(N=N, Vmax=Vmax, T=400.0, dt=dt)[3]["mass_budget"]["free"][-1]
                for dt in (2.0, 1.0, 0.5, 0.25)]
        e = np.abs(np.diff(ends))
        self.assertLess(e[1], 0.7 * e[0])
        self.assertLess(e[2], 0.7 * e[1])


if __name__ == "__main__":
    unittest.main()
