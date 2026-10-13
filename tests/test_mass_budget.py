# -*- coding: utf-8 -*-
"""
tests/test_mass_budget.py - этап 1: замкнутый баланс числа молекул в геометрии Barbour.

released = free + uptaken + escaped (+ ошибка) на каждом сохранённом кадре.
Запуск: python -m unittest tests.test_mass_budget -v
"""
import unittest

import numpy as np

from tests._physical_helpers import run_physical

TOL = 1e-6   # допуск относительной ошибки баланса (инженерный, ТЗ этап 1)


def _rel_balance(mb):
    return float(np.max(np.abs(mb["balance_error"])) / mb["released"][-1])


class TestMassBudget(unittest.TestCase):
    def test_no_flux_no_uptake_conserves_N(self):
        c, r, t, info = run_physical(boundary="no_flux", T=2000.0)
        mb = info["mass_budget"]
        self.assertAlmostEqual(mb["released"][-1], 3000.0, places=6)
        np.testing.assert_allclose(mb["free"], 3000.0, rtol=TOL)
        self.assertLess(_rel_balance(mb), TOL)
        self.assertEqual(mb["uptaken"][-1], 0.0)
        self.assertEqual(mb["escaped"][-1], 0.0)

    def test_dirichlet_mass_decreases_and_balance_closes(self):
        c, r, t, info = run_physical(boundary="dirichlet", R=1500.0, T=3000.0)
        mb = info["mass_budget"]
        self.assertTrue(np.all(np.diff(mb["free"]) <= 1e-12 * mb["released"][-1]))
        self.assertGreater(mb["escaped"][-1], 0.0)
        self.assertLess(_rel_balance(mb), TOL)

    def test_balance_with_uptake_and_boundary(self):
        c, r, t, info = run_physical(boundary="dirichlet", uptake="saturable", Vmax_pm_uM=0.01, T=3000.0)
        mb = info["mass_budget"]
        self.assertGreater(mb["uptaken"][-1], 0.0)
        self.assertLess(_rel_balance(mb), TOL)
        self.assertGreaterEqual(mb["min_c_preclip"], 0.0)

    def test_no_negative_concentration_without_clipping(self):
        c, r, t, info = run_physical(boundary="dirichlet", uptake="saturable", Vmax_pm_uM=0.01, clip=False)
        self.assertGreaterEqual(float(c.min()), 0.0)
        self.assertGreaterEqual(info["mass_budget"]["min_c_preclip"], 0.0)


if __name__ == "__main__":
    unittest.main()
