import unittest

import pandas as pd

import fundingutils


class MinMatchingFloorTests(unittest.TestCase):
    def setUp(self):
        self.donations = pd.DataFrame(
            {
                "alpha": [1.0, 1.0, 0.0],
                "beta": [0.0, 0.0, 4.0],
                "gamma": [0.0, 0.0, 9.0],
            },
            index=["donor1", "donor2", "donor3"],
        )

    def _matches(self, **kwargs):
        result = fundingutils.get_qf_matching(
            "QF",
            self.donations,
            kwargs.pop("matching_cap_percent", 100),
            kwargs.pop("matching_amount", 300),
            **kwargs,
        )
        return result.set_index("project_name")["matching_amount"]

    def test_floor_zero_preserves_existing_qf_distribution(self):
        matches = self._matches(min_matching_floor=0)

        self.assertAlmostEqual(matches.sum(), 300.0)
        self.assertAlmostEqual(matches["alpha"], 300.0)
        self.assertAlmostEqual(matches["beta"], 0.0)
        self.assertAlmostEqual(matches["gamma"], 0.0)

    def test_floor_is_reserved_before_residual_qf(self):
        matches = self._matches(min_matching_floor=25)

        self.assertAlmostEqual(matches.sum(), 300.0)
        self.assertAlmostEqual(matches["alpha"], 250.0)
        self.assertAlmostEqual(matches["beta"], 25.0)
        self.assertAlmostEqual(matches["gamma"], 25.0)

    def test_cap_applies_after_floor_and_redistributes_excess(self):
        matches = self._matches(min_matching_floor=25, matching_cap_percent=50)

        self.assertAlmostEqual(matches.sum(), 300.0)
        self.assertAlmostEqual(matches["alpha"], 150.0)
        self.assertAlmostEqual(matches["beta"], 75.0)
        self.assertAlmostEqual(matches["gamma"], 75.0)

    def test_cap_below_floor_keeps_total_within_pool(self):
        matches = self._matches(min_matching_floor=50, matching_cap_percent=10)

        self.assertLessEqual(matches.sum(), 300.0)
        self.assertAlmostEqual(matches["alpha"], 30.0)
        self.assertAlmostEqual(matches["beta"], 30.0)
        self.assertAlmostEqual(matches["gamma"], 30.0)

    def test_insufficient_pool_scales_effective_floor_evenly(self):
        matches = self._matches(min_matching_floor=200, matching_amount=300)

        self.assertAlmostEqual(matches.sum(), 300.0)
        self.assertAlmostEqual(matches["alpha"], 100.0)
        self.assertAlmostEqual(matches["beta"], 100.0)
        self.assertAlmostEqual(matches["gamma"], 100.0)


if __name__ == "__main__":
    unittest.main()
