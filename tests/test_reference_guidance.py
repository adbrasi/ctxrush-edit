import unittest

import torch

from reference_guidance import (
    full_reference_guidance,
    mix_reference,
    reference_gain_map,
    runner_reference_guidance,
)


class ReferenceGuidanceMathTest(unittest.TestCase):
    def setUp(self):
        self.no_ref = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        self.full = torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]])

    def test_reference_mix_endpoints_and_extrapolation(self):
        self.assertTrue(torch.equal(mix_reference(self.no_ref, self.full, 0.0), self.no_ref))
        self.assertTrue(torch.equal(mix_reference(self.no_ref, self.full, 1.0), self.full))
        expected = self.no_ref + 2.0 * (self.full - self.no_ref)
        self.assertTrue(torch.equal(mix_reference(self.no_ref, self.full, 2.0), expected))

    def test_soft_spatial_mask_is_pixelwise(self):
        mask = torch.tensor([[[0.0, 0.25], [0.5, 1.0]]])
        gain = reference_gain_map(
            self.no_ref,
            2.0,
            mask=mask,
            outside_guidance=0.0,
        )
        expected_gain = torch.tensor([[[[0.0, 0.5], [1.0, 2.0]]]])
        self.assertTrue(torch.equal(gain, expected_gain))
        expected = self.no_ref + expected_gain * (self.full - self.no_ref)
        self.assertTrue(torch.equal(mix_reference(self.no_ref, self.full, gain), expected))

    def test_mask_is_resized_and_broadcast_over_channels(self):
        prediction = torch.zeros(2, 3, 4, 4)
        gain = reference_gain_map(
            prediction,
            1.0,
            mask=torch.ones(1, 2, 2),
            outside_guidance=0.0,
        )
        self.assertEqual(tuple(gain.shape), (2, 1, 4, 4))
        self.assertTrue(torch.equal(gain, torch.ones_like(gain)))

    def test_mask_broadcasts_over_video_time(self):
        prediction = torch.zeros(1, 16, 3, 4, 4)
        gain = reference_gain_map(
            prediction,
            1.5,
            mask=torch.ones(1, 2, 2),
            outside_guidance=0.25,
        )
        self.assertEqual(tuple(gain.shape), (1, 1, 1, 4, 4))
        self.assertTrue(torch.equal(gain, torch.full_like(gain, 1.5)))

    def test_full_reference_four_corner_invariants(self):
        u = torch.full((1, 1, 1, 1), 1.0)
        t = torch.full_like(u, 3.0)
        r = torch.full_like(u, 5.0)
        c = torch.full_like(u, 11.0)
        text_cfg = 1.7

        at_zero = full_reference_guidance(u, t, r, c, text_cfg, 0.0)
        expected_zero = u + text_cfg * (t - u)
        self.assertTrue(torch.allclose(at_zero, expected_zero))

        at_one = full_reference_guidance(u, t, r, c, text_cfg, 1.0)
        expected_one = r + text_cfg * (c - r)
        self.assertTrue(torch.allclose(at_one, expected_one))

        text_one = full_reference_guidance(u, t, r, c, 1.0, 0.35)
        expected_text_one = t + 0.35 * (c - t)
        self.assertTrue(torch.allclose(text_one, expected_text_one))

    def test_full_reference_spatial_map_selects_corners(self):
        u = torch.zeros(1, 1, 1, 2)
        t = torch.full_like(u, 2.0)
        r = torch.full_like(u, 10.0)
        c = torch.full_like(u, 14.0)
        gain = torch.tensor([[[[0.0, 1.0]]]])
        result = full_reference_guidance(u, t, r, c, 1.0, gain)
        self.assertTrue(torch.equal(result, torch.tensor([[[[2.0, 14.0]]]])))

    def test_runner_formula_matches_documented_cases(self):
        n0 = torch.full((1, 1, 1, 1), 1.0)
        p0 = torch.full_like(n0, 2.0)
        n1 = torch.full_like(n0, 5.0)
        p1 = torch.full_like(n0, 11.0)

        text_one = runner_reference_guidance(n0, p0, n1, p1, 1.0, 0.25)
        self.assertTrue(torch.allclose(text_one, p0 + 0.25 * (p1 - p0)))

        text_cfg = 3.0
        general = runner_reference_guidance(n0, p0, n1, p1, text_cfg, 0.25)
        expected = n0 + 0.25 * (n1 - n0) + text_cfg * (p1 - n1)
        self.assertTrue(torch.allclose(general, expected))

        canonical = runner_reference_guidance(n0, p0, n1, p1, text_cfg, 1.0)
        expected_canonical = n1 + text_cfg * (p1 - n1)
        self.assertTrue(torch.allclose(canonical, expected_canonical))


if __name__ == "__main__":
    unittest.main()
