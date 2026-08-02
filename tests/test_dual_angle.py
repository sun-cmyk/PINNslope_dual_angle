import math
import unittest

import torch
import torch.nn as nn

from pinnslope.dual_angle import (
    DualAnglePINN,
    dual_angle_loss,
    map_slope_outputs,
    train_dual_angle_step,
)


class AnalyticWavefields(nn.Module):
    def forward(self, coordinates):
        x = coordinates[:, 0]
        t = coordinates[:, 1]
        phi_1 = torch.sin(2.0 * math.pi * (t - 0.4 * x))
        phi_2 = 0.7 * torch.cos(2.0 * math.pi * (t + 0.3 * x))
        return torch.stack((phi_1, phi_2), dim=1)


class ZeroSlopes(nn.Module):
    def forward(self, coordinates):
        return coordinates.new_zeros((coordinates.shape[0], 2))


class DualAngleTests(unittest.TestCase):
    def test_slope_mapping_respects_positive_and_negative_ranges(self):
        raw = torch.tensor([[-100.0, -100.0], [100.0, 100.0], [0.0, 0.0]])
        mapped = map_slope_outputs(
            raw,
            shifts=torch.tensor([0.05, -0.8]),
            scales=torch.tensor([0.75, 0.75]),
        )

        self.assertTrue(torch.all(mapped[:, 0] >= 0.05))
        self.assertTrue(torch.all(mapped[:, 0] <= 0.8))
        self.assertTrue(torch.all(mapped[:, 1] >= -0.8))
        self.assertTrue(torch.all(mapped[:, 1] <= -0.05))

    def test_analytic_plane_waves_have_near_zero_loss(self):
        model = DualAnglePINN(
            wavefield_hidden=(4,),
            slope_hidden=(4,),
            positional_encoding=(2, 2, 2),
            slope_shifts=(0.4, -0.3),
            slope_scales=(0.0, 0.0),
            wavefield_network=AnalyticWavefields(),
            slope_network=ZeroSlopes(),
        )
        grid = torch.rand(64, 2)
        traces = torch.rand(24, 2)
        trace_values = AnalyticWavefields()(traces).sum(dim=1, keepdim=True)

        losses = dual_angle_loss(model, grid, traces, trace_values, data_weight=1.0)

        self.assertLess(losses.physics.item(), 1e-10)
        self.assertLess(losses.data.item(), 1e-10)
        self.assertLess(losses.total.item(), 1e-10)

    def test_training_step_updates_both_networks(self):
        torch.manual_seed(7)
        model = DualAnglePINN(
            wavefield_hidden=(16, 16),
            slope_hidden=(8, 8),
            positional_encoding=(2, 3, 2),
        )
        wavefield_optimizer = torch.optim.Adam(
            model.wavefield_network.parameters(), lr=1e-3
        )
        slope_optimizer = torch.optim.Adam(model.slope_network.parameters(), lr=1e-3)
        grid = torch.rand(32, 2)
        traces = torch.rand(16, 2)
        trace_values = AnalyticWavefields()(traces).sum(dim=1, keepdim=True)
        wavefield_before = [
            parameter.detach().clone()
            for parameter in model.wavefield_network.parameters()
        ]
        slope_before = [
            parameter.detach().clone() for parameter in model.slope_network.parameters()
        ]

        losses = train_dual_angle_step(
            model,
            wavefield_optimizer,
            slope_optimizer,
            grid,
            traces,
            trace_values,
            data_weight=10.0,
        )

        self.assertTrue(torch.isfinite(losses.total))
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(
                    wavefield_before, model.wavefield_network.parameters()
                )
            )
        )
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(slope_before, model.slope_network.parameters())
            )
        )


if __name__ == "__main__":
    unittest.main()
