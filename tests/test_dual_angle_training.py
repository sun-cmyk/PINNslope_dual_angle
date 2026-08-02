import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pinnslope.dual_angle import DualAnglePINN
from pinnslope.dual_angle_training import (
    export_predictions,
    load_gather,
    normalize_slope_interval,
    prepare_interpolation_data,
)


class DualAngleTrainingTests(unittest.TestCase):
    def test_physical_slope_conversion_matches_normalized_coordinates(self):
        bounds = normalize_slope_interval(
            (0.001, 0.002), units="physical", x_extent=1000.0, t_extent=2.0
        )
        self.assertEqual(bounds, (0.5, 1.0))

    def test_load_gather_applies_orientation_and_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gather.npy"
            np.save(path, np.arange(30).reshape(5, 6))
            gather = load_gather(
                str(path), orientation="t-x", x_slice=(1, 5), t_slice=(1, 4)
            )

        self.assertEqual(gather.shape, (4, 3))
        np.testing.assert_array_equal(gather, np.arange(30).reshape(5, 6).T[1:5, 1:4])

    def test_prepare_interpolation_data_selects_complete_traces(self):
        gather = np.arange(35, dtype=np.float32).reshape(5, 7)
        data = prepare_interpolation_data(gather, trace_step=2, device="cpu")

        self.assertEqual(tuple(data["full_coordinates"].shape), (35, 2))
        self.assertEqual(tuple(data["observed_coordinates"].shape), (21, 2))
        self.assertEqual(data["observed_trace_indices"].tolist(), [0, 2, 4])
        np.testing.assert_array_equal(
            data["observed_values"].numpy().reshape(3, 7), gather[[0, 2, 4]]
        )
        self.assertEqual(data["missing_trace_mask"].tolist(), [False, True, False, True, False])

    def test_export_predictions_writes_all_fields_and_metrics(self):
        model = DualAnglePINN(
            wavefield_hidden=(8,),
            slope_hidden=(4,),
            positional_encoding=(2, 2, 2),
        )
        gather = np.zeros((3, 4), dtype=np.float32)
        data = prepare_interpolation_data(gather, trace_step=2, device="cpu")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metrics = export_predictions(
                output,
                model,
                data["full_coordinates"],
                data["target"],
                data["missing_trace_mask"],
                gather.shape,
                inference_batch_size=5,
            )
            names = {path.name for path in output.iterdir()}

        self.assertEqual(
            names,
            {
                "metrics.json",
                "phi_1.npy",
                "phi_2.npy",
                "reconstruction.npy",
                "slope_1.npy",
                "slope_2.npy",
            },
        )
        self.assertIn("missing_mse", metrics)


if __name__ == "__main__":
    unittest.main()
