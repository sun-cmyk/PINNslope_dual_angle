# SEAM dual-angle interpolation data

`seam_shot_15Hz.npy` is copied from
`DeepWave-KAUST/PINNs_signal_separation-pub` at commit `227759d`, where it is
stored as `data/interpolation/synth/seam_shot_15Hz.npy`.

The source array has shape `(900, 300)` in `(t, x)` order. The dual-angle
notebook uses the `(x=80:260, t=400:850)` crop, keeps every third spatial
trace for the data term, and holds the remaining traces out for interpolation
metrics.
