# Dual-angle interpolation

The implementation follows the double-output slope constraint architecture:

- one positional-encoding wavefield network predicts `phi_1` and `phi_2`;
- one slope network predicts two unconstrained slope outputs;
- sigmoid scaling maps the slope outputs into two user-defined intervals;
- the sum `phi_1 + phi_2` fits the amplitudes on available traces.

For coordinates ordered as `(x, t)`, the two plane-wave residuals are

```text
r_1 = d(phi_1)/dx + sigma_1 * d(phi_1)/dt
r_2 = d(phi_2)/dx + sigma_2 * d(phi_2)/dt
```

The implemented loss is

```text
L = mean(r_1^2) + mean(r_2^2) + lambda * L1(phi_1 + phi_2, observed_data)
```

## Slope constraints

Each slope output uses

```text
sigma_k = sigmoid(raw_k) * c_k + a_k
```

The default values reproduce separate positive and negative normalized slopes:

```text
a = [0, 0]
c = [1, -1]
```

More restrictive intervals can avoid ambiguity around zero. For example,
`a=[0.05, -0.8]` and `c=[0.75, 0.75]` produce `[0.05, 0.8]` and
`[-0.8, -0.05]`.

When physical coordinates are normalized before training, physical slopes must
be converted to the normalized-coordinate convention before choosing these
intervals.

## Smoke test

Run the small crossing-plane-wave interpolation example from the repository
root:

```powershell
python -m examples.dual_angle_interpolation --epochs 200 --device cpu
```

The paper-sized starting configuration is a four-layer 512-unit wavefield
network, a four-layer 8-unit slope network, positional encoding `[8, 32, 2]`,
learning rate `1e-3`, and data weight `1000`.
