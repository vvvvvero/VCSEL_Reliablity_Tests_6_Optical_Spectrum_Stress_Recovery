# The counterfactual temperature-monotonicity anomaly

Across every A/B/C ablation run, the physics-conditioned generator (A) scored
*worst* on counterfactual temperature monotonicity, and the shuffled-physics
control (C) often scored best:

| run | A_full | B_no_physics | C_shuffled |
|---|---|---|---|
| 6-feature model | 0.037 | 0.407 | 0.148 |
| 11-feature, 4 generated | 0.148 | 0.148 | 0.185 |
| 11-feature, 9 generated | **0.000** | 0.333 | 0.370 |

Read at face value this says physics conditioning *destroys* the model's
temperature response — the opposite of the hypothesis, and the kind of result
a reviewer would seize on. It turned out to be a defect in the metric, not in
the model. **`frac_monotone_275_300_325` should not be used or reported.**

## Three independent reasons the metric is invalid

### 1. Its premise is false: the real residual is not monotone in T

The metric rewards a generator whose residual magnitude *increases* from
275 → 300 → 325 °C. Measured on the trained backbone across all 200 devices:

| T | n devices | mean \|residual\| | median |
|---|---|---|---|
| 548 K (275 °C) | 55 | 0.1002 | 0.0512 |
| 573 K (300 °C) | 74 | **0.0655** | 0.0522 |
| 598 K (325 °C) | 71 | 0.2093 | 0.0798 |

The real residual **dips at 300 °C**. It is a U shape, not a ramp. The dip is
not surprising: 300 °C is the reference temperature `T_REF_K`, where the
Arrhenius factors are exactly 1 and the ODE is best conditioned, so the model
fits it best. The rise at 325 °C is the long-standing high-temperature error
(RMSE 0.365 versus 0.140/0.146).

So a generator that faithfully reproduces the data's own temperature profile
is scored as *wrong*, and one that emits a bland monotone ramp is scored as
*right*. The metric inverts the thing it claims to measure.

Shape correlation against the real profile confirms the inversion — the
condition the metric ranks last is the one that best captures the U shape's
falling limb:

| condition | normalised profile | corr. with real shape |
|---|---|---|
| A_full | 1.034 / 0.998 / 0.967 | −0.696 |
| B_no_physics | 0.986 / 0.986 / 1.028 | +0.969 |
| C_shuffled | 0.961 / 1.031 / 1.008 | −0.048 |

B wins the metric by producing a monotone ramp, which is what the metric asks
for and not what the data does.

### 2. It measures a quantity that is ~zero by construction

The metric takes `deltas.mean(dim=0).abs()` — the absolute value of the
*ensemble mean* residual. A well-centred generative model has an ensemble mean
near zero, so this is measuring the residual asymmetry, not its magnitude:

| condition | generated range across T | spread |
|---|---|---|
| A_full | 0.01471 – 0.01573 | 6.7 % |
| B_no_physics | 0.01501 – 0.01566 | 4.3 % |
| C_shuffled | 0.01504 – 0.01615 | 7.1 % |

All three sit at ≈0.015 — an order of magnitude below the real residual
(0.066–0.209) — and vary by only a few percent across a 50 °C span. The
monotone/not-monotone verdict is therefore decided at the third decimal place.

### 3. Its tolerance makes it a near-coin-flip

The test applies `tol = 1e-4` to differences of order 1e-3. With three
temperatures, a device passes only if two independent noisy comparisons both
fall the right way, so the statistic is closer to a biased coin than to a
physical measurement. This explains why it swings between 0.000 and 0.407
across runs whose CRPSS differs by a few percent.

## What to use instead

The question the metric was meant to answer — *does the generator respond to
temperature the way the physics says it should?* — is worth answering. Doing
it properly needs a target that is neither near-zero nor assumed monotone:

1. **Compare against the measured profile**, not a monotone assumption: score
   the correlation (or the RMSE) between the generated per-temperature
   residual profile and the measured one. This is the shape-correlation column
   above, and it can be computed from the artefacts already saved.
2. **Score the ensemble SPREAD**, not the ensemble mean: `deltas.std(dim=0)`
   is O(0.1) and is the quantity Arrhenius scaling actually predicts, which is
   what `arrhenius_trend_loss` already constrains during training.
3. **Use the counterfactual on `sigma` directly.** The Arrhenius-sigma
   generator exposes `Ea_sigma` per feature; comparing the learned values with
   GaN literature is a stronger and more interpretable physics check than any
   monotonicity count.

## Status

`frac_monotone_275_300_325` is left in the code, because removing it would
silently change the saved artefacts of earlier runs, but it is documented here
as invalid and must not appear in any write-up. The A/B/C conclusion rests on
CRPSS, coverage, W1 and MACE, all of which are computed against measured data
rather than an assumed shape.
