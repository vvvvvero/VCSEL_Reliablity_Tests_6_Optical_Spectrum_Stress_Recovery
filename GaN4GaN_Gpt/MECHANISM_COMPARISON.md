# Candidate rate-law comparison — the data prefer SRH

## What was attempted

`23_baseline_vanilla.py` showed the physics prior beats a free vector field by
13.2 % (4 seeds, CI [−16.3, −10.2]). That is *physics beats no physics*, which
is weaker than *the data support THIS physics*. The pipeline assumes
Shockley-Read-Hall kinetics for the reversible trap states; the GaN literature
also argues for power-law, stretched-exponential (KWW) and logarithmic creep.
`25_mechanism_candidates.py` compares the four on equal terms.

## The screening result, at face value

40 epochs, one seed, everything but the reversible rate law held fixed:

| candidate | val loss | test RMSE | t=500h | t=1000h | t=2000h | stopped at |
|---|---|---|---|---|---|---|
| **srh** | **0.0455** | **0.2425** | 0.2051 | 0.2236 | 0.3016 | ran all 40 |
| stretched | 0.1074 | 0.3335 | 0.2941 | 0.3245 | 0.4022 | epoch 34 |
| power | 0.1530 | 0.4226 | 0.4272 | 0.4809 | 0.5594 | epoch 13 |
| log | 0.1483 | 0.4903 | 0.4103 | 0.4242 | 0.5247 | epoch 14 |

SRH wins by a wide margin on every horizon. **This result must not be
reported**, for the reason below.

## Why it is invalid

The losers barely trained. SRH improved steadily (val 0.107 → 0.058 → 0.045);
`power` and `log` never beat their epoch-1 value and early-stopped, and the
learned exponents came back essentially at their initialisation
(0.500 / 0.400 / 0.649 against inits of 0.50 / 0.40 / 0.65).

Measuring the gradients on a single batch explains it:

| candidate | params receiving gradient | worst \|grad\| |
|---|---|---|
| srh | 18 / 18 | 6.7e−03 |
| power | 15 / 21 | **1.7e+16** |
| stretched | 15 / 21 | **2.0e+13** |
| log | 12 / 21 | 5.8e+03, and **all three exponents get `None`** |

Gradients of 10¹³–10¹⁶ are clipped to the 5.0 norm bound every step, so those
parameters move in an essentially arbitrary direction at a fixed magnitude.
`log`'s exponents receive no gradient at all — its rate law does not use the
exponent, so three of its 21 parameters are dead weight and the model is
effectively smaller than the comparison claims.

The forward pass is not the problem: `p·(t+t₀)^(p−1)` is bounded by 2.41 over
the whole time grid at these exponents. The blow-up is in the backward pass
through 88 chained RK4 substeps (8 per interval × 11 intervals), where a modest
per-step Jacobian amplification compounds — the same class of failure the IMEX
integrator was introduced to solve for SRH, and which the alternatives cannot
use because it assumes a *linear autonomous* relaxation.

**So the comparison comes down to: SRH has a stable integrator and the others
do not.** That is a statement about this implementation, not about which
mechanism the devices follow.

## What would make it valid

1. **Stabilise the backward pass.** Options: adjoint sensitivity instead of
   backprop-through-solver; fewer substeps with a stiff-aware step; or
   gradient checkpointing per interval. Until then the three time-dependent
   laws cannot be fitted to convergence.
2. **Give `log` a real free parameter** in place of the unused exponent, or
   drop it from the parameter count so the comparison is like-for-like.
3. **Re-run with matched effort**, judged by whether each candidate's
   validation loss actually plateaus rather than by a fixed epoch budget.

## Standing conclusion

The claim "the data prefer SRH over the alternatives" is **not established**.
What survives from this work is the honest negative: the alternatives could not
be fitted stably under the current solver, so the pipeline's SRH assumption
remains an assumption. The separately-measured result that the physics prior
as a whole beats free-form dynamics (13.2 %, 4 seeds) is unaffected — it never
depended on this comparison.


---

## The fix: every candidate has a closed-form solution

The first attempt failed because the three time-dependent laws were integrated
with an 8-substep RK4 whose backward pass produced gradients of 10^13-10^16.
The remedy is not a better integrator — it is no integrator at all.

All three candidates are **separable**:

    dz/dt = k*f(t)*(1-z)   =>   z(t1) = 1 - (1-z(t0)) * exp(-k * [F(t1)-F(t0)])

and F, the antiderivative of the rate shape, is analytic in every case:

| candidate | rate shape f(t) | antiderivative F(t) |
|---|---|---|
| power, stretched | p·(t+t0)^(p-1) | (t+t0)^p |
| log | 1/(1+t/t0) | t0·ln(1+t/t0) |

Checked against 200k-point quadrature: agreement to **≤ 6e-11** on intervals
from [0,1] h to [1000,2000] h. The stretched law's (1-z)^2 envelope integrates
to a rational form, 1/(1-z1) = 1/(1-z0) + c·ΔF, which is also exact.

This puts the candidates on the same footing as SRH, whose IMEX path was
already an exact update — z1 = z_eq + (z0-z_eq)·exp(-(a+b)·dt). The comparison
now differs only in the rate law, not in solver quality.

**Measured effect on the gradients** (same batch, same initialisation):

| candidate | before (RK4) | after (closed form) |
|---|---|---|
| power | 1.7e+16 | 1.3e-02 |
| stretched | 2.0e+13 | 9.5e-03 |
| log | 5.8e+03 | 7.2e-03 |
| srh (control) | 6.7e-03 | 6.7e-03 |

All now sit at SRH's order of magnitude, so the 5.0 gradient-norm clip no
longer fires every step and the parameters can actually move.

## A second unfairness, also fixed

`log`'s three shape exponents received no gradient at all — its rate law,
A/(t0+t), has no exponent to fit. They were nevertheless counted as
parameters, making log look like a 21-parameter model against SRH's 18. They
are now buffers rather than parameters:

| candidate | trainable dynamics parameters |
|---|---|
| srh | 18 |
| log | 18 (was 21) |
| power, stretched | 21 |

## Re-run with exact updates

40 epochs, one seed, everything but the reversible rate law held fixed.

| candidate | params | val loss | test RMSE | t=500h | t=1000h | t=2000h |
|---|---|---|---|---|---|---|
| **srh** | 18 | **0.0455** | **0.2425** | 0.2051 | 0.2236 | 0.3016 |
| stretched | 21 | 0.0791 | 0.3332 | 0.2529 | 0.3422 | 0.4503 |
| power | 21 | 0.0861 | 0.3316 | 0.2540 | 0.4214 | 0.4266 |
| log | 18 | 0.1305 | 0.4213 | 0.3507 | 0.4710 | 0.6065 |

SRH leads on every horizon, with 3 fewer parameters than power and stretched.

### The fix did what it was supposed to

Comparing against the archived RK4 run isolates the integrator's effect. SRH is
byte-identical, as expected — its path never used the RK4 that was broken:

| candidate | val (RK4) | val (exact) | RMSE (RK4) | RMSE (exact) |
|---|---|---|---|---|
| srh | 0.04546 | 0.04546 | 0.2425 | 0.2425 |
| power | 0.15304 | **0.08611** | 0.4226 | **0.3316** |
| stretched | 0.10744 | **0.07914** | 0.3335 | 0.3332 |
| log | 0.14833 | **0.13049** | 0.4903 | **0.4213** |

Every alternative improved — power's validation loss nearly halved — which
confirms the first comparison was measuring solver stability, not physics.
All four now train properly: no early stops, val loss decreasing monotonically
from epoch 1 through 40.

### Why this still is not a finished result

**Nothing converged.** Every candidate was still improving at epoch 40, and by
a large margin:

| candidate | ep1 | ep20 | ep40 | improvement over the last 20 epochs |
|---|---|---|---|---|
| srh | 0.1066 | 0.0579 | 0.0455 | 21.5 % |
| stretched | 0.1247 | 0.0948 | 0.0791 | 16.5 % |
| power | 0.1274 | 0.0986 | 0.0861 | 12.6 % |
| log | 0.1725 | 0.1446 | 0.1305 | 9.8 % |

A screening budget was chosen deliberately, but it means the ranking reflects
40-epoch *learning speed* as much as final quality. SRH is also improving
fastest, so the gap could widen or narrow with a full 200-epoch budget.

**The exponents barely moved** — 0.009 to 0.019 from their initialisation of
0.50 / 0.40 / 0.65. Gradients now reach them (1e-2 to 1e-3, verified), so this
is not the earlier dead-parameter failure, but it does mean the fitted values
carry little information and should not be quoted as measured GaN exponents.

### What can be said

Supported: with a matched budget, a matched solver class and honest parameter
counts, **SRH fits this data better than power-law, stretched-exponential or
logarithmic creep at 40 epochs**, and the ordering is consistent across every
prediction horizon.

Not yet supported: that SRH is the converged optimum, or that the fitted
exponents estimate a physical quantity. A 200-epoch run of all four, plus a
second seed, is needed before the claim goes in a paper — roughly 3 hours,
since only SRH is slow.

The invalid RK4 results stay archived under
`results/mechanism_candidates/rk4_invalid/` so the failure remains inspectable.


---

## Full budget: 200 epochs, two seeds

The screen's caveat was that nothing had converged. Re-run at the full Stage-3
budget with a second seed:

| candidate | params | val (s42) | val (s43) | RMSE (s42) | RMSE (s43) | mean RMSE | vs SRH |
|---|---|---|---|---|---|---|---|
| **srh** | **18** | 0.02698 | 0.02751 | 0.2061 | 0.2040 | **0.2050** | — |
| stretched | 21 | 0.04961 | 0.05008 | 0.2837 | 0.2943 | 0.2890 | **+41.0 %** |
| power | 21 | 0.05587 | 0.05647 | 0.2929 | 0.2869 | 0.2899 | **+41.4 %** |
| log | 18 | 0.09615 | 0.09452 | 0.3893 | 0.4200 | 0.4046 | +97.3 % |

SRH wins on **every seed against every alternative**, by 41 % over the closest
competitor and with 3 fewer parameters than power and stretched. Seed-to-seed
spread within a candidate is ~1-3 %, an order of magnitude below the gaps
between candidates, so the ranking is not a seed artefact. The margin is also
15x this dataset's ~2.7 % noise floor.

The SRH run reproduces the independently-trained Stage-3 model exactly
(val 0.02698, RMSE 0.2061 for seed 42), confirming the harness introduces no
drift.

### Convergence

Full budget shifted every number but not the order. SRH went from
val 0.04546 / RMSE 0.2425 at 40 epochs to 0.02698 / 0.2061 at 200 — so the
screen was indeed reading unconverged models, and the ranking would have been
reported at inflated error had it stopped there.

All four are still improving ~17-25 % over their last 100 epochs, so none is
fully converged even now. That is a limit on the absolute numbers, not on the
comparison: the improvement rates are close (SRH 24 %, stretched 21 %,
power 22 %, log 17 %), so more budget would lower all four roughly together
rather than reorder them. SRH improving *fastest* means additional epochs
would, if anything, widen its lead.

### The fitted exponents

| candidate | seed | zG | zB | zF |
|---|---|---|---|---|
| power | 42 | 0.481 | 0.387 | 0.710 |
| power | 43 | 0.475 | 0.384 | 0.710 |
| stretched | 42 | 0.485 | 0.411 | 0.710 |
| stretched | 43 | 0.473 | 0.407 | 0.707 |

Now clearly displaced from their initialisation (0.50 / 0.40 / 0.65) and
**reproducible across seeds to within 0.01**, unlike the screen where they had
barely moved. The fast mode's exponent rises consistently to ~0.71 in both
laws and both seeds.

These remain fitted values of a *losing* model, so they should not be quoted
as measured GaN dispersion exponents; their interest is that the alternatives
were genuinely optimised, which is what makes the comparison fair.

## Conclusion

With a matched budget, matched solver class, honest parameter counts and two
seeds, **the data prefer Shockley-Read-Hall kinetics over power-law,
stretched-exponential and logarithmic-creep rate laws** — by 41 % in test RMSE
over the closest alternative, consistently across seeds.

This upgrades the pipeline's SRH choice from an assumption to a measurement,
and complements the separate finding that the physics prior as a whole beats
free-form dynamics by 13.2 % (4 seeds, CI [-16.3, -10.2]).

Caveat for a write-up: two seeds support a 41 % gap comfortably, but the
absolute RMSEs are not converged values, and the comparison covers the
reversible trap states only — zM, zL and zC keep their saturating-drift form
in every candidate and were never in dispute.
