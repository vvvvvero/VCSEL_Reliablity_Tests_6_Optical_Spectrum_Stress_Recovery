# Candidate rate-law comparison — INCONCLUSIVE, do not cite

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
