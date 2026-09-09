# Stage 5 on the repaired backbone

## Why this was re-run

The standing verdict — "adversarial fine-tuning gives no significant benefit" —
came from a 15-fold cross-validation dated **2026-08-24**, i.e. before every
fix in this work. That model had 6 features, 5 latents, a decoder of effective
rank 1.06, and an `alpha` pinned at its lower bound with std 3.3e-05. Its
`z_phys` was a deterministic function of (T, t) with per-device spread ~1e-8.

A discriminator conditioned on a channel carrying no per-device information
cannot learn anything device-specific. So that null result was evidence about
the *channel*, not about adversarial training — the same root cause as the
A/B/C ablation tie. Repeating the claim without re-testing would have been
citing a result whose premise no longer holds.

On the repaired backbone (11 features, 6 latents, effective rank 2.33, alpha
std 1.07e-01) the discriminator has something to discriminate for the first
time.

## What happened

Training ran clean, all collapse guards passing until a single failure at
epoch 9. Best epoch 3.

| | Stage 4C | Stage 5 | change |
|---|---|---|---|
| **validation** CRPS | 0.05366 | 0.05321 | **−0.8 %** |

That looks like a win. It is not one, and the test split shows why.

## Test-split result (100 samples, identical seed)

| metric | Stage 4C | Stage 5 | change |
|---|---|---|---|
| CRPS (generated features) | 0.0784 | 0.0784 | **+0.03 %** |
| CRPS (all) | 0.1261 | 0.1261 | +0.02 % |
| CRPSS | 0.2041 | 0.2053 | +0.62 % |
| **Cov50** | 0.5430 | 0.5120 | **−5.70 %** |
| **Cov80** | 0.7847 | 0.7487 | **−4.59 %** |
| **Cov90** | 0.8540 | 0.8272 | **−3.14 %** |
| **MACE** | 0.0466 | 0.0522 | **+12.05 %** |
| W1 (increments) | 0.0726 | 0.0671 | −7.54 % |
| interval width @90 | 0.2802 | 0.2520 | −10.05 % |

**CRPS is unchanged to three decimal places while calibration gets worse.**
The mechanism is visible in the last two rows: Stage 5 narrows the predictive
intervals by 10 %, which improves the distributional distance W1 but pushes
coverage further below nominal and raises calibration error by 12 %. For a
reliability model, intervals that are too narrow are the dangerous failure —
they understate risk.

Per-feature CRPS shows the same story: the four original features get slightly
worse (+0.1 to +1.1 %), the curve-shape features slightly better (−0.5 to
−1.4 %), and the leakage pair is untouched because it is not generated. Nothing
moves more than ~1 %.

## The validation gain was selection noise

`val_CRPS` oscillates between 0.053 and 0.065 across the 9 epochs, and Stage 5
selects its best epoch *on that metric*. Picking the minimum of a noisy
sequence produces an apparent improvement whether or not one exists — which is
exactly what the −0.8 % validation gain turns out to be, since it does not
survive to test.

## Conclusion

**The original verdict stands, but for a better-supported reason.** Stage 5
gives no CRPS benefit on the repaired backbone, and actively degrades
calibration. The 2026-08-24 result could not distinguish "adversarial training
does not help" from "the conditioning channel is empty"; this one can, because
the channel now demonstrably carries per-device information.

That is a stronger negative result than the one it replaces, and it is worth
stating positively in a write-up: with a physics prior supplying the inductive
bias and CRPS supplying a proper scoring rule, **the adversarial term has no
remaining work to do**. The discriminator is not being starved of information —
it has information and still adds nothing.

## Multi-seed confirmation (5 seeds x 5 folds, 25 paired runs)

Each fold retrains both stages, so the spread reflects real training variation
rather than evaluation noise.

| | Stage 4C | Stage 5 |
|---|---|---|
| CRPS | 0.11365 | 0.11283 |
| Cov90 | 0.8656 ± 0.0309 | 0.8549 ± 0.0335 |

Paired CRPS difference over 1015 points: mean 0.00077,
95 % CI [0.00040, 0.00116], **Wilcoxon p = 0.0027**.

**Stage 5's CRPS advantage is statistically significant — and not worth
having.** Both statements are true, and the gap between them is the point:

* The effect is **0.68 %** (CI 0.35–1.02 %), with **Cohen's d = 0.12**, below
  the conventional 0.2 threshold for "small".
* It is detectable only because n = 1015. At the 25-run level it would not be.
* **47.9 % of individual points get worse** — barely better than a coin flip.
* Meanwhile **Cov90 falls from 0.8656 to 0.8549**, moving 1.07 points further
  from its 0.90 target, exactly the interval-narrowing seen in the single run.

So adversarial fine-tuning buys a sub-1 % improvement in a distributional
score by making the predictive intervals less trustworthy. For a reliability
model whose output is a risk interval, that is a bad trade regardless of the
p-value.

This also revises the single-run reading above: CRPS is not *unchanged*, it
improves very slightly. The conclusion is unaffected, and the reason is
sharper — the issue is not that Stage 5 does nothing, but that what it does is
not what a reliability model needs.

## Caveats

The CV uses its own fold splits rather than the fixed train/val/test split, so
its absolute CRPS (0.1136) is not comparable with the single-run test figure
(0.0784); only the paired within-fold comparison is meaningful. Both analyses
agree on direction, which is what matters.
