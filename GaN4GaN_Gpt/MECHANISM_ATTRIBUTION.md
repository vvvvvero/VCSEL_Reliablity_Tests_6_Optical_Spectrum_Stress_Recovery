# Mechanism attribution, and how far it can be trusted

How much of each observed degradation does each physical mechanism account
for — and is that decomposition a property of the data, of one optimisation
run, or of the prior we imposed?

Three questions, answered in order, because each only makes sense once the
previous one has an answer:

1. **Is the split reproducible?** Yes — four independent fits agree on the
   dominant mechanism for 10/10 free rows.
2. **Is it therefore correct?** Not established by (1). Four fits of the same
   model under the same mask can agree and be wrong together.
3. **Does it agree with physics measured independently?** Partly. One of the
   two activation energies is constrained by the data and lands inside the
   published band; the other is not identified at all.

Scripts: `29_mechanism_attribution.py`, `36_attribution_summary.py`,
`39_activation_energy_validation.py`.

---

## 1. The decomposition

Each observable's drift at t = 2000 h is decomposed over the six latents.
Because the decoder is a **sparse constrained linear map** rather than a free
MLP, a share is a real coefficient ratio and not an attribution heuristic —
this is what the decoder redesign (see `DECODER_REDESIGN.md`) bought.

Mean ± sd over four independent backbones (`ext11_filtered` unseeded, plus
seeds 101/202/303), all on the filtered 200-device set:

| observable | zG gate | zB buffer | zF fast rev. | zM transport | zL leakage | zC cumulative |
|---|---|---|---|---|---|---|
| Vth | 0.123±.016 | 0.345±.018 | — | **0.396±.019** | — | 0.137±.007 |
| IDSS | 0.101±.007 | 0.288±.006 | — | **0.351±.008** | — | 0.260±.005 |
| RON | 0.121±.013 | 0.350±.009 | — | **0.378±.011** | — | 0.151±.005 |
| gmmax | 0.124±.018 | 0.339±.020 | — | **0.412±.029** | — | 0.125±.005 |
| IDLeak | — | 0.213±.015 | — | — | **0.787±.015** | — |
| IGLeak | 0.060±.015 | — | — | — | **0.940±.015** | — |
| SS_lin | 0.293±.044 | — | **0.707±.044** | — | — | — |
| SS_sat | 0.285±.032 | **0.715±.032** | — | — | — | — |
| gm_fwhm_sat | — | — | 0.425±.035 | **0.575±.035** | — | — |
| V_gmpeak_sat | 0.235±.034 | — | — | **0.765±.034** | — | — |

DIBL is omitted: its sparsity mask admits a single latent, so its share
restates the prior rather than measuring anything. Including it would inflate
the agreement statistic, so `36_` labels such rows `forced` and excludes them.

**Reproducibility.** The dominant latent is identical on **10/10 free rows**
across all four fits. Share sd across backbones: mean 0.0194, median 0.0159,
max 0.0436 over 28 non-zero terms. The backbones themselves agree too — Stage
1 val loss 0.0725–0.0729, Stage 2 0.1056–0.1067, rollout MSE 0.02587 /
0.02661 / 0.02609 / 0.02785.

Shares are therefore quotable at about **±0.02** (1 sd).

---

## 2. Why that is not yet identification

Reproducibility rules out **run-to-run optimisation noise**, and nothing more.
That was the standing objection to quoting these numbers, and it is now
settled. But four fits of the *same* model, under the *same* sparsity mask and
sign constraints, can agree with each other and be wrong together. Agreement
among them is not evidence about the physics; it is evidence about the
optimiser.

The only way across that gap is agreement with something the model never saw.

---

## 3. External validation: activation energy

The ODE carries two shared Arrhenius energies — `Ea_rev` for the reversible
trap channels, `Ea_irrev` for the irreversible ones. Both are dimensional
physical quantities with decades of independent measurement behind them, which
makes them the one testable point of contact in the whole model.

Method (`39_`): freeze one energy at a grid of values, leave everything else at
its fitted solution, and profile the **Stage 3 rollout loss**. A parameter the
data constrains shows a minimum; one it cannot see shows a flat profile.

### Ea_irrev — constrained, and consistent with the literature

| Ea [eV] | 0.05 | 0.20 | 0.40 | **0.60** | 0.80 | 1.30 | 2.20 |
|---|---|---|---|---|---|---|---|
| val MSE | 0.026147 | 0.026014 | 0.025900 | **0.025862** | 0.025903 | 0.026377 | 0.028454 |

Clear interior minimum, parabolic vertex at **0.583 eV**, rising on both sides,
10.0 % worse at the grid ends. The four backbones independently fit
**0.6444–0.6987 eV** (mean 0.667, CV 3.4 %), sitting at that minimum.

Published irreversible-degradation energies for GaN HEMTs span roughly
**0.5–1.3 eV**, with 0.6–0.7 eV commonly attributed to buffer and interface
trap processes under thermal storage. The fit falls inside that band.

### Ea_rev — not identified, and must not be quoted

The profile decreases **monotonically** across 0.05–2.20 eV with no minimum,
and the entire spread is **0.83 %** of the MSE. Correspondingly the four
backbones scatter **0.2778–0.6587 eV**, a coefficient of variation of **38 %**
against 3.4 % for the irreversible energy.

The fitted `Ea_rev` is a product of initialisation and the optimiser's path.
It is not a measurement and does not belong in a results table.

### How strong is the Ea_irrev agreement

**Consistency, not confirmation.** A broad literature band overlapping a
loosely constrained fit is the weakest useful form of external validation. It
would have been informative had the fit landed at 0.1 or 2 eV, and it did not.
What it rules out is that the irreversible channel is fitting something with no
thermal-activation character at all. Report it at that strength.

---

## A probe that would have given a confident wrong answer

The first version of the profile returned a **perfectly flat** loss — identical
to six decimal places from 0.05 to 2.20 eV, for *both* energies.

It used `08_training._forward`, which is the encoder → decoder reconstruction
path and never invokes the ODE, so no Arrhenius parameter could reach it. Read
at face value the output says "neither energy is identified", which is half
wrong and would have retired a valid result.

A flat profile is exactly what a broken probe looks like. The profile must run
through `validation_prefix_rollout_mse`, the Stage 3 selection metric, which
integrates the ODE forward from the prefix.

---

## What would actually identify the mechanisms

None of these is available in the present dataset.

**More stress temperatures.** Three points give two independent ratios, while
the model carries two activation energies plus per-mechanism rate constants —
so Ea and k are near-collinear. Five or six temperatures would separate them,
and would likely be enough to identify `Ea_rev`.

**Recovery experiments.** The reversible channels predict relaxation once
stress is removed. Measuring it tests the reversible/irreversible split
directly, which is precisely what `Ea_rev`'s unidentifiability currently
blocks. A stress–measure–recover cycle is the single most informative addition.

**Bias-dependent stress.** Gate versus drain stress should shift attribution
between zG and zM in a direction the physics predicts. Agreement there would
test the *assignment* of latents to mechanisms, not just their energies.

---

## What to claim in a write-up

Supported:

* the decomposition is reproducible across independent fits, ±0.02 on shares,
  with the dominant mechanism stable on 10/10 free rows;
* `Ea_irrev` is constrained by the data at 0.58–0.67 eV and is consistent with
  published ranges for irreversible GaN HEMT degradation.

Not supported:

* that the shares are physically correct — reproducibility does not establish
  this, and only one of two energies has any external corroboration;
* any value for `Ea_rev`;
* the `forced` rows (DIBL), whose shares restate the sparsity mask.
