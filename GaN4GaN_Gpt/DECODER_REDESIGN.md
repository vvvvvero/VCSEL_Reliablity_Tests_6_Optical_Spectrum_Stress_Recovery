# Decoder redesign: giving each latent its own observable signature

Status: **proposal, not yet implemented.** Nothing in the training pipeline is
changed by this file.

## The problem, restated

`20_latent_degeneracy.py` measured the trained decoder's main-channel block
(`Vth, IDSS, RON, gmmax` x `zG, zB, zM, zC`):

| quantity | value |
|---|---|
| mean pairwise row cosine | 0.959 (min 0.907) |
| sigma1 share of Frobenius energy | 96.99 % |
| effective rank | **1.062 / 4** |
| zB, zM reproduced by other latents | residual 1.7 % |

The four rows of the current mask are *identical*:

```
             zG  zB  zM  zL  zC
Vth         [1,  1,  1,  0,  1]
IDSS        [1,  1,  1,  0,  1]     <- same pattern
RON         [1,  1,  1,  0,  1]     <- same pattern
gmmax       [1,  1,  1,  0,  1]     <- same pattern
IDLeak      [0,  1,  0,  1,  0]
IGLeak      [1,  0,  0,  1,  0]
```

Nothing distinguishes zG from zB from zM on the features that carry the
signal. `zL` is the one latent that stays identifiable (substitutability
residual 56 %) precisely because the mask gives it observables of its own.
**The fix is to do for the other latents what the mask already does for zL.**

## Evidence used to choose the new observables

From `21_iv_curve_features.py`, on all 203 trained devices.

**Independence from the old scalars** (linear R^2, 1775 clean records):

| feature | R^2 | |
|---|---|---|
| SS_lin | 0.074 | independent |
| n_ideality | 0.116 | independent |
| SS_sat | 0.128 | independent |
| SS_ratio | 0.220 | independent |
| V_gmpeak_sat | 0.294 | independent |
| DIBL | 0.478 | partly |
| gm_fwhm_sat | 0.479 | partly |

For comparison the old scalars predict *each other* at r = 0.99
(`gm_peak_sat` vs `Ion_sat`) — that correlation **is** the degeneracy.

**Does it move with stress?** (median relative drift vs the t=0 baseline)

| feature | @10h | @100h | @1000h | @2000h |
|---|---|---|---|---|
| SS_sat | +0.196 | +0.184 | +0.252 | **+0.693** |
| log_IonIoff_sat | -0.209 | -0.200 | -0.155 | -0.269 |
| SS_lin | +0.172 | +0.157 | +0.141 | +0.244 |
| SS_ratio | +0.021 | +0.018 | +0.108 | +0.192 |
| DIBL | +0.074 | +0.070 | +0.106 | +0.172 |
| gm_peak_lin | -0.058 | -0.054 | -0.071 | -0.114 |
| Ron_out | +0.027 | +0.031 | +0.049 | +0.098 |
| gm_fwhm_sat | +0.067 | +0.065 | +0.065 | +0.095 |
| n_ideality | -0.062 | -0.071 | -0.055 | -0.037 |
| Vth_lin | -0.044 | -0.014 | +0.033 | +0.043 |
| V_knee | 0.000 | 0.000 | -0.019 | +0.026 |
| **gm_fwhm_lin** | 0.000 | 0.000 | 0.000 | **0.000** |

**Does it separate devices at fixed (T, t)?** A feature that only tracks
temperature and time cannot give a latent a per-device identity — this is the
property `alpha` needs and never had.

| feature | within-(T,t) CV |
|---|---|
| SS_sat | 1.613 |
| SS_lin | 1.357 |
| Ron_out | 0.837 |
| gm_peak_sat | 0.834 |
| Ion_sat | 0.677 |
| DIBL | 0.455 |
| SS_ratio | 0.449 |
| Vth_lin | 0.351 |
| gm_fwhm_sat | 0.136 |
| V_knee | 0.111 |
| **n_ideality** | **0.044** |

## Two features rejected on the evidence

* **`gm_fwhm_lin`** — drift is *exactly* 0.000 at every horizon. The linear-region
  gm curve is only ~0.6 V wide and the VG step is 0.1 V, so the FWHM is
  quantised to a handful of grid points and cannot resolve degradation. Excluded.
* **`n_ideality`** — physically the most appealing gate-degradation probe, but it
  drifts only 3.7-7 % and separates devices at CV = 0.044, an order of magnitude
  below the SS features. Kept in the extracted set for interpretation, but **not
  given a latent to drive**; on this evidence it would be a near-constant row.

Leakage note: `IGS_leak`/`IDS_leak` are *signed* currents (99.7 % negative,
crossing zero), so the relative drift of -15 in the trend table is an artefact
of dividing by a near-zero signed baseline, not a real 1500 % change. If used
they must enter as `log10|I|`, which is what the existing pipeline already does
for IDLeak/IGLeak.

## Proposed observable set (10 rows)

Keeps the four existing main-channel features so results stay comparable to
every previous run, and adds six curve-shape rows.

| # | feature | new? | mechanism it probes |
|---|---|---|---|
| 0 | Vth | | threshold shift (any charged trap) |
| 1 | IDSS | | on-state current |
| 2 | RON | | total on-resistance |
| 3 | gmmax | | peak transconductance |
| 4 | IDLeak | | drain leakage path |
| 5 | IGLeak | | gate leakage path |
| 6 | **SS_lin** | new | interface-state density D_it |
| 7 | **SS_sat** | new | D_it under drain bias |
| 8 | **gm_fwhm_sat** | new | mobility loss vs rigid Vth shift |
| 9 | **DIBL** | new | buffer / channel depletion control |
| 10 | **V_knee** | new | access + contact resistance |
| 11 | **V_gmpeak_sat** | new | rigid shift of the gm curve |

## Proposed sparsity mask

```
                  zG  zB  zM  zL  zC
Vth              [ 1,  1,  1,  0,  1]   unchanged
IDSS             [ 1,  1,  1,  0,  1]   unchanged
RON              [ 1,  1,  1,  0,  1]   unchanged
gmmax            [ 1,  1,  1,  0,  1]   unchanged
IDLeak           [ 0,  1,  0,  1,  0]   unchanged
IGLeak           [ 1,  0,  0,  1,  0]   unchanged
SS_lin           [ 1,  0,  0,  0,  1]   <- zG ONLY (+ damage)
SS_sat           [ 1,  1,  0,  0,  1]   <- zG, zB
gm_fwhm_sat      [ 0,  0,  1,  0,  1]   <- zM ONLY (+ damage)
DIBL             [ 0,  1,  0,  0,  1]   <- zB ONLY (+ damage)
V_knee           [ 0,  1,  1,  0,  1]   <- zB, zM
V_gmpeak_sat     [ 1,  0,  1,  0,  1]   <- zG, zM
```

A first draft gave `V_gmpeak_sat` the pattern `[1,1,0,0,1]`, which duplicated
`SS_sat` exactly — reintroducing on the new rows the very same-pattern problem
being fixed. Checking the mask rather than trusting the table caught it.
`zM` is the physically correct third connection anyway: a rigid translation of
the gm curve responds to charged traps (zG) and to transport (zM), whereas
buffer depletion shows up as DIBL and knee movement, not as a rigid shift.

Rationale, row by row:

* **SS_lin ← zG.** Subthreshold swing is set by interface-state density.
  A trap that merely *fills* shifts Vth and leaves SS alone; a trap that is
  *created* degrades SS. This is the single cleanest handle for separating
  zG from everything else, and it is the strongest signal available
  (drift +0.244, within-(T,t) CV 1.357).
* **SS_sat ← zG, zB.** Same probe under drain bias, which adds sensitivity to
  the buffer depletion region, hence zB as well.
* **gm_fwhm_sat ← zM.** Mobility degradation lowers *and broadens* the gm
  curve; a pure threshold shift translates it without changing its width.
  Width therefore isolates transport (zM) from threshold effects.
* **DIBL ← zB.** Drain-induced barrier lowering is governed by how well the
  buffer confines the channel — a direct buffer-trap probe.
* **V_knee ← zB, zM.** The knee moves out when access resistance grows
  (zM) or when the buffer traps up (zB).
* **V_gmpeak_sat ← zG, zB.** The *position* of the gm peak is the rigid-shift
  counterpart to gm_fwhm: it responds to charged traps, not to mobility.

`zC` (cumulative damage) is allowed into every new row: it is the monotone
irreversible term and must be able to appear anywhere. `zL` stays confined to
the leakage rows, which is what has kept it identifiable.

Verified properties of the proposed mask (checked programmatically, not by eye):

| property | current | proposed |
|---|---|---|
| mask rank | 3 / 5 | **5 / 5** |
| distinct row patterns | 3 / 6 | 9 / 12 |
| distinct patterns among the *new* rows | — | **6 / 6** |
| rows isolating zG | IGLeak | IGLeak, **SS_lin** |
| rows isolating zB | IDLeak | IDLeak, **DIBL** |
| rows isolating zM | **none** | **gm_fwhm_sat** |
| zG-zM column overlap (Jaccard) | 0.80 | 0.50 |
| zB-zM column overlap | 0.80 | 0.50 |

The decisive change is that **zM is isolated for the first time**: in the
current mask no row exists where zM is the only trap latent, so nothing ever
forced it to mean anything specific.

The four original rows deliberately keep their shared pattern, so results stay
comparable with earlier runs. Their degeneracy is broken *indirectly*: the new
rows pin zG/zB/zM to distinct signatures, so the optimiser can no longer
permute them freely without paying a cost on those rows.

## Sign constraints

Free-sign rows (degradation may go either way):

* `Vth`, `IDLeak`, `IGLeak` — unchanged from the current config.
* `V_gmpeak_sat` — new. It tracks Vth, whose direction depends on trap
  polarity, so it must be free-signed for the same reason Vth is.
  Measured drift is negative early (-0.042 at 10 h) and positive later
  (+0.042 at 1000 h), i.e. it genuinely changes direction.

Non-negative (degradation is one-directional):

* `SS_lin`, `SS_sat` — interface states only ever *worsen* the swing.
* `gm_fwhm_sat`, `DIBL`, `V_knee` — all observed to increase with stress
  (+0.095, +0.172, +0.026 at 2000 h).

## Expected outcome and how it will be judged

The mask alone does not guarantee identifiability; it removes the structural
obstacle. Success criteria, measured with the tools already written:

1. `20_latent_degeneracy.py` effective rank **> 2.5** (from 1.06).
2. Substitutability residual **> 20 %** for zG, zB and zM individually
   (currently 1.7-3.8 %).
3. `alpha` standard deviation across devices materially above 3.3e-05.
4. Only then is the A/B/C ablation (`16_`) worth re-running — and only then
   can its result be interpreted as evidence about physics conditioning.

If (1) and (2) improve but the A/B/C ablation stays null, that is a genuine
and reportable finding about the physics prior. Until they improve, the
ablation cannot distinguish "the prior does not help" from "the prior channel
is empty".

## Work required to implement

1. Extend `01_data_preprocessing.py` to merge the six new features into the
   observation tensor — writing to a **new** processed file, never
   overwriting `processed_data.pkl`.
2. Update `config.py`: `FEATURES`, `FEATURE_DIM` (6 -> 12),
   `DECODER_SPARSITY`, `FREE_SIGN_ROWS`, `ENCODER_INPUT_DIM` (15 -> 27).
3. Normalisation statistics for the new features, and their own degradation
   transform: SS and V_knee are positive quantities (log-ratio like RON),
   while DIBL and V_gmpeak_sat are signed voltages (difference, not ratio).
4. Retrain Stage 1-3 (~11 h), then re-run `20_`, then the ablation.

Step 3 is the one with real design content — the existing `-log(P/P_0)`
transform assumes a positive quantity and is wrong for the signed features.
