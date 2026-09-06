# Physics-free baseline comparison

Answers the question the A/B/C ablation structurally cannot: **is the physics
prior itself worth anything?** In A/B/C all three conditions share one trained
physics backbone, so even "no physics" gets its predictions from the
SRH/Arrhenius ODE. Here the ODE is replaced outright.

Only the latent dynamics change. Encoder, decoder, alpha net, data, splits,
losses, optimiser, integrator and epoch budget are identical.

## Main comparison (all temperatures, full data)

| model | dynamics params | test RMSE | t=500h | t=1000h | t=2000h |
|---|---|---|---|---|---|
| **physics (PI-TimeGAN)** | **18** | **0.2061** | 0.1569 | 0.1871 | 0.2877 |
| Neural ODE (free field) | 5126 | 0.2422 | **0.1443** | **0.1733** | **0.2812** |
| GRU | 198 | 0.3257 | 0.1980 | 0.2515 | 0.3822 |

Physics wins overall by **14.9 %** while using **285x fewer** dynamics
parameters, and beats the structure-free GRU by 36.7 %.

**But the long-horizon result is the opposite of what was predicted.** The
Neural ODE is slightly BETTER at 500/1000/2000 h (0.1443 vs 0.1569 at 500 h).
The physics advantage comes from the short and middle horizons, not from
extrapolation. The prediction that saturation and Arrhenius scaling would
dominate at long horizons is not supported.

## OOD temperature (train 275+300 C, test on held-out 325 C)

| model | test RMSE | t=500h | t=1000h | t=2000h |
|---|---|---|---|---|
| physics | **0.6310** | 0.2479 | **1.2792** | **0.3473** |
| Neural ODE | 0.6447 | **0.2341** | 1.2833 | 0.3601 |
| GRU | 0.7074 | 0.3309 | 1.3253 | 0.4859 |

Physics leads by only **2.1 %** — far less than the "physics extrapolates in
temperature, a free field cannot" argument would predict. Every model degrades
about 3x versus the in-distribution case (0.21 -> 0.63), so held-out
temperature is hard for all of them and Arrhenius structure is not rescuing it.
The t=1000 h column is anomalous for all three models (~1.28) and is worth a
separate look.

## Learning curve (test RMSE)

| model | 25 % | 50 % | 100 % |
|---|---|---|---|
| physics | **0.2383** | **0.2220** | **0.2061** |
| Neural ODE | 0.2554 | 0.2570 | 0.2422 |
| physics advantage | -6.7 % | -13.6 % | -14.9 % |

Physics wins at every training-set size. The advantage does NOT grow as data
shrinks, which is the other half of the standard inductive-bias argument that
this data does not support. Note the physics model at 25 % of the data
(0.2383) already beats the Neural ODE with 100 % (0.2422) — 36 devices against
143.

## What can honestly be claimed

Supported:
* Physics beats both baselines on overall test RMSE at every data fraction and
  in the OOD setting — 7 comparisons, no losses.
* It does so with 18 dynamics parameters against 5126, i.e. the advantage is
  not bought with capacity.
* Physics trained on a quarter of the devices outperforms the free field
  trained on all of them.
* The parameters are interpretable (Ea_rev 0.348 eV, Ea_irrev 0.651 eV,
  tau_F 0.57 h) where the baseline's 5126 weights are not.

NOT supported, despite being the intuitive story:
* That the advantage is largest at long horizons — the Neural ODE is
  marginally better there.
* That the advantage grows as data becomes scarce — it shrinks.
* That Arrhenius structure gives a large OOD-temperature benefit — it gives
  2.1 %.

## Caveats

Single seed per configuration. The 14.9 % main-comparison gap is comfortably
larger than the ~2.7 % that the prefix-length sweep established as noise on
this dataset, but the 2.1 % OOD gap is NOT, and should not be reported as a
physics win without multi-seed repetition. The 6.7-14.9 % learning-curve gaps
are non-monotone in data fraction, which is itself a sign that single-seed
noise is present at that scale.

Training time differs by ~30x (80 min vs 2.6 min) because the physics ODE uses
the IMEX/RK4 driver with substeps. That is a cost of the method, not a
confound for accuracy.
