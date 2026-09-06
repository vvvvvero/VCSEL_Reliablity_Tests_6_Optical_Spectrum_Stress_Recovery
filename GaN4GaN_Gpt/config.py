"""
config.py
=========
Central configuration for the PI-TimeGAN reliability prediction pipeline.
All paths, hyperparameters, and architecture settings are defined here.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = r"D:\2026\article\GaN4GaN\data\thermalstorage_Data"
OUTPUT_PATH = r"D:\2026\article\GaN4GaN\output\pi_timegan"
CHECKPOINT_DIR = os.path.join(OUTPUT_PATH, "checkpoints")
RESULTS_DIR = os.path.join(OUTPUT_PATH, "results")
FIGURES_DIR = os.path.join(OUTPUT_PATH, "figures")
PROCESSED_DATA_PATH = os.path.join(OUTPUT_PATH, "processed_data.pkl")

# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------
# Device type identifiers as they appear in filenames
DEVICE_TYPES = ["A2ACH4FP", "A2ACH4", "A8A", "P10C"]

# Storage temperatures in degrees Celsius (appear in filenames)
TEMPERATURES_C = [275, 300, 325]

# Canonical time points in hours (all files share this grid)
TIME_POINTS_H = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]

# Map of logical parameter names to filename substring patterns
# For each device_type + temperature, the loader searches for files matching
# the pattern:  {device_type}_{temp}_{param_pattern}*.xlsx
# Files containing '%' in the name are always skipped (percentage files).
PARAM_FILE_PATTERNS = {
    "IDSS":   "IDSS",
    "Vth":    "Vth",
    "IDLeak": "IDLeak",
    "IGLeak": "IGLeak",
    "RON":    "RON",
    "gmmax":  "gmmax",
}

# ---------------------------------------------------------------------------
# Observation set
# ---------------------------------------------------------------------------
# EXTENDED_FEATURES switches between the original six extracted scalars and
# the 12-feature set that adds curve-shape observables from the raw IDVG/IDVD
# sweeps (built by 22_build_extended_dataset.py into processed_data_ext.pkl).
#
# Why the extra six exist: with only the original scalars the decoder's
# main-channel rows (Vth, IDSS, RON, gmmax) share an identical sparsity
# pattern, so zG/zB/zM are interchangeable -- measured effective rank 1.06 out
# of 4, with zB and zM reproducible from the other latents to within 1.7 %.
# That is why alpha never identifies and why the A/B/C physics-conditioning
# ablation compares three informationally identical models. See
# DECODER_REDESIGN.md and 20_latent_degeneracy.py.
#
# Set to False to reproduce every result obtained before the extension.
EXTENDED_FEATURES = True

BASE_FEATURES = ["Vth", "IDSS", "RON", "gmmax", "IDLeak", "IGLeak"]
# Curve-shape features, in the order 22_build_extended_dataset.py appends them.
CURVE_FEATURES = ["SS_lin", "SS_sat", "gm_fwhm_sat", "DIBL", "V_knee",
                  "V_gmpeak_sat"]

FEATURES = BASE_FEATURES + CURVE_FEATURES if EXTENDED_FEATURES else list(BASE_FEATURES)
FEATURE_DIM = len(FEATURES)   # 6 or 12

# Width of the per-device static vector x0 (initial absolute parameter values).
# This is NOT FEATURE_DIM: x0 comes from the six measured scalars and does not
# grow when curve-shape observables are added, because those are defined
# relative to the t=0 sweep and so carry no independent initial value.
# DeviceAlphaNet consumes [x0_static, T_norm], so it must size itself from
# this, not from FEATURE_DIM -- otherwise the extended run builds a 13-wide
# input layer and is handed 7.
X0_STATIC_DIM = len(BASE_FEATURES)   # 6

# Degradation transformation epsilon to avoid log(0)
EPSILON = 1e-9

# Leakage preprocessing (Debug6)
# Estimated on training split only; used as absolute floors in log-ratio transform.
LEAKAGE_FLOOR_PERCENTILE = 1.0
LEAKAGE_FLOOR_MIN = 1e-12
LEAKAGE_FLOOR_DEFAULT = 1e-9
# Clip transformed leakage log-ratios to reduce detector-floor dominated tails.
LEAKAGE_LOG_CLIP = 6.0

# ---------------------------------------------------------------------------
# Physics / latent space
# ---------------------------------------------------------------------------
# Latent state names and ordering  [zG, zB, zM, zL, zC]
LATENT_NAMES = ["zG", "zB", "zM", "zL", "zC"]
LATENT_DIM = 5

# Physical constants
KB_EV = 8.617333e-5     # Boltzmann constant [eV/K]
T_REF_K = 573.15        # Reference temperature = 300 °C [K]

# Temperature offset (Celsius → Kelvin)
CELSIUS_TO_KELVIN = 273.15

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
# Encoder (GRU)
# Input = [x(F), feature_mask(F), T_norm, log_t, delta_log_t]
# -> 15 with the base 6 features, 27 with the extended 12.
ENCODER_INPUT_DIM = FEATURE_DIM * 2 + 3   # 15 (base) / 27 (extended)
GRU_HIDDEN_DIM = 16
GRU_NUM_LAYERS = 2

# Decoder (sparse linear physics decoder)
# Sparsity mask — rows = features (x1..x6), columns = latent (zG,zB,zM,zL,zC)
# 1 = connection allowed, 0 = forced zero
#                     zG  zB  zM  zL  zC
_BASE_SPARSITY = [
    [1,  1,  1,  0,  1],   # x1: Vth      <- zG, zB, zM, zC
    [1,  1,  1,  0,  1],   # x2: IDSS     <- zG, zB, zM, zC
    [1,  1,  1,  0,  1],   # x3: RON      <- zG, zB, zM, zC
    [1,  1,  1,  0,  1],   # x4: gmmax    <- zG, zB, zM, zC
    [0,  1,  0,  1,  0],   # x5: IDLeak   <- zB, zL
    [1,  0,  0,  1,  0],   # x6: IGLeak   <- zG, zL
]

# Curve-shape rows. These exist to give each latent an observable of its own,
# the way the leakage rows already do for zL -- which is why zL is the only
# latent that stays identifiable in the 6-feature model (substitutability
# residual 56 %, versus 1.7 % for zB and zM).
#
# Note the four base rows above are deliberately left identical to each other,
# so results stay comparable with earlier runs. Their degeneracy is broken
# INDIRECTLY: the rows below pin zG/zB/zM to distinct signatures, so the
# optimiser can no longer permute the latents without paying a cost here.
#
# zL stays confined to the leakage rows, which is what has kept it the one
# identifiable latent in the 6-feature model.
#                     zG  zB  zM  zL  zC
# zC is deliberately EXCLUDED from every curve row (its column is 0 below).
# It is admitted into the four base rows only, where cumulative damage is
# genuinely one of several contributors.
#
# Why: zC entering nearly every row gives the optimiser a near-null direction
# it can inflate without bound. An unregularised least-squares fit of this
# decoder to the data ran away to weights of +-5e5 along zG = -zC, and with
# ridge damping zC still shared its signal with zG (substitutability 30 % and
# 25 %). Tested five zC connection patterns by direct fit; removing zC from
# all six curve rows was the clear winner:
#
#   variant                      resid   eff.rank    zG     zB     zM
#   zC everywhere (first draft)  0.199      1.337   25.2   59.3   92.2
#   zC on damage rows only       0.248      1.094    7.6   38.3    7.6
#   zC off all curve rows  <--   0.276      1.682   98.4   33.5   88.8
#   zC minimal                   0.242      1.072   13.2   48.7   11.8
#
# The residual is slightly worse (0.276 vs 0.199) because zC was absorbing
# variance it had no business explaining; the identifiability gain is the
# point. zG goes from 25 % to 98 % non-substitutable.
#                     zG  zB  zM  zL  zC
_CURVE_SPARSITY = [
    [1,  0,  0,  0,  0],   # SS_lin       <- zG only: subthreshold swing is set
                           #    by interface-state density. A trap that merely
                           #    FILLS shifts Vth and leaves SS alone; a trap
                           #    that is CREATED degrades SS. Cleanest separator
                           #    available (drift +0.244, within-(T,t) CV 1.36).
    [1,  1,  0,  0,  0],   # SS_sat       <- zG, zB: same probe under drain
                           #    bias, which adds buffer-depletion sensitivity.
    [0,  0,  1,  0,  0],   # gm_fwhm_sat  <- zM only: mobility loss lowers AND
                           #    broadens the gm curve, while a pure threshold
                           #    shift translates it without changing its width.
    [0,  1,  0,  0,  0],   # DIBL         <- zB only: drain-induced barrier
                           #    lowering is governed by buffer confinement.
    [0,  1,  1,  0,  0],   # V_knee       <- zB, zM: the knee moves out when
                           #    access resistance grows or the buffer traps up.
    [1,  0,  1,  0,  0],   # V_gmpeak_sat <- zG, zM: peak POSITION is the
                           #    rigid-shift counterpart to the width above.
]

DECODER_SPARSITY = (_BASE_SPARSITY + _CURVE_SPARSITY if EXTENDED_FEATURES
                    else _BASE_SPARSITY)

# Rows that allow negative weights in the decoder
# x1 (Vth): sign can be either direction depending on trap polarity
# x5 (IDLeak) and x6 (IGLeak): leakage can increase or decrease depending on
# reversible/irreversible dynamics and trap occupancy effects.
VTH_DECODER_ROW = 0   # index of Vth feature in FEATURES
IDLEAK_DECODER_ROW = 4   # index of IDLeak feature in FEATURES
IGLEAK_DECODER_ROW = 5   # index of IGLeak feature in FEATURES
FREE_SIGN_ROWS = [
    VTH_DECODER_ROW,
    IDLEAK_DECODER_ROW,
    IGLEAK_DECODER_ROW,
]
if EXTENDED_FEATURES:
    # V_gmpeak_sat tracks the threshold, whose direction depends on trap
    # polarity, so it needs a free sign for the same reason Vth does -- and it
    # is observed to reverse: -0.042 relative at 10 h, +0.042 at 1000 h.
    # SS_lin/SS_sat (interface states only worsen the swing), gm_fwhm_sat,
    # DIBL and V_knee all move one way with stress and stay non-negative.
    FREE_SIGN_ROWS = FREE_SIGN_ROWS + [FEATURES.index("V_gmpeak_sat")]
# Allow signed weights only for the reversible branch columns of leakage rows
# while keeping zL weights non-negative in the monotone leakage channel.
FREE_SIGN_DECODER_COLUMNS = {
    IDLEAK_DECODER_ROW: {1},  # zB (reversible branch)
    IGLEAK_DECODER_ROW: {0},  # zG (reversible branch)
}

# Generator
GENERATOR_NOISE_DIM = 8      # Gaussian noise dimension for generator
GENERATOR_HIDDEN_DIM = 32    # MLP hidden dimension

# Discriminator
DISC_HIDDEN_DIM = 16
DISC_GRU_LAYERS = 1

# ---------------------------------------------------------------------------
# ODE integration
# ---------------------------------------------------------------------------
# Fixed-step RK4 substeps between consecutive log-time observations
# Each log-decade is divided into ODE_SUBSTEPS steps
ODE_SUBSTEPS_PER_LOG_DECADE = 20
# Use the IMEX / exponential integrator (exact closed-form update for the
# fast linear trap modes zG/zB, RK4 for the slow nonlinear zM/zL/zC).
# The fast modes relax in ~5-8 h while the observation grid steps out to
# dt = 1000 h, so plain explicit RK4 is outside its stability region on the
# long steps. Set False to restore the old pure-RK4 path for comparison.
ODE_USE_IMEX = True
# Minimum physical dt [hours] for ODE substeps
ODE_MIN_DT_H = 0.05

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
RANDOM_SEED = 42

# Per-stage learning rates
LR_STAGE1 = 1e-3    # autoencoder
LR_STAGE2 = 5e-4    # ODE consistency
LR_STAGE3 = 3e-4    # multi-step prediction
LR_STAGE4 = 5e-4    # generator distribution matching
LR_STAGE5 = 1e-4    # adversarial fine-tuning

# Per-stage epochs
EPOCHS_STAGE1 = 300
EPOCHS_STAGE2 = 300
EPOCHS_STAGE3 = 200
EPOCHS_STAGE4 = 150
EPOCHS_STAGE5 = 150

# Early stopping patience (in epochs without improvement on validation loss)
EARLY_STOPPING_PATIENCE = 40

# Loss weights
LAMBDA_RECON = 1.0        # Reconstruction loss
LAMBDA_ODE = 0.5          # ODE residual loss
LAMBDA_BOUNDS = 0.2       # Bounds penalty
LAMBDA_MONOTONE = 0.3     # Monotonicity penalty (zM, zL, zC only)
LAMBDA_TEMP_ORDER = 0.15  # Temperature ordering penalty (raised from 0.1 to better separate 325C)
LAMBDA_ADV_G = 0.05       # Generator adversarial loss weight
LAMBDA_ADV_D = 1.0        # Discriminator loss weight
# ODE V2 (SRH-form trap kinetics) boundary-continuity condition #2: penalise
# discontinuous jumps in the trap-to-damage handoff driving force between
# consecutive observed time steps. 0 = off (default); set > 0 to activate
# once validated against a Stage 1-3 retrain with the new ODE.
LAMBDA_HANDOFF = 0.0
# Stage 3 z_phys residual-rank loss (07_losses.py::z_phys_rank_loss): shapes
# the encoder's z_pfx to be rank-predictive of per-device future-residual
# magnitude, addressing the A/B/C ablation finding (2026-08-16/17) that
# z_phys otherwise carries no usable signal for Stage 4C. 0 = off (default);
# set > 0 to activate once validated against a Stage 1-3 retrain.
LAMBDA_ZPHYS_RANK = 0.5   # activated 2026-08-17 after smoke-test validation
ZPHYS_RANK_MARGIN = 0.1
LAMBDA_MULTISTEP_DECAY = 1.0  # Uniform weighting for future rollout steps
LAMBDA_INITIAL_ANCHOR = 0.20  # Anchor on zM(0), zL(0), zC(0) — RAISED from 0.05 (fix zM0=0.9 saturation)

# Debug5 prefix-reconstruction tuning
LOSS_HUBER_BETA = 0.2
STAGE1_LAMBDA_ANCHOR = 0.10   # RAISED from 0.02 — stronger zM initial anchor in Stage 1
STAGE1_LAMBDA_SMOOTH = 0.01
STAGE3_LAMBDA_FUTURE = 1.0
STAGE3_LAMBDA_PREFIX = 0.5
STAGE3_LAMBDA_ODE = 0.2
STAGE3_LAMBDA_ANCHOR = 0.10   # RAISED from 0.02 — keep anchor active through Stage 3
STAGE3_LAMBDA_ZC_SEPARATION = 0.8
STAGE3_LAMBDA_LEAKAGE = 0.4
STAGE3_DECODER_LR_SCALE = 0.2

# zM anchor boost: extra multiplicative weight specifically for zM(0)
# zM represents mechanism damage and must start near 0; zG/zB are fast-trap states
# that are allowed to have large initial values (physical trap occupancy equilibrium).
LAMBDA_ZM_ANCHOR_BOOST = 5.0   # 5x stronger anchor for zM vs zL, zC
# zC anchor boost: moderate additional weight for zC(0) anchor
# Goal: prevent zC from becoming a storage slot for initial device info
# (debug13: soft target, do not set too high; currently zC(0)~0.47 should come down to ~0.15-0.20)
LAMBDA_ZC_ANCHOR_BOOST = 2.0

# Stage 3 temperature-stratified loss: up-weight 325C devices in future rollout loss
# to partially compensate for the 2x RMSE gap at the highest stress condition.
STAGE3_HIGH_TEMP_LOSS_WEIGHT = 1.5   # multiplied onto 325C future-rollout loss
STAGE3_HIGH_TEMP_C = 325             # Celsius threshold for the up-weighting

# ODE rate-constant upper bounds (debug13: bound kM/kL to prevent over-fast saturation)
# kM = softplus(kM_raw); to cap kM at KM_MAX, add a penalty when kM > KM_MAX
KM_MAX = 0.012    # ~2x current learned kM (~0.0053), hard upper guidance not a wall
KL_MAX = 0.010    # similar bound for leakage path rate

# Leakage confidence weighting for Stage3 mean-modeling
LEAKAGE_CONFIDENCE_MARGIN = 0.25
LEAKAGE_CONFIDENCE_MIN = 0.25
LEAKAGE_CONFIDENCE_MAX = 1.0

# Stronger prefix-only separation for zC so the cumulative driver does not
# collapse into the observation prefix too early.
STAGE3_ZC_PREFIX_MARGIN = 0.03

# Bounded device alpha multiplier for zC dynamics (Debug6)
ALPHA_MAX = 2.0

# Gradient clipping
GRAD_CLIP_NORM = 1.0

# Forecasting / evaluation alignment
# Stage 3 uses the same open-loop prefix length as evaluation.
STAGE3_PREFIX_LEN = 4

# Device for PyTorch
TORCH_DEVICE = "cuda"   # will fall back to "cpu" if CUDA unavailable

# ---------------------------------------------------------------------------
# Data splitting (device-wise)
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Minimum observed time points a device must have to be included
MIN_VALID_TIMEPOINTS = 4

# Outlier removal: maximum allowed ratio |x(t)/x(ref) - 1| for absolute params
OUTLIER_RATIO_THRESHOLD = 5.0   # flag if parameter changes >500% from initial

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
# Prediction horizons (hours) for evaluation metrics
EVAL_HORIZONS_H = [100, 500, 1000, 2000]
N_RANDOM_SEEDS = 5   # seeds for stability check

# ---------------------------------------------------------------------------
# Phase 3: Generator prefix context conditioning
# ---------------------------------------------------------------------------
# When True, the generator receives the last prefix encoder state z_prefix_last
# and the log-time at the prefix end as additional conditioning inputs.
GENERATOR_USE_PREFIX_CONTEXT = True
# z0 perturbation: exp(GENERATOR_Z0_PERT_LOGSCALE_INIT) ≈ 0.135 → max Δz0 per dim
GENERATOR_Z0_PERT_LOGSCALE_INIT = -2.0
# If True: generator outputs alpha=1.0 (use when alpha is not identifiable)
GENERATOR_FIXED_ALPHA  = False
# Alpha range clamp: prevents runaway diversity during Stage 4
GENERATOR_ALPHA_MIN    = 0.5
GENERATOR_ALPHA_MAX    = 2.0
# Stage 4 alpha diversity: set to 0 — verify identifiability before enabling
STAGE4_ALPHA_MIN_STD        = 0.05
STAGE4_ALPHA_DIVERSITY_WEIGHT = 0.0

# ---------------------------------------------------------------------------
# Phase 4: Stage 5 generative checkpoint scoring
# Checkpoint selected by:  Score = CRPS + λ_w1*W1(Δx) + λ_cov*|Cov90-0.9| + λ_phys*phys_viol
# ---------------------------------------------------------------------------
STAGE5_W1_WEIGHT        = 0.5
STAGE5_COVERAGE_WEIGHT  = 2.0
STAGE5_PHYSICS_WEIGHT   = 1.0
STAGE5_N_GEN_SAMPLES    = 20   # samples per device for generative validation

# ---------------------------------------------------------------------------
# Phase 2: Stochastic residual baseline
# ---------------------------------------------------------------------------
STOCH_RESIDUAL_N_SAMPLES = 100  # trajectories to generate for probabilistic evaluation
