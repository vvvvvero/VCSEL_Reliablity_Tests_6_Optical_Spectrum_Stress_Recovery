"""
04_model_decoder.py
===================
Sparse physics-constrained decoder that maps the 5 effective latent states
to the 6 observed degradation variables.

Sparsity mask (rows = x1..x6, cols = zG, zB, zM, zL, zC)
-----------------------------------------------------------
  x1 (ΔVth)   : zG, zB, zM, zC      sign FREE (trap polarity uncertain)
  x2 (IDSS)   : zG, zB, zM, zC      weights ≥ 0
  x3 (RON)    : zG, zB, zM, zC      weights ≥ 0
  x4 (gmmax)  : zG, zB, zM, zC      weights ≥ 0
  x5 (IDLeak) : zB, zL, zC          sign FREE (leakage can rise or fall)
  x6 (IGLeak) : zG, zL, zC          sign FREE (leakage can rise or fall)

The decoder is a single linear layer with:
  - zero-masked entries (hard constraint via registered buffer)
  - non-negative weights for all rows except the free-sign rows (via softplus)
  - free weights (positive and negative) for the rows listed in cfg.FREE_SIGN_ROWS

Optional bias term per feature to capture initial offsets.

Usage
-----
    from 04_model_decoder import SparsePhysicsDecoder
    dec = SparsePhysicsDecoder()
    x_hat = dec(z_traj)   # (B, T, 6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class SparsePhysicsDecoder(nn.Module):
    """
    Sparse linear physics decoder.

    Attributes
    ----------
    W_raw   : (6, 5)   raw weight parameters (unconstrained)
    mask    : (6, 5)   binary sparsity mask (buffer, not trained)
    bias    : (6,)     per-feature bias
    VTH_ROW : int      index of the Vth feature row (sign-free)
    """

    VTH_ROW = cfg.VTH_DECODER_ROW   # 0
    FREE_SIGN_ROWS = set(getattr(cfg, "FREE_SIGN_ROWS", [VTH_ROW]))
    FREE_SIGN_COLUMNS = getattr(cfg, "FREE_SIGN_DECODER_COLUMNS", {})

    def __init__(self):
        super().__init__()

        # Sparsity mask as a non-trainable buffer
        mask = torch.tensor(cfg.DECODER_SPARSITY, dtype=torch.float32)  # (6,5)
        self.register_buffer("mask", mask)

        # Raw weight parameters (6 × 5)
        self.W_raw = nn.Parameter(
            torch.zeros(cfg.FEATURE_DIM, cfg.LATENT_DIM)
        )
        # Bias per output feature
        self.bias = nn.Parameter(torch.zeros(cfg.FEATURE_DIM))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.W_raw, mean=0.0, std=0.01)
        nn.init.zeros_(self.bias)

    def _effective_weight(self) -> torch.Tensor:
        """
        Build the effective weight matrix W ∈ ℝ^{6×5}:
          - entries where mask=0 are forced to 0
          - row VTH_ROW: weights are free (positive or negative)
          - all other rows: weights are non-negative (via softplus)
        """
        W = torch.zeros_like(self.W_raw)

        for row in range(cfg.FEATURE_DIM):
            row_mask = self.mask[row]          # (5,)
            w_raw    = self.W_raw[row]         # (5,)
            free_cols = self.FREE_SIGN_COLUMNS.get(row, set())
            w_eff = torch.empty_like(w_raw)
            for col in range(w_raw.shape[0]):
                if row in self.FREE_SIGN_ROWS and col in free_cols:
                    w_eff[col] = w_raw[col]
                else:
                    w_eff[col] = F.softplus(w_raw[col])
            W[row] = w_eff * row_mask          # zero out masked entries

        return W   # (6, 5)

    def forward(self, z: torch.Tensor, z_ref: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z : (B, T, 5)  or  (B, 5)  latent states ∈ [0,1]^5
            z_ref : optional reference latent state used to decode relative
                    degradation. If provided, decoder predicts from
                    (z - z_ref), enforcing x(z_ref)=0.

        Returns:
            x_hat : (B, T, 6)  or  (B, 6)  reconstructed degradation features
        """
        W = self._effective_weight()   # (6, 5)

        if z_ref is not None:
            if z.dim() == 3 and z_ref.dim() == 2:
                z_ref = z_ref[:, None, :]
            z_effective = z - z_ref
        else:
            z_effective = z

        if z_effective.dim() == 3:
            # z: (B, T, 5)  →  x_hat: (B, T, 6)
            x_hat = z_effective @ W.T   # (B, T, 6)
        else:
            # z: (B, 5)
            x_hat = z_effective @ W.T   # (B, 6)

        return x_hat

    def get_weight_matrix(self) -> torch.Tensor:
        """Return the effective weight matrix as a detached numpy array."""
        with torch.no_grad():
            return self._effective_weight().cpu().numpy()

    def feature_sensitivity(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute ∂x_hat_i / ∂z_j at a given latent point.
        (For sparse linear decoder, this is just the weight matrix,
        but we expose it as a method for compatibility with non-linear decoders.)

        Args:
            z : (B, T, 5)  — unused for linear decoder

        Returns:
            sensitivity : (6, 5)  same as effective weight matrix
        """
        return self._effective_weight().detach()


class BiasOnlyDecoder(nn.Module):
    """
    Trivial decoder that predicts a learnable constant per feature
    (used for ablation baseline).
    """

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(cfg.FEATURE_DIM))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:
            B, T, _ = z.shape
            return self.bias.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        else:
            B, _ = z.shape
            return self.bias.unsqueeze(0).expand(B, -1)
