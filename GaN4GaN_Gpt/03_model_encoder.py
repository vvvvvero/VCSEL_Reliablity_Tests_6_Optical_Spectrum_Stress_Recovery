"""
03_model_encoder.py
===================
GRU-based encoder that maps observable electrical degradation sequences to
physics-informed latent states.

Architecture
------------
Input at each time step t:
  [x1(t), x2(t), x3(t), x4(t), x5(t), x6(t),  T_norm,  Δlog_t]
  = 8-dimensional vector

GRU: input_dim=8, hidden=16, layers=2, unidirectional (no future leakage)

Output head:
  Linear(16 → 5) + Sigmoid  →  z ∈ [0,1]^5

The encoder produces ONE latent state per time step.
At inference, only early time steps (seen measurements) are fed in, and
the ODE propagates the last estimated state into the future.

Missing-timestep handling
--------------------------
Where a time step has no valid observation (mask=False), the corresponding
encoder output is not used in any loss.  The GRU hidden state is still
propagated through (using the last valid feature vector or zeros).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class PhysicsEncoder(nn.Module):
    """
    Unidirectional GRU encoder: sequence → physics latent states.

    Parameters
    ----------
    input_dim  : int  – dimension of input features per time step (default 8)
    hidden_dim : int  – GRU hidden units (default 16)
    num_layers : int  – GRU layers (default 2)
    latent_dim : int  – output latent dimension (default 5)
    """

    def __init__(
        self,
        input_dim:  int = cfg.ENCODER_INPUT_DIM,
        hidden_dim: int = cfg.GRU_HIDDEN_DIM,
        num_layers: int = cfg.GRU_NUM_LAYERS,
        latent_dim: int = cfg.LATENT_DIM,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )

        # Projection to latent space
        self.proj = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        enc_input: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            enc_input : (B, T, 8)  encoder input (x1..x6, T_norm, Δlog_t)
            mask      : (B, T)  bool, True where observation is valid
                        If None, all steps are treated as valid.

        Returns:
            z       : (B, T, 5)  sigmoid-bounded latent states ∈ [0,1]^5
            h_last  : (num_layers, B, hidden_dim)  final GRU hidden state
        """
        B, T, _ = enc_input.shape

        # Replace NaN in input with 0 (masked-out positions)
        x = enc_input.clone()
        x = torch.nan_to_num(x, nan=0.0)

        # Forward GRU
        h_out, h_last = self.gru(x)   # h_out: (B, T, hidden_dim)

        # Project to latent space and bound to [0,1]
        z = torch.sigmoid(self.proj(h_out))   # (B, T, 5)

        return z, h_last

    def encode_prefix(
        self,
        enc_input: torch.Tensor,
        prefix_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode only the first `prefix_len` time steps.
        Returns (z_prefix, h_last) where z_prefix has shape (B, prefix_len, 5).
        Useful for inference mode: encode early data, predict future via ODE.
        """
        z_full, h_last = self.forward(enc_input[:, :prefix_len, :])
        return z_full, h_last

    def get_last_valid_state(
        self,
        enc_input: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run full-sequence encoding and return the last valid latent state
        for each device (used as initial state for future ODE integration).

        Args:
            enc_input : (B, T, 8)
            mask      : (B, T) bool

        Returns:
            z_init    : (B, 5)
        """
        z, _ = self.forward(enc_input, mask)   # (B, T, 5)
        B, T, D = z.shape

        # Find last valid index per device
        # mask: (B,T), find last True index
        # If no valid index, default to t=0
        last_idx = torch.zeros(B, dtype=torch.long, device=z.device)
        for b in range(B):
            valid_steps = mask[b].nonzero(as_tuple=False)
            if len(valid_steps) > 0:
                last_idx[b] = valid_steps[-1, 0]

        # Gather: z_init[b] = z[b, last_idx[b], :]
        idx_expanded = last_idx.view(B, 1, 1).expand(B, 1, D)
        z_init = z.gather(1, idx_expanded).squeeze(1)  # (B, 5)
        return z_init


class InitialStateEncoder(nn.Module):
    """
    Lightweight encoder that produces an initial latent state z0 purely
    from the initial static features x0_static and temperature.
    Used for devices where no sequence data is available at inference.

    MLP: [x0_static (6), T_norm (1)] → z0 (5 via sigmoid)
    """

    def __init__(self,
                 input_dim: int = cfg.FEATURE_DIM + 1,
                 hidden_dim: int = 16,
                 latent_dim: int = cfg.LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self,
                x0_static: torch.Tensor,
                T_K: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x0_static : (B, 6)
            T_K       : (B,)

        Returns:
            z0        : (B, 5)  ∈ [0,1]^5
        """
        T_norm = ((T_K - cfg.T_REF_K) / cfg.T_REF_K).unsqueeze(1)
        inp = torch.cat([x0_static, T_norm], dim=1)
        return torch.sigmoid(self.net(inp))
