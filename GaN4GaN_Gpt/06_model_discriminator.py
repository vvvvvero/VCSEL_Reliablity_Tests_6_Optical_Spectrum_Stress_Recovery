"""
06_model_discriminator.py
=========================
Discriminator pair for the PI-TimeGAN framework.

Two complementary discriminators are used:
  1. LatentDiscriminator  — distinguishes real (encoder-produced) latent
     trajectories from fake (generator + ODE) trajectories.
  2. ObservationDiscriminator — distinguishes real electrical degradation
     trajectories from decoded generated trajectories.

Both use a small GRU to model temporal structure, followed by a binary
output.  Architectures are intentionally small to prevent memorisation
of the ~150 training trajectories.

Design notes
------------
- Discriminator input includes condition c = [T_norm] so the discriminators
  are aware of the temperature group.
- Both discriminators produce a single scalar per trajectory (not per step),
  following the common practice in TimeGAN.
- Input normalisation: features are assumed already normalised.
- No spectral normalisation added in V1 (add if training instability observed).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


# ---------------------------------------------------------------------------
# Utility: pack masked sequences for GRU
# ---------------------------------------------------------------------------

def _masked_gru_output(
    gru: nn.GRU,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Run a GRU on x and return the hidden state at the last valid time step
    for each batch item.

    Args:
        gru  : nn.GRU (batch_first=True)
        x    : (B, T, input_dim)
        mask : (B, T) bool, True = valid

    Returns:
        h_last : (B, hidden_dim)
    """
    x = torch.nan_to_num(x, nan=0.0)
    h_out, _ = gru(x)               # (B, T, hidden)
    B, T, H = h_out.shape

    # Find last valid step per device
    last_idx = torch.zeros(B, dtype=torch.long, device=x.device)
    for b in range(B):
        valid = mask[b].nonzero(as_tuple=False)
        if len(valid) > 0:
            last_idx[b] = valid[-1, 0]

    idx = last_idx.view(B, 1, 1).expand(B, 1, H)
    h_last = h_out.gather(1, idx).squeeze(1)   # (B, H)
    return h_last


# ---------------------------------------------------------------------------
# Latent discriminator
# ---------------------------------------------------------------------------

class LatentDiscriminator(nn.Module):
    """
    Distinguishes real (encoder-produced) vs. fake (generator+ODE) latent
    trajectories.

    Input per time step: [zG, zB, zM, zL, zC, T_norm] = 6 dims
    """

    def __init__(
        self,
        input_dim:  int = cfg.LATENT_DIM + 1,   # 6
        hidden_dim: int = cfg.DISC_HIDDEN_DIM,
        num_layers: int = cfg.DISC_GRU_LAYERS,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        z_traj: torch.Tensor,
        T_K:    torch.Tensor,
        mask:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_traj : (B, T, 5)   latent trajectory
            T_K    : (B,)        temperature [K]
            mask   : (B, T)      bool, True = valid time step

        Returns:
            logit  : (B,)        unnormalised score (higher = more real)
        """
        B, T, _ = z_traj.shape
        T_norm = ((T_K - cfg.T_REF_K) / cfg.T_REF_K).view(B, 1, 1).expand(B, T, 1)
        inp = torch.cat([z_traj, T_norm], dim=2)         # (B, T, 6)

        h_last = _masked_gru_output(self.gru, inp, mask) # (B, hidden)
        logit  = self.head(h_last).squeeze(1)             # (B,)
        return logit


# ---------------------------------------------------------------------------
# Observation discriminator
# ---------------------------------------------------------------------------

class ObservationDiscriminator(nn.Module):
    """
    Distinguishes real electrical degradation trajectories from decoded
    generated trajectories.

    Input per time step: [x1..x6, T_norm] = 7 dims
    """

    def __init__(
        self,
        input_dim:  int = cfg.FEATURE_DIM + 1,   # 7
        hidden_dim: int = cfg.DISC_HIDDEN_DIM,
        num_layers: int = cfg.DISC_GRU_LAYERS,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x_traj: torch.Tensor,
        T_K:    torch.Tensor,
        mask:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_traj : (B, T, 6)  degradation feature trajectory
            T_K    : (B,)
            mask   : (B, T)

        Returns:
            logit  : (B,)
        """
        B, T, _ = x_traj.shape
        T_norm = ((T_K - cfg.T_REF_K) / cfg.T_REF_K).view(B, 1, 1).expand(B, T, 1)
        inp = torch.cat([x_traj, T_norm], dim=2)         # (B, T, 7)

        h_last = _masked_gru_output(self.gru, inp, mask) # (B, hidden)
        logit  = self.head(h_last).squeeze(1)             # (B,)
        return logit


# ---------------------------------------------------------------------------
# Combined discriminator wrapper
# ---------------------------------------------------------------------------

class PITimeGANDiscriminator(nn.Module):
    """
    Wrapper that holds both discriminators.

    Usage
    -----
        disc = PITimeGANDiscriminator()
        logit_lat_real = disc.latent(z_real, T_K, mask)
        logit_obs_real = disc.observation(x_real, T_K, mask)
    """

    def __init__(self):
        super().__init__()
        self.latent      = LatentDiscriminator()
        self.observation = ObservationDiscriminator()

    def parameters_latent(self):
        return self.latent.parameters()

    def parameters_observation(self):
        return self.observation.parameters()
