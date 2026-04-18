"""
Variational Free-Energy Autoencoder
====================================

A variational autoencoder (VAE) re-interpreted through the lens of
Karl Friston's Free Energy Principle.

The ELBO objective that trains a VAE

        L = E_q[log p(x | z)]  -  KL[q(z | x) || p(z)]

is exactly the negative of variational free energy:

        F = -L
          = (accuracy term)  +  (complexity term)
          = prediction error +  divergence from prior

Minimising F = maximising evidence lower bound = the brain doing
Bayesian inference over hidden causes.  This module therefore
serves two purposes: (a) a working VAE you can train, and (b) an
explicit free-energy decomposition that makes the neural-Bayesian
connection visible.

References
----------
Friston, K. (2010). The free-energy principle: a unified brain
    theory? Nature Reviews Neuroscience, 11(2), 127-138.
Kingma, D.P. & Welling, M. (2013). Auto-encoding variational Bayes.

Author: Dimitri Romanov
Project: neuro-ai-bridge
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def free_energy_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Variational free energy for a Gaussian-latent VAE.

        F = -ELBO
          = recon_error + β · KL[q(z|x) || N(0, I)]

    Parameters
    ----------
    x, x_recon : torch.Tensor
        Original input and its reconstruction, shape (B, D).
    mu, logvar : torch.Tensor
        Encoder outputs: posterior mean and log-variance, shape (B, L).
    beta : float
        KL weighting (β-VAE).  β = 1 recovers the standard ELBO.
    reduction : {"mean", "sum"}
        How to aggregate across the batch.

    Returns
    -------
    F : torch.Tensor     scalar free energy
    recon : torch.Tensor scalar reconstruction term (accuracy)
    kl : torch.Tensor    scalar KL divergence term (complexity)
    """
    # Gaussian log-likelihood with unit variance ≡ ½·MSE + const
    recon = 0.5 * (x_recon - x).pow(2).sum(dim=-1)

    # KL[q(z|x) || N(0, I)] for diagonal Gaussian q
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)

    F = recon + beta * kl

    if reduction == "mean":
        return F.mean(), recon.mean(), kl.mean()
    elif reduction == "sum":
        return F.sum(), recon.sum(), kl.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


class FreeEnergyVAE(nn.Module):
    """
    A small MLP-based VAE framed as a free-energy minimiser.

    Parameters
    ----------
    input_size : int
        Dimensionality of x.
    hidden_size : int
        Hidden-layer width in encoder and decoder.
    latent_size : int
        Dimensionality of z.
    beta : float
        KL weight (for β-VAE style disentanglement).
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 256,
        latent_size: int = 20,
        beta: float = 1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.beta = float(beta)

        # Encoder (recognition model q(z | x))
        self.enc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)

        # Decoder (generative model p(x | z))
        self.dec = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, input_size),
        )

    # ------------------------------------------------------------------
    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterise(
        mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode → sample z → decode. Returns (x_recon, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Variational free-energy loss on a batch."""
        x_recon, mu, logvar = self.forward(x)
        return free_energy_loss(
            x, x_recon, mu, logvar, beta=self.beta
        )


if __name__ == "__main__":
    torch.manual_seed(0)

    # Tiny smoke test with random data
    batch, dim = 64, 50
    model = FreeEnergyVAE(input_size=dim, hidden_size=64, latent_size=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.rand(batch, dim)

    print("=" * 60)
    print("Neuro-AI Bridge: Free-energy minimisation (β-VAE style)")
    print("=" * 60)
    print(f"{'epoch':>6} {'F':>10} {'recon':>10} {'KL':>10}")
    for epoch in range(30):
        opt.zero_grad()
        F_val, recon, kl = model.loss(x)
        F_val.backward()
        opt.step()
        if epoch % 5 == 0:
            print(f"{epoch:>6d} {F_val.item():>10.4f} "
                  f"{recon.item():>10.4f} {kl.item():>10.4f}")
    print("\nFree energy ↓ = better evidence lower bound — the model "
          "is learning a generative explanation of the input.")
