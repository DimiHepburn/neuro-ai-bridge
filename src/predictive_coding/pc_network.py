"""
Hierarchical Predictive Coding Network
=======================================

A PyTorch implementation of the Rao & Ballard (1999) predictive
coding scheme for hierarchical perception.

Key idea
--------
Each layer L_k maintains a latent representation x_k that tries to
*predict* the representation of the layer below, x_{k-1}:

        x̂_{k-1} = f(W_k · x_k)

The **prediction error** is what gets passed upward:

        ε_{k-1} = x_{k-1} - x̂_{k-1}

Inference proceeds by iteratively updating each x_k to minimise
the total squared prediction error across the hierarchy.  This
gives a biologically plausible alternative to backpropagation in
which information flow is local and bidirectional (top-down
predictions, bottom-up errors).

References
----------
Rao, R.P.N. & Ballard, D.H. (1999). Predictive coding in the visual
    cortex. Nature Neuroscience, 2(1), 79-87.
Bogacz, R. (2017). A tutorial on the free-energy framework for
    modelling perception and learning. J. Mathematical Psychology.

Author: Dimitri Romanov
Project: neuro-ai-bridge
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCLayer(nn.Module):
    """
    One level of the predictive coding hierarchy.

    Parameters
    ----------
    size_below : int
        Dimension of the layer this one predicts (x_{k-1}).
    size_here : int
        Dimension of this layer's latent representation (x_k).
    activation : callable
        Non-linearity applied to the top-down prediction.
    """

    def __init__(
        self,
        size_below: int,
        size_here: int,
        activation=torch.tanh,
    ):
        super().__init__()
        self.W = nn.Linear(size_here, size_below, bias=True)
        self.activation = activation

    def predict(self, x_here: torch.Tensor) -> torch.Tensor:
        """Top-down prediction of the layer below."""
        return self.activation(self.W(x_here))


class PredictiveCodingNetwork(nn.Module):
    """
    Hierarchical predictive coding network.

    Inference is performed by **gradient descent on the total
    prediction-error energy** with respect to the latent variables
    x_k, holding weights fixed.  Weights are updated afterwards with
    a standard optimiser.

    Parameters
    ----------
    layer_sizes : list[int]
        Sizes from input (bottom) to top, e.g. [784, 256, 64].
    inference_steps : int
        Number of latent-update iterations per forward call.
    inference_lr : float
        Step size for latent updates.
    activation : callable
        Non-linearity for top-down predictions.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        inference_steps: int = 20,
        inference_lr: float = 0.1,
        activation=torch.tanh,
    ):
        super().__init__()
        assert len(layer_sizes) >= 2, "Need at least input + 1 latent layer"
        self.layer_sizes = layer_sizes
        self.inference_steps = inference_steps
        self.inference_lr = inference_lr

        # One PCLayer per upward connection
        self.layers = nn.ModuleList([
            PCLayer(layer_sizes[k], layer_sizes[k + 1],
                    activation=activation)
            for k in range(len(layer_sizes) - 1)
        ])

    def _init_latents(
        self, batch_size: int, device: torch.device
    ) -> List[torch.Tensor]:
        """Zero-initialise all latent tensors above the input layer."""
        return [
            torch.zeros(batch_size, sz, device=device, requires_grad=True)
            for sz in self.layer_sizes[1:]
        ]

    def free_energy(
        self,
        x_input: torch.Tensor,
        latents: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Total squared prediction-error energy across the hierarchy.

        F = Σ_k ½ · ‖x_{k-1} - x̂_{k-1}‖²
        """
        total = 0.0
        below = x_input
        for layer, here in zip(self.layers, latents):
            pred = layer.predict(here)
            err = below - pred
            total = total + 0.5 * (err ** 2).sum(dim=-1).mean()
            below = here
        return total

    def infer(
        self, x_input: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Iterative latent-state inference via gradient descent on
        the free-energy functional (weights fixed).
        """
        latents = self._init_latents(x_input.shape[0], x_input.device)

        for _ in range(self.inference_steps):
            F_val = self.free_energy(x_input, latents)
            grads = torch.autograd.grad(F_val, latents, create_graph=False)
            latents = [
                (z - self.inference_lr * g).detach().requires_grad_(True)
                for z, g in zip(latents, grads)
            ]

        return latents

    def forward(
        self, x_input: torch.Tensor
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run inference, then return (free energy, converged latents).

        The returned free-energy is differentiable w.r.t. the network
        weights, so the standard training loop is:

            F, _ = model(x)
            F.backward()
            optimiser.step()
        """
        latents = self.infer(x_input)
        F_val = self.free_energy(x_input, latents)
        return F_val, latents


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cpu")

    # Toy data: random "images" of size 20
    x = torch.randn(64, 20, device=device)

    model = PredictiveCodingNetwork(
        layer_sizes=[20, 12, 6],
        inference_steps=30,
        inference_lr=0.1,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("=" * 60)
    print("Neuro-AI Bridge: Predictive-coding free-energy minimisation")
    print("=" * 60)
    for epoch in range(20):
        opt.zero_grad()
        F_val, _ = model(x)
        F_val.backward()
        opt.step()
        if epoch % 4 == 0:
            print(f"  epoch {epoch:2d}   free energy = {F_val.item():.4f}")

    print("\nFree energy should decrease — the network is learning a "
          "generative model of the input.")
