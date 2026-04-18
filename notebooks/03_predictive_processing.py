# %% [markdown]
# # 03 — Predictive Processing & the Free Energy Principle
#
# One of the most influential ideas in contemporary neuroscience is
# that **the brain is a prediction machine**. Instead of passively
# processing sensory input, it continuously generates predictions
# about what it expects to see, hear, and feel — and learning is
# driven by the mismatch (the *prediction error*) between prediction
# and reality.
#
# This idea gives us two complementary computational frameworks:
#
# 1. **Hierarchical predictive coding** (Rao & Ballard, 1999) —
#    stacked layers where each predicts the one below; errors flow
#    upward, predictions flow downward.
# 2. **Variational free energy** (Friston, 2010) — a single scalar
#    objective whose minimisation unifies perception, learning and
#    action. The variational autoencoder (VAE) is its most popular
#    machine-learning incarnation.
#
# In this notebook we play with both.

# %%
from __future__ import annotations
import sys
import pathlib

ROOT = pathlib.Path().resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.predictive_coding import (
    PredictiveCodingNetwork,
    FreeEnergyVAE,
    free_energy_loss,
)

torch.manual_seed(0)
np.random.seed(0)

# %% [markdown]
# ## 1. Hierarchical predictive coding
#
# The core move: each layer $k$ holds latent activity $x_k$ which
# is updated to **minimise the total squared prediction error**
# across the hierarchy. Weights are learned *afterwards* with
# standard gradient descent.
#
# Let's train a 3-layer PC network on toy data and watch the free
# energy come down.

# %%
X = torch.randn(256, 20)                       # "sensory" data

model = PredictiveCodingNetwork(
    layer_sizes=[20, 12, 6],
    inference_steps=30,
    inference_lr=0.1,
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

F_history = []
for epoch in range(80):
    opt.zero_grad()
    F_val, _ = model(X)
    F_val.backward()
    opt.step()
    F_history.append(F_val.item())

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(F_history)
ax.set_xlabel("epoch")
ax.set_ylabel("Free energy  (Σ ½‖ε‖²)")
ax.set_title("PC network — prediction-error energy during learning")
plt.tight_layout()
plt.show()

# %% [markdown]
# A steadily decreasing curve = the network is **learning a
# generative model** of the data.  Higher layers capture
# increasingly abstract regularities; the bottom layer reconstructs
# the input via top-down prediction.

# %% [markdown]
# ## 2. Inference as active iteration
#
# Unlike a feed-forward net, inference in predictive coding is an
# **iterative procedure** — the latents *settle* as prediction
# errors propagate. Here's a single inference trajectory.

# %%
x_one = X[:1].clone()                          # single example
latents = model._init_latents(1, x_one.device)

F_trajectory = []
for step in range(model.inference_steps):
    F_val = model.free_energy(x_one, latents)
    F_trajectory.append(F_val.item())
    grads = torch.autograd.grad(F_val, latents)
    latents = [
        (z - model.inference_lr * g).detach().requires_grad_(True)
        for z, g in zip(latents, grads)
    ]

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(F_trajectory, marker="o", ms=3)
ax.set_xlabel("inference step")
ax.set_ylabel("free energy for 1 datum")
ax.set_title("Perception as gradient descent on prediction error")
plt.tight_layout()
plt.show()

# %% [markdown]
# Brains are thought to do this via cortico-cortical feedback loops
# on millisecond timescales. The convergence curve here is a
# caricature of that process.

# %% [markdown]
# ## 3. The free-energy VAE
#
# The same objective — minimise variational free energy — gives us
# a standard VAE when we parameterise the posterior and likelihood
# as neural networks.
#
# $$F = \underbrace{\mathbb{E}_{q(z|x)}[-\log p(x|z)]}_{\text{reconstruction error}}
# \;+\;
# \beta \cdot \underbrace{\mathrm{KL}[q(z|x) \,\|\, p(z)]}_{\text{complexity / prior-fit}}$$
#
# Let's train one on a small binary-digit toy to make the two
# terms' behaviour visible.

# %%
# Build a toy "digit" dataset: random binary patterns + small noise
def make_toy_digits(n: int = 1024, dim: int = 64, n_proto: int = 6):
    protos = (torch.rand(n_proto, dim) > 0.5).float()
    idx = torch.randint(0, n_proto, (n,))
    X = protos[idx].clone()
    flip = torch.rand_like(X) < 0.05
    X = torch.where(flip, 1 - X, X)
    return X


X_toy = make_toy_digits()
vae = FreeEnergyVAE(input_size=64, hidden_size=48, latent_size=8, beta=1.0)
opt = torch.optim.Adam(vae.parameters(), lr=3e-3)

F_list, recon_list, kl_list = [], [], []
for epoch in range(120):
    perm = torch.randperm(X_toy.shape[0])
    e_F = e_R = e_K = 0.0
    for i in range(0, X_toy.shape[0], 64):
        b = perm[i : i + 64]
        Fb, Rb, Kb = vae.loss(X_toy[b])
        opt.zero_grad()
        Fb.backward()
        opt.step()
        e_F += Fb.item(); e_R += Rb.item(); e_K += Kb.item()
    F_list.append(e_F); recon_list.append(e_R); kl_list.append(e_K)

fig, ax = plt.subplots(1, 3, figsize=(11, 3))
ax[0].plot(F_list);      ax[0].set_title("Total free energy F")
ax[1].plot(recon_list);  ax[1].set_title("Reconstruction  (accuracy)")
ax[2].plot(kl_list);     ax[2].set_title("KL divergence  (complexity)")
for a in ax:
    a.set_xlabel("epoch")
plt.tight_layout()
plt.show()

# %% [markdown]
# Notice the classic dynamic:
#
# * The reconstruction term falls fast as the decoder learns to
#   replicate the patterns.
# * The KL term can **rise** initially as the encoder discovers
#   useful latent structure, then settle as priors and posterior
#   reach a compromise.
# * Together, F minimises — the network is finding the best tradeoff
#   between *fitting the data* and *staying close to its prior
#   beliefs*. This is exactly Friston's "Bayesian brain" in code.

# %% [markdown]
# ## 4. Take-aways
#
# * Predictive coding and VAEs are two faces of the same
#   variational-Bayesian coin.
# * **Perception = inference**: latent states settle to minimise
#   prediction error. The brain likely implements something
#   analogous with bidirectional cortical signalling.
# * **Learning = updating the generative model** so that future
#   inferences are easier.
#
# ### Suggested next steps
#
# * Replace the toy digits with real MNIST and compare learned
#   latent spaces of PC vs. VAE.
# * Add **active inference**: let the model choose actions that
#   reduce *expected* future free energy (links to RL).
# * Explore **predictive coding as training algorithm**: recent
#   work shows local PC updates can rival backprop on real tasks.
