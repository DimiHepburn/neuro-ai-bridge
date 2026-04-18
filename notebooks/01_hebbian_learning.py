# %% [markdown]
# # 01 — Hebbian Learning & Synaptic Plasticity
#
# **"Neurons that fire together, wire together."**
#
# In this notebook we compare three classic local learning rules:
#
# 1. **Classical Hebbian** — the oldest rule; strong but unstable.
# 2. **Oja's rule** — a normalised Hebbian variant; extracts the
#    first principal component of the input.
# 3. **BCM** — a sliding-threshold rule that develops *selectivity*,
#    mirroring how V1 simple cells tune to specific orientations.
#
# All three are **local**: a synapse updates using only the activity
# of the two neurons it connects — no global error signal. This is
# biologically realistic and stands in contrast to backpropagation.

# %%
from __future__ import annotations
import sys
import pathlib

# Make the repo root importable from this notebook
ROOT = pathlib.Path().resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.plasticity import HebbianNetwork, OjaNetwork, BCMNetwork

np.random.seed(0)

# %% [markdown]
# ## 1. Weight-norm dynamics
#
# First, let's see how the *total weight magnitude* evolves under
# each rule for the same input. Stability is the key question.

# %%
patterns = np.random.randn(20, 8)

rules = {
    "Hebbian (raw)": HebbianNetwork(8, 4, learning_rate=0.01, decay=0.0),
    "Hebbian + decay": HebbianNetwork(8, 4, learning_rate=0.01, decay=0.005),
    "Oja": OjaNetwork(8, 4, learning_rate=0.01, decay=0.0),
    "BCM": BCMNetwork(8, 4, learning_rate=0.01, theta_rate=0.05),
}

histories = {}
for name, net in rules.items():
    histories[name] = net.train(patterns, n_epochs=100)

fig, ax = plt.subplots(figsize=(7, 4))
for name, h in histories.items():
    ax.plot(h, label=name)
ax.set_xlabel("Update step")
ax.set_ylabel("||W||  (Frobenius norm)")
ax.set_title("Weight-norm evolution under different plasticity rules")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **What to look for**
#
# * Raw Hebbian grows without bound — synapses keep strengthening.
# * Hebbian + decay is bounded but still coarse.
# * Oja self-normalises around a fixed point.
# * BCM stabilises as its sliding threshold catches up with mean
#   post-synaptic activity.

# %% [markdown]
# ## 2. Oja extracts the first principal component
#
# Oja's rule is famous for being a *neural PCA*. Let's verify this
# by training on data whose first principal component is known.

# %%
from numpy.linalg import eigh

# Build data with a clear dominant direction
n_samples, dim = 500, 6
true_pc = np.zeros(dim)
true_pc[0] = 1.0
X = (np.random.randn(n_samples, 1) * 3.0) @ true_pc[None, :]
X += 0.3 * np.random.randn(n_samples, dim)

# Ground-truth PC via covariance eigendecomposition
cov = np.cov(X.T)
eigvals, eigvecs = eigh(cov)
gt_pc = eigvecs[:, np.argmax(eigvals)]

# Train a single-output Oja network
oja = OjaNetwork(n_input=dim, n_output=1, learning_rate=0.005)
oja.train(X, n_epochs=50)
learned = oja.weights[:, 0] / np.linalg.norm(oja.weights[:, 0])

cos = float(np.abs(learned @ gt_pc))
print(f"|cos(learned weight, ground-truth PC1)| = {cos:.3f}")
print("(1.00 = perfectly aligned; sign flips are allowed)")

# %% [markdown]
# ## 3. BCM develops selectivity
#
# Train a BCM network on a small set of "stimuli" and check that
# each output unit ends up responding preferentially to a subset —
# the hallmark of cortical feature selectivity.

# %%
n_patterns, n_in, n_out = 6, 12, 4
stimuli = np.eye(n_patterns, n_in) + 0.05 * np.random.randn(n_patterns, n_in)

bcm = BCMNetwork(n_in, n_out, learning_rate=0.02, theta_rate=0.05)
bcm.train(stimuli, n_epochs=500)

responses = np.array([bcm.forward(p) for p in stimuli])   # (patterns, units)

fig, ax = plt.subplots(figsize=(6, 3.5))
im = ax.imshow(np.abs(responses.T), aspect="auto", cmap="viridis")
ax.set_xlabel("Pattern")
ax.set_ylabel("Output unit")
ax.set_title("BCM tuning matrix — bright cells = strong response")
plt.colorbar(im, ax=ax, label="|response|")
plt.tight_layout()
plt.show()

print("Per-unit selectivity (1 = fully selective, 0 = uniform):")
print(np.round(bcm.selectivity(stimuli), 3))

# %% [markdown]
# ## 4. Take-aways
#
# * Local learning rules can do meaningful computation — PCA
#   (Oja), selectivity (BCM) — without any global error signal.
# * The biological story is a tight analogue of what these rules
#   compute.
# * In modern deep learning, **BPTT / backprop** dominate because
#   they're efficient and flexible — but there is active work on
#   hybrid systems where a Hebbian-style *fast* pathway shapes
#   representations in ways backprop alone struggles to discover
#   (e.g. associative memories, continual learning).
#
# ### Suggested next steps
#
# * Swap the random `patterns` for real image patches and see which
#   rule extracts the most interpretable features.
# * Combine Oja and BCM in a layered network — let Oja do
#   dimensionality reduction, BCM do feature specialisation.
