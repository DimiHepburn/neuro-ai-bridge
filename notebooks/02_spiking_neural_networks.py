# %% [markdown]
# # 02 — Spiking Neural Networks
#
# Real neurons don't transmit continuous real numbers — they emit
# brief, all-or-nothing electrical events called **spikes**. Spiking
# Neural Networks (SNNs) are the artificial analogue:
#
# * Information is encoded in the **timing** and **rate** of spikes.
# * Computation is **event-driven** → potentially very energy-efficient
#   on neuromorphic hardware (Intel Loihi, IBM TrueNorth).
# * They sit between classical neuroscience models and deep learning.
#
# In this notebook we:
#
# 1. Play with a single **Leaky Integrate-and-Fire (LIF)** neuron.
# 2. Explore **rate coding** — how pixel intensity becomes spike
#    probability.
# 3. Train a 2-layer **spiking classifier** end-to-end with
#    surrogate gradients on a tiny synthetic dataset.

# %%
from __future__ import annotations
import sys
import pathlib

ROOT = pathlib.Path().resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.snn import LIFNeuron, LIFLayer, SpikingClassifier
from src.snn.snn_classifier import poisson_encode

torch.manual_seed(0)
np.random.seed(0)

# %% [markdown]
# ## 1. A single LIF neuron
#
# The LIF model is the simplest plausible spiking neuron:
#
# $$\tau \frac{dV}{dt} = -(V - V_{\text{rest}}) + R\, I(t),
# \quad V \ge V_{\text{thresh}} \Rightarrow \text{spike}; V \leftarrow V_{\text{reset}}$$
#
# Let's watch its membrane potential evolve under a constant drive.

# %%
lif = LIFNeuron(beta=0.9, threshold=1.0)
mem = lif.init_state(torch.Size((1,)))

n_steps = 200
drive = torch.full((1,), 0.25)

mem_trace, spike_trace = [], []
for _ in range(n_steps):
    s, mem = lif(drive, mem)
    mem_trace.append(mem.item())
    spike_trace.append(s.item())

fig, ax = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
ax[0].plot(mem_trace)
ax[0].axhline(1.0, ls="--", color="red", label="threshold")
ax[0].set_ylabel("V (membrane)")
ax[0].legend()
ax[1].vlines(
    [i for i, s in enumerate(spike_trace) if s > 0],
    0, 1, color="k",
)
ax[1].set_ylabel("spike")
ax[1].set_xlabel("timestep")
ax[1].set_yticks([])
fig.suptitle("LIF neuron under constant drive — classic saw-tooth spiking")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Rate coding
#
# A grayscale pixel of intensity 0.8 can be encoded as a neuron
# that spikes with probability 0.8 at each timestep — the average
# spike count over a window recovers the intensity.
#
# Below we encode a simple greyscale gradient and visualise the
# resulting spike raster.

# %%
x = torch.linspace(0, 1, 16).unsqueeze(0)       # 1 x 16 greyscale "image"
spikes = poisson_encode(x, n_steps=40)           # (T, 1, 16)
raster = spikes.squeeze(1).T.numpy()             # pixel × time

fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].imshow(x.numpy(), aspect="auto", cmap="gray")
ax[0].set_title("'image' (16 pixels, 0→1)")
ax[0].set_yticks([])
ax[1].imshow(raster, aspect="auto", cmap="Greys", interpolation="nearest")
ax[1].set_title("Poisson spike raster  (row = pixel, col = time)")
ax[1].set_xlabel("timestep")
ax[1].set_ylabel("pixel")
plt.tight_layout()
plt.show()

# %% [markdown]
# Brighter pixels spike more often — exactly as in retinal ganglion
# cells and V1 simple cells under constant luminance.

# %% [markdown]
# ## 3. Train a spiking classifier
#
# We'll build a tiny synthetic task (classify random 4-class
# Gaussian blobs in 16-D) and train a 2-layer SNN with
# **surrogate gradients** — the trick that makes the non-
# differentiable spike function trainable end-to-end.

# %%
def make_blobs(n_per_class: int = 128, dim: int = 16, n_classes: int = 4):
    centres = torch.randn(n_classes, dim) * 2.0
    X, y = [], []
    for c in range(n_classes):
        X.append(centres[c] + 0.5 * torch.randn(n_per_class, dim))
        y.append(torch.full((n_per_class,), c, dtype=torch.long))
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)
    # Normalise to [0, 1] so Poisson encoding works
    X = (X - X.min()) / (X.max() - X.min() + 1e-9)
    idx = torch.randperm(X.shape[0])
    return X[idx], y[idx]


X_train, y_train = make_blobs(n_per_class=128)
X_test, y_test = make_blobs(n_per_class=32)

model = SpikingClassifier(
    input_size=16, hidden_size=48, n_classes=4,
    n_steps=20, beta=0.9, threshold=1.0,
)
opt = torch.optim.Adam(model.parameters(), lr=5e-3)

losses, accs = [], []
for epoch in range(25):
    # Mini-batch training
    perm = torch.randperm(X_train.shape[0])
    epoch_loss = 0.0
    for i in range(0, X_train.shape[0], 64):
        b = perm[i : i + 64]
        loss = model.loss(X_train[b], y_train[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item()

    # Eval
    with torch.no_grad():
        pred = model.predict(X_test)
        acc = (pred == y_test).float().mean().item()
    losses.append(epoch_loss)
    accs.append(acc)
    if epoch % 5 == 0:
        print(f"epoch {epoch:2d} | loss {epoch_loss:7.3f} | test acc {acc:.3f}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].plot(losses)
ax[0].set_title("Training loss")
ax[0].set_xlabel("epoch")
ax[1].plot(accs)
ax[1].set_title("Test accuracy")
ax[1].set_xlabel("epoch")
ax[1].set_ylim(0, 1.05)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Take-aways
#
# * LIF neurons, despite their simplicity, produce realistic
#   spike-train dynamics.
# * Rate coding is a straightforward bridge between pixel
#   intensities and spike trains.
# * Surrogate gradients let us train SNNs the same way we train
#   standard deep networks — they become a drop-in replacement
#   where energy efficiency matters.
#
# ### Suggested next steps
#
# * Swap rate coding for **temporal coding** (the first-spike time
#   carries the information) — much more energy-efficient.
# * Plug the SNN into a **neuromorphic dataset** like N-MNIST.
# * Combine with the STDP rule from notebook 01 to train
#   without surrogate gradients at all.
