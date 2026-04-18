# %% [markdown]
# # 04 — Hippocampal Memory & Complementary Learning Systems
#
# The brain appears to run **two** learning systems side-by-side:
#
# | System       | Speed         | Capacity  | Function                          |
# |--------------|---------------|-----------|-----------------------------------|
# | Hippocampus  | Fast (one-shot) | Limited   | Episodic memory, rapid binding   |
# | Neocortex    | Slow (gradual)  | Large     | Statistical regularities, semantics |
#
# The **Complementary Learning Systems** (CLS) theory (McClelland,
# McNaughton & O'Reilly, 1995) proposes that these two systems
# cooperate: the hippocampus rapidly stores episodes, then
# **replays** them to the neocortex during rest / sleep so the
# slower system can consolidate knowledge without overwriting what
# it already knows.
#
# This is *the* neuroscience-inspired solution to **catastrophic
# forgetting** in deep networks — and in fact, it's the idea behind
# experience replay in DQN (Mnih et al., 2015).
#
# In this notebook we:
#
# 1. Play with a hippocampus-style **episodic buffer**.
# 2. Reproduce catastrophic forgetting in a simple classifier.
# 3. Show how **interleaved replay** rescues performance on the
#    old task while the model learns a new one.

# %%
from __future__ import annotations
import sys
import pathlib

ROOT = pathlib.Path().resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.memory import EpisodicBuffer, ReplayBuffer, interleaved_replay_batch

torch.manual_seed(0)
np.random.seed(0)

# %% [markdown]
# ## 1. The episodic buffer — rapid, content-addressable memory
#
# Each write is one-shot; each recall returns the stored episodes
# whose **keys** are most similar to the query. This is roughly how
# CA3 pattern completion is thought to work: a partial cue retrieves
# the full stored episode.

# %%
rng = np.random.default_rng(0)
buf = EpisodicBuffer(capacity=30, key_dim=8, rng=rng)

# Store 150 "experiences" — reservoir sampling keeps a
# representative subset.
for i in range(150):
    key = rng.standard_normal(8)
    buf.write(key, payload={"episode_id": i})

# Cue with a noisy version of one of the stored keys
target = buf.all()[5]
noisy_cue = target.key + 0.3 * rng.standard_normal(8)

top3 = buf.recall(noisy_cue, k=3, metric="cosine")
print("Target episode id :", target.payload["episode_id"])
print("Top-3 recalls     :")
for sim, ep in top3:
    print(f"   sim={sim:+.3f}  id={ep.payload['episode_id']}")

# %% [markdown]
# The top hit should almost always be the target — pattern
# completion in action, despite a noisy cue.

# %% [markdown]
# ## 2. Catastrophic forgetting
#
# Now the classic demonstration.  We'll train a small MLP
# sequentially on **Task A**, then on **Task B**, with *no replay*,
# and watch accuracy on A collapse.

# %%
def make_task(centre: float, n: int = 512, dim: int = 8):
    X = torch.randn(n, dim) + centre
    y = (X.sum(dim=1) > centre * dim).long()
    return X, y


X_A, y_A = make_task(centre=-1.5)
X_B, y_B = make_task(centre=+1.5)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


def accuracy(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(dim=-1) == y).float().mean().item()


def train_sequential(model, epochs_per_task=20, batch=64, replay=None,
                     replay_ratio=0.0):
    """Train A → B. Optionally mix in replay samples during task B."""
    hist = {"A_on_A": [], "A_after_B": [], "B": []}
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    # -------- Task A
    for _ in range(epochs_per_task):
        perm = torch.randperm(X_A.shape[0])
        for i in range(0, X_A.shape[0], batch):
            b = perm[i : i + batch]
            loss = F.cross_entropy(model(X_A[b]), y_A[b])
            opt.zero_grad(); loss.backward(); opt.step()
            if replay is not None:
                for xi, yi in zip(X_A[b], y_A[b]):
                    replay.push((xi.numpy(), int(yi)))
        hist["A_on_A"].append(accuracy(model, X_A, y_A))

    # -------- Task B (optionally with replay of A)
    for _ in range(epochs_per_task):
        perm = torch.randperm(X_B.shape[0])
        for i in range(0, X_B.shape[0], batch):
            b = perm[i : i + batch]
            new_items = list(zip(
                [x.numpy() for x in X_B[b]],
                [int(v) for v in y_B[b]],
            ))
            if replay is not None and replay_ratio > 0.0:
                batch_items = interleaved_replay_batch(
                    new_items, replay, replay_ratio=replay_ratio
                )
            else:
                batch_items = new_items
            xb = torch.tensor(np.stack([x for x, _ in batch_items]),
                              dtype=torch.float32)
            yb = torch.tensor([y for _, y in batch_items], dtype=torch.long)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        hist["A_after_B"].append(accuracy(model, X_A, y_A))
        hist["B"].append(accuracy(model, X_B, y_B))

    return hist


# ----- Scenario 1: no replay
model_noreplay = MLP()
hist_noreplay = train_sequential(model_noreplay)

# ----- Scenario 2: with interleaved replay
model_replay = MLP()
rng2 = np.random.default_rng(1)
replay = ReplayBuffer(capacity=1000, rng=rng2)
hist_replay = train_sequential(model_replay, replay=replay, replay_ratio=0.5)

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

ax[0].plot(hist_noreplay["A_on_A"], label="A (during A)")
ax[0].plot(
    range(len(hist_noreplay["A_on_A"]),
          len(hist_noreplay["A_on_A"]) + len(hist_noreplay["A_after_B"])),
    hist_noreplay["A_after_B"], label="A (during B)", ls="--",
)
ax[0].plot(
    range(len(hist_noreplay["A_on_A"]),
          len(hist_noreplay["A_on_A"]) + len(hist_noreplay["B"])),
    hist_noreplay["B"], label="B", color="tab:red",
)
ax[0].set_title("No replay  →  catastrophic forgetting of A")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("accuracy")
ax[0].set_ylim(0, 1.05); ax[0].legend()

ax[1].plot(hist_replay["A_on_A"], label="A (during A)")
ax[1].plot(
    range(len(hist_replay["A_on_A"]),
          len(hist_replay["A_on_A"]) + len(hist_replay["A_after_B"])),
    hist_replay["A_after_B"], label="A (during B)", ls="--",
)
ax[1].plot(
    range(len(hist_replay["A_on_A"]),
          len(hist_replay["A_on_A"]) + len(hist_replay["B"])),
    hist_replay["B"], label="B", color="tab:red",
)
ax[1].set_title("With interleaved replay  →  A preserved")
ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1.05); ax[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# Without replay, accuracy on task A plummets as the model is
# re-trained on B — the network's weights have been overwritten.
# With interleaved replay of old experiences, both tasks are
# retained at the same time. A direct reflection of what the
# hippocampal–neocortical dialogue is hypothesised to achieve.

# %% [markdown]
# ## 3. Take-aways
#
# * One-shot episodic memory + gradual cortical learning is a
#   surprisingly powerful computational combination.
# * The same trick appears in DQN (Mnih et al., 2015), continual
#   learning, and modern memory-augmented networks (NTM, DNC).
# * There's lots of room to push this further:
#     * **Prioritised replay** (weight by surprise / TD-error).
#     * **Generative replay** — the cortex itself generates
#       pseudo-memories instead of a literal buffer.
#     * **Sharp-wave ripples** as temporally compressed replay.
#
# ### Suggested next steps
#
# * Swap the linear MLP for the SNN classifier from notebook 02 and
#   see if spiking networks forget differently.
# * Replace the FIFO replay buffer with **prioritised replay** and
#   measure the effect.
