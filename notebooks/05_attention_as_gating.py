# %% [markdown]
# # 05 — Attention as Selective Neural Gating
#
# The Transformer's attention mechanism looks uncannily like
# theories of attention in primate visual cortex. In both:
#
# * Many possible "sources" compete to influence the next stage
#   of processing.
# * A **top-down bias** (a query / a goal) up-weights sources that
#   match it and down-weights those that don't.
# * The winners are gated through; the losers are suppressed.
#
# This is the essence of the **biased competition** model (Desimone
# & Duncan, 1995) and of **feature-similarity gain** (Treue &
# Martínez-Trujillo, 1999). The Transformer equation
#
# $$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d_k}}\right) V$$
#
# is essentially biased competition made differentiable and parallel.
#
# In this notebook we:
#
# 1. Build attention from scratch and inspect its weights.
# 2. Run a tiny **biased competition** simulation to show the
#    same dynamics appear in a simple neural-ish model.
# 3. Compare what attention patterns look like when the query
#    changes — the same key/value pool, different top-down biases.

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

torch.manual_seed(0)
np.random.seed(0)

# %% [markdown]
# ## 1. Scaled dot-product attention from scratch

# %%
def scaled_dot_product_attention(Q, K, V):
    """
    Q: (batch, n_query, d_k)
