"""
Smoke tests for neuro-ai-bridge
================================

A small battery of tests that exercise every sub-package.  They
don't aim to verify correctness of the algorithms — only that all
modules import cleanly, run without error, and produce tensors /
arrays of the expected shapes.

Run from the repo root with:

    pytest tests/            # if you have pytest installed
    python -m tests.test_smoke   # plain-python fallback

Author: Dimitri Romanov
Project: neuro-ai-bridge
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Plasticity
# ---------------------------------------------------------------------------
def test_hebbian_runs():
    from src.plasticity import HebbianNetwork, OjaNetwork

    patterns = np.random.randn(5, 8)
    for Net in (HebbianNetwork, OjaNetwork):
        net = Net(n_input=8, n_output=4, learning_rate=0.01)
        history = net.train(patterns, n_epochs=10)
        assert len(history) > 0
        assert net.weights.shape == (8, 4)


def test_bcm_develops_selectivity():
    from src.plasticity import BCMNetwork

    patterns = np.eye(6, 12) + 0.05 * np.random.randn(6, 12)
    net = BCMNetwork(n_input=12, n_output=4,
                     learning_rate=0.02, theta_rate=0.05)
    net.train(patterns, n_epochs=50)
    sel = net.selectivity(patterns)
    assert sel.shape == (4,)
    assert (sel >= 0).all() and (sel <= 1).all()


def test_stdp_updates_weights():
    from src.plasticity import STDPSynapse, STDPNetwork

    syn = STDPSynapse()
    w0 = syn.weight
    syn.update(delta_t=5.0)    # LTP
    syn.update(delta_t=-5.0)   # LTD
    assert syn.weight != w0

    net = STDPNetwork(n_neurons=6, connectivity=0.5)
    for t in range(50):
        net.step(float(t), external_input=np.random.rand(6))
