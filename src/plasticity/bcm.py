"""
Bienenstock-Cooper-Munro (BCM) Learning Rule
=============================================

BCM theory models a *sliding* modification threshold θ_M that
determines whether a synapse undergoes LTP or LTD:

    Δw_ij = η * x_i * φ(y_j, θ_M)
    φ(y, θ) = y * (y - θ)
    θ_M    ← τ⁻¹ * <y²>      (running average of squared post-activity)

When post-synaptic activity y exceeds θ_M, synapses potentiate.
When below θ_M, they depress. The threshold itself moves with
activity — the core mechanism of **metaplasticity**.

Biological significance
-----------------------
BCM explains the emergence of selectivity in sensory cortex
(e.g. ocular dominance and orientation columns in V1) and provides
a stable, self-normalising alternative to raw Hebbian learning.

References
----------
Bienenstock, E.L., Cooper, L.N., & Munro, P.W. (1982). Theory for
the development of neuron selectivity. J. Neuroscience, 2(1), 32-48.

Cooper, L.N., & Bear, M.F. (2012). The BCM theory of synapse
modification at 30. Nature Reviews Neuroscience, 13(11), 798-810.

Author: Dimitri Romanov
Project: neuro-ai-bridge
"""

import numpy as np
from typing import Optional

from .hebbian import HebbianNetwork


class BCMNetwork(HebbianNetwork):
    """
    Hebbian network with BCM sliding-threshold plasticity.

    Parameters
    ----------
    n_input, n_output, learning_rate, decay
        See HebbianNetwork.
    theta_init : float
        Initial value of the sliding threshold θ_M.
    theta_rate : float
        Time constant (1/τ) of the threshold running average.
        Smaller = slower metaplasticity.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        learning_rate: float = 0.01,
        decay: float = 0.0,
        theta_init: float = 0.5,
        theta_rate: float = 0.01,
    ):
        super().__init__(n_input, n_output, learning_rate, decay)
        self.theta = float(theta_init)
        self.theta_rate = float(theta_rate)
        self.theta_history: list = [self.theta]

    def update(self, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        """Apply BCM update to all output units."""
        for j in range(post.shape[0]):
            phi = post[j] * (post[j] - self.theta)
            self.weights[:, j] += self.learning_rate * phi * pre

        # Sliding threshold tracks <y^2>
        self.theta += self.theta_rate * (np.mean(post ** 2) - self.theta)

        self.history.append(np.linalg.norm(self.weights))
        self.theta_history.append(self.theta)
        return self.weights

    def selectivity(self, patterns: np.ndarray) -> np.ndarray:
        """
        Per-output-unit selectivity.

        Selectivity = 1 - (mean response / max response).
        A fully selective neuron (responds to only one pattern) → 1.
        A non-selective neuron (uniform response) → 0.
        """
        responses = np.array([self.forward(p) for p in patterns])  # (P, O)
        mean_r = np.mean(np.abs(responses), axis=0)
        max_r = np.max(np.abs(responses), axis=0) + 1e-12
        return 1.0 - (mean_r / max_r)


if __name__ == "__main__":
    np.random.seed(0)

    # Create a small set of orthogonal-ish input "patterns"
    # — mimicking different oriented visual stimuli reaching V1.
    n_patterns, n_input, n_output = 6, 12, 4
    patterns = np.eye(n_patterns, n_input) + 0.05 * np.random.randn(
        n_patterns, n_input
    )

    net = BCMNetwork(
        n_input=n_input,
        n_output=n_output,
        learning_rate=0.02,
        theta_init=0.1,
        theta_rate=0.05,
    )
    net.train(patterns, n_epochs=300)

    print("=" * 60)
    print("Neuro-AI Bridge: BCM selectivity development")
    print("=" * 60)
    sel = net.selectivity(patterns)
    print(f"Final θ_M                : {net.theta:.4f}")
    print(f"Mean output selectivity  : {sel.mean():.3f}")
    print(f"Per-unit selectivity     : "
          f"{np.array2string(sel, precision=3)}")
    print("\nHigher selectivity ⇒ each output unit has learned to"
          "\nrespond preferentially to a subset of inputs, "
          "mirroring orientation/ocular-dominance tuning in V1.")
