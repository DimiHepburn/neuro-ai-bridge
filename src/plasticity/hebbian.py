"""
Hebbian Learning Rule Implementation
=====================================

A computational implementation of Hebb's postulate:
"Neurons that fire together, wire together."

This module maps the biological principle of synaptic plasticity
onto a simple neural network weight update rule, demonstrating how
neuroscientific principles can inform AI architecture design.

Author: Dimitri Romanov
Project: neuro-ai-bridge
"""

import numpy as np
from typing import Optional, Tuple


class HebbianNetwork:
    """
    A network that learns using Hebb's rule with optional decay.

    In biological neurons, long-term potentiation (LTP) strengthens
    synapses between co-active neurons. This implementation captures
    that principle as a weight update:

        Δw = η * pre * post

    Parameters
    ----------
    n_input : int
        Number of input neurons
    n_output : int
        Number of output neurons
    learning_rate : float
        Learning rate (η) controlling plasticity strength
    decay : float
        Weight decay factor to prevent unbounded growth (analogous
        to synaptic homeostasis / synaptic scaling in biological
        networks; Turrigiano, 2008)
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        learning_rate: float = 0.01,
        decay: float = 0.001,
    ):
        self.weights = np.random.randn(n_input, n_output) * 0.1
        self.learning_rate = learning_rate
        self.decay = decay
        self.history: list = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Propagate input through the network."""
        return np.tanh(x @ self.weights)

    def update(self, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        """
        Apply Hebbian weight update: Δw = η * (pre^T · post) - λ * w
        """
        delta_w = self.learning_rate * np.outer(pre, post)
        delta_w -= self.decay * self.weights
        self.weights += delta_w
        self.history.append(np.linalg.norm(self.weights))
        return self.weights

    def train(self, patterns: np.ndarray, n_epochs: int = 100) -> list:
        """Train on a batch of input patterns."""
        for _ in range(n_epochs):
            for pattern in patterns:
                output = self.forward(pattern)
                self.update(pattern, output)
        return self.history


class OjaNetwork(HebbianNetwork):
    """
    Oja's rule: a normalised variant of Hebbian learning.

        Δw = η * (post * pre - post² * w)

    Self-limiting — extracts the principal component of the input
    distribution without the weights exploding.

    References
    ----------
    Oja, E. (1982). Simplified neuron model as a principal component
    analyzer. Journal of Mathematical Biology, 15(3), 267-273.
    """

    def update(self, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        for j in range(post.shape[0]):
            delta_w = self.learning_rate * (
                post[j] * pre - post[j] ** 2 * self.weights[:, j]
            )
            self.weights[:, j] += delta_w
        self.history.append(np.linalg.norm(self.weights))
        return self.weights


if __name__ == "__main__":
    np.random.seed(42)
    patterns = np.random.randn(10, 8)

    print("=" * 60)
    print("Neuro-AI Bridge: Hebbian / Oja comparison")
    print("=" * 60)

    for name, Network in [
        ("Standard Hebbian", HebbianNetwork),
        ("Oja's Rule", OjaNetwork),
    ]:
        net = Network(n_input=8, n_output=4, learning_rate=0.01)
        history = net.train(patterns, n_epochs=50)
        print(f"\n{name}:")
        print(f"  Final weight norm: {history[-1]:.4f}")
        print(f"  Weight growth:     {history[-1] / history[0]:.2f}x")
        print(
            f"  Stable:            "
            f"{'Yes' if abs(history[-1] - history[-10]) < 0.1 else 'No'}"
        )
