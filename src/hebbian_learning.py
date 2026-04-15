"""
Hebbian Learning Rule Implementation
=====================================
A computational implementation of Hebb's postulate: "Neurons that fire together, wire together."

This module maps the biological principle of synaptic plasticity onto a simple
neural network weight update rule, demonstrating how neuroscientific principles
can inform AI architecture design.

Author: Dimitri Romanov
Project: neuro-ai-bridge
"""

import numpy as np
from typing import Optional, Tuple


class HebbianNetwork:
    """
    A network that learns using Hebb's rule with optional decay.
    
    In biological neurons, long-term potentiation (LTP) strengthens synapses
    between co-active neurons. This implementation captures that principle
    as a weight update: Δw = η * pre * post
    
    Parameters
    ----------
    n_input : int
        Number of input neurons
    n_output : int
        Number of output neurons
    learning_rate : float
        Learning rate (η) controlling plasticity strength
    decay : float
        Weight decay factor to prevent unbounded growth (analogous to
        synaptic homeostasis / synaptic scaling in biological networks)
    """
    
    def __init__(
        self,
        n_input: int,
        n_output: int,
        learning_rate: float = 0.01,
        decay: float = 0.001
    ):
        self.weights = np.random.randn(n_input, n_output) * 0.1
        self.learning_rate = learning_rate
        self.decay = decay
        self.history = []
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Propagate input through the network."""
        return np.tanh(x @ self.weights)
    
    def update(self, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        """
        Apply Hebbian weight update.
        
        Implements: Δw = η * (pre^T · post) - λ * w
        
        The decay term (λ * w) acts as a homeostatic mechanism,
        analogous to synaptic scaling observed in cortical networks
        (Turrigiano, 2008).
        
        Parameters
        ----------
        pre : np.ndarray
            Pre-synaptic activation (input layer)
        post : np.ndarray
            Post-synaptic activation (output layer)
            
        Returns
        -------
        np.ndarray
            Updated weight matrix
        """
        # Hebbian term: correlate pre and post-synaptic activity
        delta_w = self.learning_rate * np.outer(pre, post)
        
        # Homeostatic decay: prevents runaway excitation
        delta_w -= self.decay * self.weights
        
        self.weights += delta_w
        self.history.append(np.linalg.norm(self.weights))
        
        return self.weights
    
    def train(
        self,
        patterns: np.ndarray,
        n_epochs: int = 100
    ) -> list:
        """
        Train on a set of input patterns using Hebbian learning.
        
        Parameters
        ----------
        patterns : np.ndarray
            Input patterns of shape (n_patterns, n_input)
        n_epochs : int
            Number of training epochs
            
        Returns
        -------
        list
            Weight norm history across training
        """
        for epoch in range(n_epochs):
            for pattern in patterns:
                output = self.forward(pattern)
                self.update(pattern, output)
        
        return self.history


class OjaNetwork(HebbianNetwork):
    """
    Oja's rule: a normalised variant of Hebbian learning.
    
    Oja's rule adds a self-limiting term that constrains weight growth,
    extracting the principal component of the input distribution. This
    mirrors how biological networks maintain stability through homeostatic
    plasticity while still encoding statistical structure.
    
    Δw = η * (post * pre - post² * w)
    
    References
    ----------
    Oja, E. (1982). Simplified neuron model as a principal component
    analyzer. Journal of Mathematical Biology, 15(3), 267-273.
    """
    
    def update(self, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        """Apply Oja's normalised Hebbian update."""
        for j in range(post.shape[0]):
            delta_w = self.learning_rate * (
                post[j] * pre - post[j]**2 * self.weights[:, j]
            )
            self.weights[:, j] += delta_w
        
        self.history.append(np.linalg.norm(self.weights))
        return self.weights


class BCMNetwork(HebbianNetwork):
    """
    Bienenstock-Cooper-Munro (BCM) learning rule.
    
    BCM theory proposes a sliding modification threshold (θ) that
    determines whether synapses are potentiated or depressed. This
    provides a biologically plausible mechanism for:
    - Experience-dependent plasticity
    - Selectivity development (e.g., orientation selectivity in V1)
    - Metaplasticity (the plasticity of plasticity)
    
    When post-synaptic activity exceeds θ → LTP (potentiation)
    When post-synaptic activity is below θ → LTD (depression)
    
    References
    ----------
    Bienenstock, E.L., Cooper, L.N., & Munro, P.W. (1982). Theory for
    the development of neuron selectivity. Journal of Neuroscience, 2(1), 32-48.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta = 0.5  # Sliding threshold
        self.theta_rate = 0.01
    
    def update(self, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        """Apply BCM learning rule with sliding threshold."""
        for j in range(post.shape[0]):
            # BCM modification function: post * (post - theta)
            phi = post[j] * (post[j] - self.theta)
            delta_w = self.learning_rate * phi * pre
            self.weights[:, j] += delta_w
        
        # Update sliding threshold (tracks average post-synaptic activity)
        self.theta += self.theta_rate * (np.mean(post**2) - self.theta)
        self.history.append(np.linalg.norm(self.weights))
        
        return self.weights


if __name__ == "__main__":
    # Demo: Compare learning rules on random patterns
    np.random.seed(42)
    patterns = np.random.randn(10, 8)
    
    print("=" * 60)
    print("Neuro-AI Bridge: Hebbian Learning Rule Comparison")
    print("=" * 60)
    
    for name, Network in [
        ("Standard Hebbian", HebbianNetwork),
        ("Oja's Rule", OjaNetwork),
        ("BCM Rule", BCMNetwork)
    ]:
        net = Network(n_input=8, n_output=4, learning_rate=0.01)
        history = net.train(patterns, n_epochs=50)
        print(f"\n{name}:")
        print(f"  Final weight norm: {history[-1]:.4f}")
        print(f"  Weight growth: {history[-1] / history[0]:.2f}x")
        print(f"  Stable: {'Yes' if abs(history[-1] - history[-10]) < 0.1 else 'No'}")
