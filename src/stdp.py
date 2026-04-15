"""
Spike-Timing-Dependent Plasticity (STDP)
=========================================
Implementation of STDP — a temporally asymmetric form of Hebbian learning
observed in biological neural circuits.

The key insight: the precise timing of pre- and post-synaptic spikes
determines whether a synapse is strengthened (LTP) or weakened (LTD).

    Pre before Post (Δt > 0) → LTP (causal: pre caused post to fire)
    Post before Pre (Δt < 0) → LTD (anti-causal: reverse the association)

This temporal sensitivity is critical for understanding how the brain
encodes sequences, learns temporal patterns, and forms predictions —
principles directly applicable to transformer attention mechanisms
and recurrent architectures.

Author: Dimitri Romanov
Project: neuro-ai-bridge

References
----------
Bi, G.Q. & Poo, M.M. (1998). Synaptic modifications in cultured
    hippocampal neurons. Journal of Neuroscience, 18(24), 10464-10472.
Markram, H., et al. (1997). Regulation of synaptic efficacy by
    coincidence of postsynaptic APs and EPSPs. Science, 275(5297), 213-215.
"""

import numpy as np
from typing import List, Tuple, Optional


class STDPSynapse:
    """
    A single synapse governed by STDP.
    
    Parameters
    ----------
    tau_plus : float
        Time constant for LTP window (ms)
    tau_minus : float
        Time constant for LTD window (ms)
    a_plus : float
        Maximum LTP amplitude
    a_minus : float
        Maximum LTD amplitude
    w_max : float
        Maximum synaptic weight (hard bound)
    w_min : float
        Minimum synaptic weight (hard bound)
    """
    
    def __init__(
        self,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        a_plus: float = 0.01,
        a_minus: float = 0.012,
        w_max: float = 1.0,
        w_min: float = 0.0,
        initial_weight: float = 0.5
    ):
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_max = w_max
        self.w_min = w_min
        self.weight = initial_weight
        self.weight_history = [initial_weight]
    
    def compute_delta_w(self, delta_t: float) -> float:
        """
        Compute weight change based on spike timing difference.
        
        Parameters
        ----------
        delta_t : float
            Time difference (t_post - t_pre) in ms.
            Positive = pre before post (LTP)
            Negative = post before pre (LTD)
        
        Returns
        -------
        float
            Weight change magnitude
        """
        if delta_t > 0:
            # Pre before post: LTP
            return self.a_plus * np.exp(-delta_t / self.tau_plus)
        elif delta_t < 0:
            # Post before pre: LTD
            return -self.a_minus * np.exp(delta_t / self.tau_minus)
        else:
            return 0.0
    
    def update(self, delta_t: float) -> float:
        """Apply STDP update and clip to bounds."""
        dw = self.compute_delta_w(delta_t)
        self.weight = np.clip(self.weight + dw, self.w_min, self.w_max)
        self.weight_history.append(self.weight)
        return self.weight


class STDPNetwork:
    """
    A network of leaky integrate-and-fire neurons connected by STDP synapses.
    
    This maps onto the biological observation that cortical circuits
    self-organise through spike-timing correlations, forming structured
    connectivity patterns from initially random wiring.
    
    The same principle appears in AI as:
    - Attention mechanisms (temporal correlation → relevance weighting)
    - Contrastive learning (co-occurrence → representation similarity)
    - Temporal difference learning in RL (prediction error → value update)
    """
    
    def __init__(
        self,
        n_neurons: int,
        connectivity: float = 0.3,
        stdp_params: Optional[dict] = None
    ):
        self.n_neurons = n_neurons
        self.synapses = {}
        params = stdp_params or {}
        
        # Create random sparse connectivity
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i != j and np.random.random() < connectivity:
                    self.synapses[(i, j)] = STDPSynapse(**params)
        
        # Neuron state
        self.membrane_potential = np.zeros(n_neurons)
        self.threshold = 1.0
        self.decay = 0.95
        self.spike_times = {i: [] for i in range(n_neurons)}
    
    def step(
        self,
        t: float,
        external_input: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Simulate one timestep of network dynamics.
        
        Returns list of neuron indices that spiked.
        """
        # Decay membrane potential (leaky integration)
        self.membrane_potential *= self.decay
        
        # Add external input
        if external_input is not None:
            self.membrane_potential += external_input
        
        # Add synaptic input from connected neurons
        for (pre, post), synapse in self.synapses.items():
            if self.spike_times[pre] and (t - self.spike_times[pre][-1]) < 5:
                self.membrane_potential[post] += synapse.weight * 0.1
        
        # Check for spikes
        spiked = []
        for i in range(self.n_neurons):
            if self.membrane_potential[i] >= self.threshold:
                spiked.append(i)
                self.spike_times[i].append(t)
                self.membrane_potential[i] = 0.0  # Reset after spike
        
        # Apply STDP updates for all spiking pairs
        for neuron_i in spiked:
            for (pre, post), synapse in self.synapses.items():
                if post == neuron_i and self.spike_times[pre]:
                    delta_t = t - self.spike_times[pre][-1]
                    synapse.update(delta_t)
                elif pre == neuron_i and self.spike_times[post]:
                    delta_t = self.spike_times[post][-1] - t
                    synapse.update(delta_t)
        
        return spiked
    
    def get_weight_matrix(self) -> np.ndarray:
        """Return the current connectivity weight matrix."""
        W = np.zeros((self.n_neurons, self.n_neurons))
        for (i, j), synapse in self.synapses.items():
            W[i, j] = synapse.weight
        return W


if __name__ == "__main__":
    print("=" * 60)
    print("Neuro-AI Bridge: STDP Network Simulation")
    print("=" * 60)
    
    np.random.seed(42)
    net = STDPNetwork(n_neurons=10, connectivity=0.3)
    
    initial_W = net.get_weight_matrix()
    
    # Simulate with structured input (correlated groups)
    total_spikes = 0
    for t in range(500):
        # Group A (neurons 0-4) receives correlated input
        ext = np.zeros(10)
        if t % 10 < 3:
            ext[:5] = np.random.uniform(0.3, 0.8, 5)
        # Group B (neurons 5-9) receives independent input
        ext[5:] = np.random.uniform(0, 0.3, 5)
        
        spiked = net.step(float(t), ext)
        total_spikes += len(spiked)
    
    final_W = net.get_weight_matrix()
    
    print(f"\nTotal spikes: {total_spikes}")
    print(f"Mean weight change: {np.mean(np.abs(final_W - initial_W)):.4f}")
    print(f"Within-group A strengthening: {np.mean(final_W[:5, :5]):.4f}")
    print(f"Between-group mean: {np.mean(final_W[:5, 5:]):.4f}")
    print("\nCorrelated neurons (Group A) should show stronger connections")
    print("than cross-group connections — mirroring Hebb's principle.")
