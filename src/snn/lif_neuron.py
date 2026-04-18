"""
Leaky Integrate-and-Fire (LIF) Neuron
======================================

A minimal yet biologically faithful neuron model.  The membrane
potential V leaks back to its resting value while integrating input
current; when V crosses a threshold the neuron emits a spike and
resets.

    τ · dV/dt = -(V - V_rest) + R · I(t)
    if V >= V_thresh:  spike;  V ← V_reset

Implemented as a pure-PyTorch module so it can be composed with
standard `nn.Module` layers, optimisers and autograd.  The
discontinuous spike function uses a **surrogate gradient**
(fast-sigmoid) in the backward pass — a standard trick that
makes SNNs trainable with gradient descent.

References
----------
Gerstner, W. & Kistler, W. (2002). Spiking Neuron Models.
Neftci, E. et al. (2019). Surrogate gradient learning in spiking
    neural networks. IEEE Signal Processing Magazine, 36(6), 51-63.

Author: Dimitri Romanov
Project: neuro-ai-bridge
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _SurrogateSpike(torch.autograd.Function):
    """
    Heaviside spike in the forward pass, fast-sigmoid surrogate
    in the backward pass.

        forward :  s = 1 if v >= 0 else 0
        backward:  ds/dv ≈ 1 / (1 + α|v|)^2
    """

    alpha: float = 10.0

    @staticmethod
    def forward(ctx, v: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(v)
        return (v >= 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (v,) = ctx.saved_tensors
        alpha = _SurrogateSpike.alpha
        surrogate = 1.0 / (1.0 + alpha * v.abs()) ** 2
        return grad_output * surrogate


spike_fn = _SurrogateSpike.apply


class LIFNeuron(nn.Module):
    """
    A single LIF unit (or vectorised population) with learnable
    membrane decay.

    Parameters
    ----------
    beta : float
        Membrane potential decay per step (0 < beta < 1).
        beta ≈ exp(-dt / τ_mem)
    threshold : float
        Spike threshold V_thresh. V_reset is 0 by convention.
    learn_beta : bool
        If True, beta becomes a trainable parameter.
    """

    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        learn_beta: bool = False,
    ):
        super().__init__()
        if learn_beta:
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.register_buffer("beta", torch.tensor(float(beta)))
        self.threshold = float(threshold)

    def init_state(
        self, shape: torch.Size, device: torch.device | None = None
    ) -> torch.Tensor:
        """Return a zero-initialised membrane-potential tensor."""
        return torch.zeros(shape, device=device)

    def forward(
        self, input_current: torch.Tensor, mem: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One timestep of LIF dynamics.

        Parameters
        ----------
        input_current : torch.Tensor
            Synaptic input I(t), any shape.
        mem : torch.Tensor
            Membrane potential from previous step, same shape.

        Returns
        -------
        spikes : torch.Tensor  (0/1, same shape)
        mem    : torch.Tensor  (updated membrane potential)
        """
        # Leaky integration
        mem = self.beta * mem + input_current
        # Spike (with surrogate gradient) once threshold crossed
        spikes = spike_fn(mem - self.threshold)
        # Soft reset: subtract threshold on spike (keeps gradient flow)
        mem = mem - spikes * self.threshold
        return spikes, mem


class LIFLayer(nn.Module):
    """
    Fully-connected linear layer followed by an LIF population.

    Encapsulates the common pattern:

        I(t) = W · s_prev(t)  +  b
        (s, mem) = LIF(I, mem)

    Parameters
    ----------
    in_features, out_features : int
        Layer dimensions.
    beta : float
        Membrane decay (see LIFNeuron).
    threshold : float
        Spike threshold.
    learn_beta : bool
        Make beta trainable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        beta: float = 0.9,
        threshold: float = 1.0,
        learn_beta: bool = False,
    ):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.lif = LIFNeuron(beta=beta, threshold=threshold,
                             learn_beta=learn_beta)
        self.out_features = out_features

    def init_state(
        self, batch_size: int, device: torch.device | None = None
    ) -> torch.Tensor:
        return self.lif.init_state(
            torch.Size((batch_size, self.out_features)), device=device
        )

    def forward(
        self, spikes_in: torch.Tensor, mem: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = self.fc(spikes_in)
        return self.lif(current, mem)


if __name__ == "__main__":
    torch.manual_seed(0)

    # Drive a single LIF unit with a constant input and count spikes.
    lif = LIFNeuron(beta=0.9, threshold=1.0)
    mem = lif.init_state(torch.Size((1,)))

    n_steps = 100
    drive = torch.full((1,), 0.25)
    spikes = []
    for _ in range(n_steps):
        s, mem = lif(drive, mem)
        spikes.append(s.item())

    rate = sum(spikes) / n_steps
    print(f"LIF sanity check: {int(sum(spikes))} spikes in {n_steps} steps "
          f"(firing rate = {rate:.2%})")
