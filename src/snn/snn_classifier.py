"""
Spiking Classifier
==================

A 2-layer feed-forward Spiking Neural Network (SNN) for rate-coded
classification, using the LIF neurons from `lif_neuron.py`.

Design
------
- Input is presented for `n_steps` timesteps using **Poisson rate
  coding**: each pixel intensity is interpreted as a probability of
  spiking per step.
- Hidden and output layers are LIF populations.
- Prediction = class whose output unit spiked the most over the
  presentation window (rate decoding).
- Trained with a surrogate-gradient cross-entropy loss on the
  per-class spike counts.

This mirrors one of the simplest biologically plausible learning
setups and is compatible with MNIST / Fashion-MNIST / N-MNIST.

Author: Dimitri Romanov
Project: neuro-ai-bridge
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lif_neuron import LIFLayer


def poisson_encode(x: torch.Tensor, n_steps: int) -> torch.Tensor:
    """
    Poisson rate-encode a normalised input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input of shape (batch, features), values in [0, 1].
    n_steps : int
        Number of timesteps to generate.

    Returns
    -------
    torch.Tensor
        Binary spike train of shape (n_steps, batch, features).
    """
    x = x.clamp(0.0, 1.0)
    spikes = torch.rand((n_steps, *x.shape), device=x.device) < x
    return spikes.float()


class SpikingClassifier(nn.Module):
    """
    Two-layer feed-forward SNN classifier.

    Architecture
    ------------
        input  →  LIFLayer(in → hidden)  →  LIFLayer(hidden → n_classes)

    Parameters
    ----------
    input_size : int
        Flattened input dimension (e.g. 784 for MNIST).
    hidden_size : int
        Hidden-layer neuron count.
    n_classes : int
        Number of output classes.
    n_steps : int
        Presentation window (timesteps per sample).
    beta : float
        Membrane decay for all LIF populations.
    threshold : float
        Spike threshold for all LIF populations.
    learn_beta : bool
        Whether to learn beta (per-layer).
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 256,
        n_classes: int = 10,
        n_steps: int = 25,
        beta: float = 0.9,
        threshold: float = 1.0,
        learn_beta: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_steps = n_steps

        self.layer1 = LIFLayer(
            in_features=input_size,
            out_features=hidden_size,
            beta=beta,
            threshold=threshold,
            learn_beta=learn_beta,
        )
        self.layer2 = LIFLayer(
            in_features=hidden_size,
            out_features=n_classes,
            beta=beta,
            threshold=threshold,
            learn_beta=learn_beta,
        )

    def forward(
        self, x: torch.Tensor, encoded: bool = False
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Either:
              - raw input of shape (batch, input_size), values in [0, 1],
                which will be Poisson-encoded internally; or
              - a pre-encoded spike train of shape
                (n_steps, batch, input_size), if `encoded=True`.

        Returns
        -------
        torch.Tensor
            Per-class spike counts, shape (batch, n_classes). Use as
            logits for cross-entropy training or argmax for prediction.
        """
        if not encoded:
            spike_train = poisson_encode(x, self.n_steps)  # (T, B, F)
        else:
            spike_train = x

        batch_size = spike_train.shape[1]
        device = spike_train.device

        mem1 = self.layer1.init_state(batch_size, device=device)
        mem2 = self.layer2.init_state(batch_size, device=device)

        out_spike_sum = torch.zeros(
            (batch_size, self.n_classes), device=device
        )

        for t in range(spike_train.shape[0]):
            s1, mem1 = self.layer1(spike_train[t], mem1)
            s2, mem2 = self.layer2(s1, mem2)
            out_spike_sum = out_spike_sum + s2

        return out_spike_sum

    # ------------------------------------------------------------------
    # Convenience training helpers
    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices for a batch of inputs."""
        with torch.no_grad():
            counts = self.forward(x)
        return counts.argmax(dim=-1)

    def loss(
        self, x: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy loss over per-class spike counts."""
        counts = self.forward(x)
        # Treat spike counts as logits (works well in practice with
        # surrogate gradients; see Neftci et al. 2019).
        return F.cross_entropy(counts, targets)


if __name__ == "__main__":
    # Tiny smoke test with random data — no external dataset needed.
    torch.manual_seed(0)

    batch, feat, classes = 32, 784, 10
    model = SpikingClassifier(
        input_size=feat, hidden_size=128, n_classes=classes,
        n_steps=15,
    )

    x = torch.rand(batch, feat)
    y = torch.randint(0, classes, (batch,))

    counts = model(x)
    print(f"Output spike counts shape: {tuple(counts.shape)}  "
          f"(expected: ({batch}, {classes}))")

    loss = model.loss(x, y)
    loss.backward()
    print(f"Cross-entropy loss on random data: {loss.item():.4f}")
    print("Backward pass OK — surrogate gradients flowing through LIF units.")
