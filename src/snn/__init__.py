"""
Spiking Neural Network (SNN) module
===================================

Biologically-inspired networks where neurons communicate via
discrete spikes rather than continuous activations.

Contents
--------
- lif_neuron.py      : Leaky Integrate-and-Fire neuron model
- snn_classifier.py  : Rate-coded spiking classifier (MNIST-ready)
"""

from .lif_neuron import LIFNeuron, LIFLayer
from .snn_classifier import SpikingClassifier

__all__ = ["LIFNeuron", "LIFLayer", "SpikingClassifier"]
