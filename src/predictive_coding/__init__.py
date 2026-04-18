"""
Predictive Coding module
========================

Hierarchical generative models where each layer predicts the
activity of the layer below, and learning is driven by the
resulting prediction errors.

Contents
--------
- pc_network.py   : Hierarchical predictive coding network (Rao & Ballard)
- free_energy.py  : Variational free-energy autoencoder (Friston)
"""

from .pc_network import PredictiveCodingNetwork, PCLayer
from .free_energy import FreeEnergyVAE, free_energy_loss

__all__ = [
    "PredictiveCodingNetwork",
    "PCLayer",
    "FreeEnergyVAE",
    "free_energy_loss",
]
