"""
Plasticity module
=================

Implementations of biologically-grounded synaptic learning rules.

All rules are local (no global error signal), mirroring how real
neurons modify their synapses based only on pre- and post-synaptic
activity.

Contents
--------
- hebbian.py  : Classical Hebb rule, Oja's rule
- bcm.py      : Bienenstock-Cooper-Munro sliding-threshold rule
- stdp.py     : Spike-Timing-Dependent Plasticity
"""

from .hebbian import HebbianNetwork, OjaNetwork
from .bcm import BCMNetwork
from .stdp import STDPSynapse, STDPNetwork

__all__ = [
    "HebbianNetwork",
    "OjaNetwork",
    "BCMNetwork",
    "STDPSynapse",
    "STDPNetwork",
]
