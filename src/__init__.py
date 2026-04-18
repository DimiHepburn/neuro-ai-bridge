"""
neuro-ai-bridge
===============

Mapping neuroscientific principles onto deep learning architectures.

Top-level package: re-exports the available sub-packages so that
user code can do e.g.:

    from src.plasticity import HebbianNetwork, BCMNetwork, STDPNetwork
"""

from . import plasticity

__all__ = ["plasticity"]
