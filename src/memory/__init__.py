"""
Memory module
=============

Complementary Learning Systems: a fast, one-shot episodic buffer
(hippocampus-like) and a replay mechanism that interleaves stored
experiences with new learning (neocortex-like).

Contents
--------
- episodic_buffer.py : Fixed-capacity episodic memory with fast
                       write + similarity-based recall
- replay.py          : Experience replay buffer and interleaved-
                       training helpers for combating catastrophic
                       forgetting
"""

from .episodic_buffer import EpisodicBuffer
from .replay import ReplayBuffer, interleaved_replay_batch

__all__ = [
    "EpisodicBuffer",
    "ReplayBuffer",
    "interleaved_replay_batch",
]
