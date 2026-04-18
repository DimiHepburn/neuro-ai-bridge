"""
Experience Replay
==================

A neocortex-side companion to the hippocampal `EpisodicBuffer`:
stores (state, action, reward, next_state, done) transitions or
generic (x, y) training pairs, and supplies interleaved mini-batches
for training.

Why interleaved replay matters
------------------------------
Deep networks trained sequentially on task A, then task B, forget
task A catastrophically — because gradient descent overwrites the
weights that represented A.  In biological brains, the hippocampus
*replays* recent episodes to the neocortex during rest / sleep,
letting the slow learner interleave old and new experiences and so
consolidate knowledge gradually.  The same trick is the foundation
of DQN (Mnih et al., 2015) and modern continual-learning methods.

References
----------
Lin, L.-J. (1992). Self-improving reactive agents based on
    reinforcement learning, planning and teaching. Machine Learning.
Mnih, V., et al. (2015). Human-level control through deep
    reinforcement learning. Nature, 518, 529-533.
Hassabis, D., Kumaran, D., Summerfield, C., & Botvinick, M. (2017).
    Neuroscience-inspired artificial intelligence. Neuron, 95(2).

Author: Dimitri Romanov
Project: neuro-ai-bridge
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class ReplayBuffer:
    """
    Fixed-capacity FIFO replay buffer with uniform random sampling.

    Each item can be any tuple — the buffer does not enforce a
    specific schema, but a common choice is
    ``(state, action, reward, next_state, done)`` for RL or
    ``(x, y)`` for supervised continual learning.

    Parameters
    ----------
    capacity : int
        Maximum number of items before the oldest is discarded.
    rng : np.random.Generator, optional
        Random generator for reproducibility.
    """

    def __init__(
        self,
        capacity: int,
        rng: Optional[np.random.Generator] = None,
    ):
        self.capacity = int(capacity)
        self._buf: deque = deque(maxlen=self.capacity)
        self._rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # Basic API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._buf)

    def push(self, item: Tuple[Any, ...]) -> None:
        """Add a single transition / training pair."""
        self._buf.append(item)

    def extend(self, items: Iterable[Tuple[Any, ...]]) -> None:
        """Add many items at once."""
        for it in items:
            self._buf.append(it)

    def sample(self, batch_size: int) -> List[Tuple[Any, ...]]:
        """Uniform-random sample of `batch_size` items (without replacement
        if possible, else with)."""
        n = len(self._buf)
        if n == 0:
            return []
        replace = n < batch_size
        idx = self._rng.choice(n, size=batch_size, replace=replace)
        return [self._buf[int(i)] for i in idx]

    def clear(self) -> None:
        self._buf.clear()


def interleaved_replay_batch(
    new_batch: Sequence[Tuple[Any, ...]],
    replay: "ReplayBuffer",
    replay_ratio: float = 0.5,
) -> List[Tuple[Any, ...]]:
    """
    Build a mini-batch that mixes fresh samples with replayed ones.

    Parameters
    ----------
    new_batch : sequence of items
        The current task's incoming batch.
    replay : ReplayBuffer
        Where to pull old experiences from.
    replay_ratio : float in [0, 1]
        Fraction of the returned batch that should come from replay.

    Returns
    -------
    list of items of length ``len(new_batch)`` containing a mixture
    of new and replayed samples, shuffled together.
    """
    if not 0.0 <= replay_ratio <= 1.0:
        raise ValueError("replay_ratio must lie in [0, 1]")

    total = len(new_batch)
    n_replay = min(int(round(total * replay_ratio)), len(replay))
    n_new = total - n_replay

    mixed: List[Tuple[Any, ...]] = list(new_batch[:n_new])
    if n_replay > 0:
        mixed.extend(replay.sample(n_replay))

    # Shuffle so new and replayed samples are interleaved temporally
    rng = replay._rng
    rng.shuffle(mixed)
    return mixed


if __name__ == "__main__":
    # Toy demo: two "tasks" in sequence, with vs without replay.
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(capacity=500, rng=rng)

    # Populate buffer with task-A pairs
    for _ in range(300):
        x = rng.standard_normal(4).astype(np.float32)
        y = int(x.sum() > 0)
        buf.push((x, y))

    # Simulate a task-B batch (different distribution)
    task_b = []
    for _ in range(32):
        x = rng.standard_normal(4).astype(np.float32) + 3.0
        y = int(x.prod() > 0)
        task_b.append((x, y))

    mix = interleaved_replay_batch(task_b, buf, replay_ratio=0.5)

    print("=" * 60)
    print("Neuro-AI Bridge: Interleaved replay demo")
    print("=" * 60)
    print(f"Replay buffer size          : {len(buf)}")
    print(f"New task-B batch size       : {len(task_b)}")
    print(f"Mixed batch size            : {len(mix)}")
    print(f"Fraction from replay        : ~50%")
    print("\nIn a real continual-learning loop, training on the mixed "
          "batch prevents catastrophic forgetting of task A while the "
          "network learns task B.")
