"""
Episodic Memory Buffer
======================

A hippocampus-inspired, fixed-capacity memory store that supports:

  * Fast, **one-shot** writes  (pattern separation / rapid binding)
  * Similarity-based recall    (pattern completion)
  * A reservoir-sampling write  policy (so older episodes aren't
    systematically biased out)

This mirrors the Complementary Learning Systems (CLS) hypothesis
(McClelland, McNaughton & O'Reilly, 1995): the hippocampus stores
episodes rapidly with minimal interference, which can later be
*replayed* to a slow, statistical learner (the neocortex / the deep
network) to consolidate knowledge without catastrophic forgetting.

References
----------
McClelland, J.L., McNaughton, B.L., & O'Reilly, R.C. (1995). Why
    there are complementary learning systems in the hippocampus and
    neocortex: insights from the successes and failures of
    connectionist models of learning and memory.
    Psychological Review, 102(3), 419-457.
Kumaran, D., Hassabis, D., & McClelland, J.L. (2016). What learning
    systems do intelligent agents need? Complementary learning systems
    theory updated. Trends in Cognitive Sciences, 20(7), 512-534.

Author: Dimitri Romanov
Project: neuro-ai-bridge
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class Episode:
    """One stored experience: a key vector, a payload, and a timestamp."""
    key: np.ndarray
    payload: Any
    t: int = 0
    tag: Optional[str] = None


class EpisodicBuffer:
    """
    Fixed-capacity episodic memory with similarity-based recall.

    Parameters
    ----------
    capacity : int
        Maximum number of episodes to store.
    key_dim : int
        Dimensionality of the key vector used for recall.
    rng : np.random.Generator, optional
        Random generator for reservoir sampling (for reproducibility).
    """

    def __init__(
        self,
        capacity: int,
        key_dim: int,
        rng: Optional[np.random.Generator] = None,
    ):
        self.capacity = int(capacity)
        self.key_dim = int(key_dim)
        self._store: List[Episode] = []
        self._seen: int = 0
        self._rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._store)

    def write(
        self,
        key: np.ndarray,
        payload: Any,
        tag: Optional[str] = None,
    ) -> None:
        """
        Store an episode.  Once the buffer is full, we use reservoir
        sampling so each seen episode has equal probability of being
        retained — an unbiased approximation to "keep everything".
        """
        key = np.asarray(key, dtype=np.float32).reshape(-1)
        if key.shape[0] != self.key_dim:
            raise ValueError(
                f"Expected key of dim {self.key_dim}, got {key.shape[0]}"
            )

        episode = Episode(
            key=key, payload=payload, t=self._seen, tag=tag
        )
        self._seen += 1

        if len(self._store) < self.capacity:
            self._store.append(episode)
        else:
            # Reservoir sampling: replace a random slot with prob k/n
            idx = int(self._rng.integers(0, self._seen))
            if idx < self.capacity:
                self._store[idx] = episode

    def recall(
        self,
        query: np.ndarray,
        k: int = 1,
        metric: str = "cosine",
    ) -> List[Tuple[float, Episode]]:
        """
        Return the `k` most similar stored episodes to `query`.

        Parameters
        ----------
        query : np.ndarray
            Query key, shape (key_dim,).
        k : int
            Number of episodes to return.
        metric : {"cosine", "euclidean"}
            Similarity metric.

        Returns
        -------
        list of (similarity, Episode), best first.
        """
        if len(self._store) == 0:
            return []

        query = np.asarray(query, dtype=np.float32).reshape(-1)
        keys = np.stack([e.key for e in self._store], axis=0)

        if metric == "cosine":
            q_norm = query / (np.linalg.norm(query) + 1e-12)
            k_norm = keys / (np.linalg.norm(keys, axis=1, keepdims=True)
                             + 1e-12)
            scores = k_norm @ q_norm            # higher = better
            order = np.argsort(-scores)
        elif metric == "euclidean":
            diffs = keys - query[None, :]
            scores = -np.linalg.norm(diffs, axis=1)   # negate → higher=better
            order = np.argsort(-scores)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        k = min(k, len(self._store))
        return [(float(scores[i]), self._store[i]) for i in order[:k]]

    def sample(self, n: int) -> List[Episode]:
        """Return `n` episodes sampled uniformly at random."""
        if len(self._store) == 0:
            return []
        n = min(n, len(self._store))
        idx = self._rng.choice(len(self._store), size=n, replace=False)
        return [self._store[i] for i in idx]

    def all(self) -> List[Episode]:
        """Return a shallow copy of every stored episode."""
        return list(self._store)

    def clear(self) -> None:
        self._store.clear()
        self._seen = 0


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    buf = EpisodicBuffer(capacity=50, key_dim=8, rng=rng)

    # Write 200 episodes; capacity is 50, reservoir sampling will
    # keep a representative subset.
    for i in range(200):
        key = rng.standard_normal(8)
        buf.write(key, payload={"id": i, "reward": float(rng.uniform())})

    print("=" * 60)
    print("Neuro-AI Bridge: Episodic buffer demo")
    print("=" * 60)
    print(f"Stored episodes : {len(buf)}  (capacity {buf.capacity})")
    print(f"Episodes seen    : {buf._seen}")

    # Recall nearest neighbours of a fresh query
    query = rng.standard_normal(8)
    hits = buf.recall(query, k=3, metric="cosine")
    print("\nTop-3 cosine recalls for a random query:")
    for sim, ep in hits:
        print(f"  sim={sim:+.3f}  t={ep.t:3d}  payload={ep.payload}")
