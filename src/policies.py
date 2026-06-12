"""
policies.py
-----------
Policy representations for CSCN 8020 Assignment 1.

Provides:
    - Policy          : base class
    - GreedyPolicy    : deterministic argmax policy over Q(s,a)
    - UniformPolicy   : uniform random behavior policy for off-policy MC
    - EpsilonGreedy   : epsilon-greedy soft policy

Reference:
    Sutton & Barto (2018), Chapter 3-5.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Policy:
    """Abstract base class for all policies."""

    def select_action(self, state: Tuple[int, int], **kwargs) -> int:
        raise NotImplementedError

    def action_probability(self, state: Tuple[int, int], action: int, **kwargs) -> float:
        raise NotImplementedError


class GreedyPolicy(Policy):
    """
    Deterministic greedy policy.

    pi(s) = argmax_a Q(s, a)

    Ties are broken by the lowest action index for reproducibility.
    Used as the TARGET POLICY in off-policy Monte Carlo.
    """

    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions
        # pi[r][c] = best action index
        self._pi: Optional[np.ndarray] = None

    def update(self, Q: np.ndarray) -> None:
        """Update policy to be greedy w.r.t. Q."""
        self._pi = np.argmax(Q, axis=2).astype(int)

    def select_action(self, state: Tuple[int, int], **kwargs) -> int:
        if self._pi is None:
            return 0
        return int(self._pi[state[0]][state[1]])

    def action_probability(
        self, state: Tuple[int, int], action: int, **kwargs
    ) -> float:
        """
        Returns 1.0 if action == greedy action, else 0.0.
        Used in the importance sampling ratio computation.
        """
        return 1.0 if action == self.select_action(state) else 0.0


class UniformPolicy(Policy):
    """
    Uniform random behavior policy.

    b(a|s) = 1 / |A|  for all a, s.

    Satisfies the COVERAGE ASSUMPTION required for off-policy MC:
        pi(a|s) > 0  =>  b(a|s) > 0
    since b(a|s) = 0.25 > 0 for all actions.

    Used as the BEHAVIOR POLICY in off-policy Monte Carlo.
    """

    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions
        self._prob     = 1.0 / n_actions

    def select_action(self, state: Tuple[int, int], **kwargs) -> int:
        return int(np.random.randint(self.n_actions))

    def action_probability(
        self, state: Tuple[int, int], action: int, **kwargs
    ) -> float:
        return self._prob


class EpsilonGreedyPolicy(Policy):
    """
    Epsilon-greedy soft policy.

    With probability (1 - epsilon) choose greedy action;
    with probability epsilon choose uniformly at random.

    Satisfies coverage assumption for any epsilon > 0.
    """

    def __init__(self, n_actions: int, epsilon: float = 0.1) -> None:
        self.n_actions = n_actions
        self.epsilon   = epsilon
        self._Q: Optional[np.ndarray] = None

    def update(self, Q: np.ndarray) -> None:
        self._Q = Q

    def select_action(self, state: Tuple[int, int], **kwargs) -> int:
        if self._Q is None or np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self._Q[state[0]][state[1]]))

    def action_probability(
        self, state: Tuple[int, int], action: int, **kwargs
    ) -> float:
        if self._Q is None:
            return 1.0 / self.n_actions
        greedy = int(np.argmax(self._Q[state[0]][state[1]]))
        if action == greedy:
            return 1.0 - self.epsilon + self.epsilon / self.n_actions
        return self.epsilon / self.n_actions
