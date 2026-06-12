"""
environments.py
---------------
GridWorld environment implementations for CSCN 8020 Assignment 1.

Reward Convention (per assignment PDF):
    R(s) is the reward for being IN state s (current-state reward).
    V*(s) = max_a [R(s) + gamma * V*(s')]
    Terminal state: V*(s_goal) = R(s_goal) = +10 (fixed absorbing state).

Reference:
    Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning:
    An Introduction (2nd ed.). MIT Press. Chapter 3.
    Course playground: https://github.com/CSCN8020/playground/tree/main/lec3_DP
"""
from __future__ import annotations
import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)


class GridWorldEnvironment(gym.Env):
    """
    5x5 deterministic gridworld as a Gymnasium environment.

    Reward convention: R(s) = reward for being IN state s.
    Bellman update:
        V(s) = max_a [R(s) + gamma * V(s')]   for non-terminal s
        V(s_goal) = R(s_goal) = +10            fixed terminal value

    Observation space : Discrete(25)  flat (row*5 + col)
    Action space      : Discrete(4)   0=right,1=down,2=left,3=up

    Action space note: assignment lists a3=down (duplicate of a2).
    Corrected to a3=left per Sutton & Barto (2018) Example 4.1.
    Reference: https://github.com/CSCN8020/playground/tree/main/lec3_DP
    """
    metadata = {"render_modes": ["ansi"]}
    RIGHT, DOWN, LEFT, UP = 0, 1, 2, 3
    ACTION_DELTAS: Dict[int, Tuple[int,int]] = {
        0: (0,+1), 1:(+1,0), 2:(0,-1), 3:(-1,0)
    }
    ACTION_NAMES   = {0:"right",1:"down",2:"left",3:"up"}
    ACTION_SYMBOLS = {0:chr(8594),1:chr(8595),2:chr(8592),3:chr(8593)}
    GRID_SIZE   = 5
    GOAL_STATE  = (4, 4)
    GREY_STATES = frozenset({(2,2),(3,0),(0,4)})
    # REWARD_LIST[0]=terminal, [1]=grey, [2]=regular
    REWARD_LIST = [+10, -5, -1]

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Discrete(self.GRID_SIZE**2)
        self.action_space      = spaces.Discrete(len(self.ACTION_DELTAS))
        self._state: Tuple[int,int] = (0,0)
        self._build_transition_table()
        logger.info("GridWorldEnvironment init: %dx%d goal=%s grey=%s",
                    self.GRID_SIZE, self.GRID_SIZE,
                    self.GOAL_STATE, self.GREY_STATES)

    def state_to_obs(self, s: Tuple[int,int]) -> int:
        return s[0]*self.GRID_SIZE + s[1]

    def obs_to_state(self, o: int) -> Tuple[int,int]:
        return divmod(o, self.GRID_SIZE)

    def get_reward(self, state: Tuple[int,int]) -> float:
        """
        Reward for being IN state (current-state convention).
        R(s_goal) = +10, R(grey) = -5, R(other) = -1.
        Uses REWARD_LIST for O(1) lookup.
        """
        if state == self.GOAL_STATE:    return float(self.REWARD_LIST[0])
        if state in self.GREY_STATES:   return float(self.REWARD_LIST[1])
        return float(self.REWARD_LIST[2])

    def get_next_state(self, state: Tuple[int,int], action: int) -> Tuple[int,int]:
        """Deterministic transition. Wall collision: s' = s."""
        dr,dc = self.ACTION_DELTAS[action]
        nr,nc = state[0]+dr, state[1]+dc
        if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE:
            return (nr,nc)
        return state

    def is_terminal(self, state: Tuple[int,int]) -> bool:
        return state == self.GOAL_STATE

    def all_states(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                yield (r,c)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        while True:
            r = int(self.np_random.integers(0, self.GRID_SIZE))
            c = int(self.np_random.integers(0, self.GRID_SIZE))
            self._state = (r,c)
            if not self.is_terminal(self._state):
                break
        return self.state_to_obs(self._state), {}

    def step(self, action: int):
        ns       = self.get_next_state(self._state, action)
        reward   = self.get_reward(ns)          # reward on entering ns
        term     = self.is_terminal(ns)
        self._state = ns
        return self.state_to_obs(self._state), reward, term, False, {}

    def render(self):
        if self.render_mode != "ansi": return None
        lines = []
        for r in range(self.GRID_SIZE):
            row = ""
            for c in range(self.GRID_SIZE):
                s = (r,c)
                if s == self.GOAL_STATE:     row += " [G] "
                elif s in self.GREY_STATES:  row += " [X] "
                elif s == self._state:       row += " [A] "
                else:                        row += "  .  "
            lines.append(row)
        return "\n".join(lines)

    def _build_transition_table(self):
        self.T: Dict = {}
        for state in self.all_states():
            self.T[state] = {}
            for a in self.ACTION_DELTAS:
                ns   = self.get_next_state(state, a)
                r    = self.get_reward(ns)
                term = self.is_terminal(ns)
                self.T[state][a] = (ns, r, term)


class GridWorld2x2:
    """
    2x2 gridworld for Problem 2 manual verification.
    Layout:
        | s1(R=5) | s2(R=10) |
        | s3(R=1) | s4(R=2)  |
    Reward convention: R(s) = reward for being IN state s.
    """
    STATES  = ["s1","s2","s3","s4"]
    ACTIONS = ["up","down","left","right"]
    STATE_POS = {"s1":(0,0),"s2":(0,1),"s3":(1,0),"s4":(1,1)}
    POS_STATE = {v:k for k,v in STATE_POS.items()}
    REWARDS   = {"s1":5,"s2":10,"s3":1,"s4":2}
    ACTION_DELTAS = {"up":(-1,0),"down":(+1,0),"left":(0,-1),"right":(0,+1)}

    def get_next_state(self, state: str, action: str) -> str:
        r,c   = self.STATE_POS[state]
        dr,dc = self.ACTION_DELTAS[action]
        nr,nc = r+dr, c+dc
        if (nr,nc) in self.POS_STATE: return self.POS_STATE[(nr,nc)]
        return state

    def get_reward(self, state: str) -> float:
        return float(self.REWARDS[state])
