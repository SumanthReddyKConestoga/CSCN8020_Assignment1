"""
agents.py
---------
RL agent implementations for CSCN 8020 Assignment 1.

Reward Convention (standard S&B entry-reward):
    R_{t+1} = reward received when ENTERING next state s'.
    Bellman update:
        V(s) = max_a [R(s') + gamma * V(s')]  for non-terminal s
        V(s_goal) = 0  (terminal absorbing; future reward = 0)

    This is the standard formulation in Sutton & Barto (2018), Ch. 3-4.
    VI and MC both use this identical convention, ensuring consistency.

    NOTE on terminal state display:
        V(s_goal) = 0 in the value array.
        The terminal reward R(s_goal)=+10 is received on the TRANSITION
        into the goal, not as an extra future reward from the goal.
        One step before goal: V(adj) = R(goal) + gamma*V(goal) = 10+0.9*0=10.

Reference:
    Sutton & Barto (2018), p.83 (Value Iteration pseudocode).
    Sutton & Barto (2018), p.111 (Off-policy MC pseudocode).
    Course playground: https://github.com/CSCN8020/playground/tree/main/lec3_DP
    Course playground: https://github.com/CSCN8020/playground/tree/main/lec4_MC
"""
from __future__ import annotations
import logging, time, timeit
from typing import Dict, List, Optional, Tuple
import numpy as np
from .environments import GridWorldEnvironment
from .policies import GreedyPolicy, UniformPolicy

logger = logging.getLogger(__name__)


class ValueIterationAgent:
    """
    Standard Value Iteration — TWO-ARRAY synchronous method.

    Bellman optimality update (Sutton & Barto, 2018, Eq. 4.10):
        V_{k+1}(s) = max_a [R(s') + gamma * V_k(s')]   non-terminal
        V(s_goal)  = 0                                  terminal absorbing

    Reward convention: R(s') = reward for ENTERING next state s'.
    Terminal state has V=0 (absorbing; reward received on transition into it,
    not as an additional future reward from it).

    All states updated from previous sweep values only (synchronous).
    Convergence: max_s |V_{k+1}(s) - V_k(s)| < theta.
    Guaranteed by Banach contraction theorem (gamma=0.9 < 1).
    """
    def __init__(self, env: GridWorldEnvironment,
                 gamma: float = 0.9, theta: float = 1e-6):
        self.env, self.gamma, self.theta = env, gamma, theta
        g = env.GRID_SIZE
        self.V            = np.zeros((g,g))
        self.policy       = GreedyPolicy(env.action_space.n)
        self.iterations   = 0
        self.delta_history: List[float] = []
        self.elapsed_ms   = 0.0
        logger.info("ValueIterationAgent: gamma=%.3f theta=%.2e", gamma, theta)

    def _bellman_update(self, V: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        One full synchronous sweep.
        V_new(s) = max_a [R(s') + gamma * V_k(s')]
        Terminal: V_new(s_goal) = 0 (absorbing fixed point).
        """
        g     = self.env.GRID_SIZE
        V_new = np.zeros((g, g))
        delta = 0.0
        for state in self.env.all_states():
            r, c = state
            if self.env.is_terminal(state):
                V_new[r][c] = 0.0   # terminal absorbing: V=0
                continue
            q_vals = []
            for a in self.env.ACTION_DELTAS:
                ns, _, _ = self.env.T[state][a]
                # R(s') = reward for entering next state
                q = self.env.get_reward(ns) + self.gamma * V[ns[0]][ns[1]]
                q_vals.append(q)
            V_new[r][c] = max(q_vals)
            delta = max(delta, abs(V_new[r][c] - V[r][c]))
        return V_new, delta

    def train(self) -> np.ndarray:
        logger.info("ValueIterationAgent.train() start")
        t0 = time.perf_counter()
        self.V          = np.zeros((self.env.GRID_SIZE, self.env.GRID_SIZE))
        self.iterations = 0
        self.delta_history.clear()
        while True:
            V_new, delta = self._bellman_update(self.V)
            self.V = V_new
            self.iterations += 1
            self.delta_history.append(delta)
            logger.debug("  VI iter %d delta=%.6e", self.iterations, delta)
            if delta < self.theta: break
        self.elapsed_ms = (time.perf_counter()-t0)*1000
        self.policy.update(self._compute_Q())
        logger.info("ValueIterationAgent.train() done: iters=%d time=%.4fms",
                    self.iterations, self.elapsed_ms)
        return self.V

    def benchmark(self, n_runs: int = 500) -> float:
        total = timeit.timeit(lambda: self.train(), number=n_runs)
        avg   = (total/n_runs)*1000
        logger.info("Benchmark: avg %.4fms over %d runs", avg, n_runs)
        return avg

    def _compute_Q(self) -> np.ndarray:
        """
        Q(s,a) = R(s') + gamma * V(s')
        Entry-reward convention: R is for next state entered.
        """
        g, na = self.env.GRID_SIZE, self.env.action_space.n
        Q     = np.full((g, g, na), -np.inf)
        for state in self.env.all_states():
            r, c = state
            for action in self.env.ACTION_DELTAS:
                ns, _, _ = self.env.T[state][action]
                Q[r][c][action] = (self.env.get_reward(ns)
                                   + self.gamma * self.V[ns[0]][ns[1]])
        return Q

    def get_policy_grid(self) -> List[List[str]]:
        Q   = self._compute_Q()
        sym = self.env.ACTION_SYMBOLS
        out = []
        for r in range(self.env.GRID_SIZE):
            row = []
            for c in range(self.env.GRID_SIZE):
                state = (r, c)
                if self.env.is_terminal(state): row.append("G")
                else: row.append(sym[int(np.argmax(Q[r][c]))])
            out.append(row)
        return out


class InPlaceValueIterationAgent(ValueIterationAgent):
    """
    In-Place Value Iteration — SINGLE ARRAY.

    Identical Bellman update to standard VI but uses one array.
    Updated values immediately reused within the same sweep (async DP).
    Sutton & Barto (2018), Section 4.5.

    Wall-clock speedup: eliminates V_new allocation/copy each sweep.
    Convergence: same Banach contraction argument as standard VI.
    """
    def train(self) -> np.ndarray:
        logger.info("InPlaceValueIterationAgent.train() start")
        t0 = time.perf_counter()
        self.V          = np.zeros((self.env.GRID_SIZE, self.env.GRID_SIZE))
        self.iterations = 0
        self.delta_history.clear()
        while True:
            delta = 0.0
            for state in self.env.all_states():
                r, c = state
                if self.env.is_terminal(state):
                    self.V[r][c] = 0.0  # terminal absorbing
                    continue
                old = self.V[r][c]
                q_vals = []
                for a in self.env.ACTION_DELTAS:
                    ns, _, _ = self.env.T[state][a]
                    q_vals.append(self.env.get_reward(ns)
                                  + self.gamma * self.V[ns[0]][ns[1]])
                self.V[r][c] = max(q_vals)
                delta = max(delta, abs(self.V[r][c] - old))
            self.iterations += 1
            self.delta_history.append(delta)
            logger.debug("  InPlace iter %d delta=%.6e", self.iterations, delta)
            if delta < self.theta: break
        self.elapsed_ms = (time.perf_counter()-t0)*1000
        self.policy.update(self._compute_Q())
        logger.info("InPlaceValueIterationAgent.train() done: iters=%d %.4fms",
                    self.iterations, self.elapsed_ms)
        return self.V


class MonteCarloAgent:
    """
    Off-Policy Monte Carlo Control — Incremental Weighted IS.

    Implements EXACTLY Sutton & Barto (2018), p.111 pseudocode.
    Reference: https://github.com/CSCN8020/playground/tree/main/lec4_MC

    Reward convention (CONSISTENT with VI above):
        R_{t+1} = reward for ENTERING next state = R(s_{t+1}).
        Episode: (S0,A0,R1), (S1,A1,R2), ..., (S_{T-1},A_{T-1},R_T)
        Return: G_t = R_{t+1} + gamma*R_{t+2} + ... + gamma^{T-1-t}*R_T
        This matches VI: V(s) = max_a[R(s') + gamma*V(s')].

    Terminal state: V(goal) = 0 consistent with VI.
    The +10 reward appears as R_T when transitioning INTO goal,
    making one-step-before-goal V ≈ 10 in both VI and MC.

    Pseudocode (Sutton & Barto, 2018, p.111):
    Initialize: Q(s,a)=0, C(s,a)=0, pi(s)=argmax_a Q(s,a)
    Loop (each episode):
        b <- uniform random  [b(a|s)=0.25, satisfies coverage]
        Generate: S0,A0,R1,...,S_{T-1},A_{T-1},R_T  using b
        G=0, W=1
        Loop t=T-1,...,0 while W!=0:
            G  <- gamma*G + R_{t+1}
            C(St,At) <- C(St,At) + W
            Q(St,At) <- Q(St,At) + (W/C(St,At))[G - Q(St,At)]
            pi(St) <- argmax_a Q(St,a)  [ties: lowest index]
            If At != pi(St): break       [pi(At|St)=0 => IS ratio=0]
            W <- W * 1/b(At|St)          [= W*4 for uniform b]

    Weighted IS (preferred over Ordinary IS):
        Ordinary IS: unbiased but UNBOUNDED variance.
        Weighted IS: slightly biased but BOUNDED variance.
        (Precup, Sutton & Singh, 2000; S&B 2018 Sec. 5.6)
    """
    def __init__(self, env: GridWorldEnvironment,
                 gamma: float = 0.9, n_episodes: int = 100000,
                 max_steps: int = 1000, random_seed: int = 42):
        self.env, self.gamma     = env, gamma
        self.n_episodes          = n_episodes
        self.max_steps           = max_steps
        self.random_seed         = random_seed
        g, na = env.GRID_SIZE, env.action_space.n
        self.Q = np.zeros((g, g, na))
        self.C = np.zeros((g, g, na))
        self.V = np.zeros((g, g))
        self.target_policy   = GreedyPolicy(na)
        self.behavior_policy = UniformPolicy(na)
        self.elapsed_ms      = 0.0
        self.mae_history: List[Tuple[int, float]] = []
        logger.info("MonteCarloAgent: gamma=%.3f eps=%d max_steps=%d",
                    gamma, n_episodes, max_steps)

    def _generate_episode(self) -> List[Tuple]:
        """
        Generate episode using behavior policy b.
        Format: [(S_t, A_t, R_{t+1}), ...]
        R_{t+1} = R(S_{t+1}) = reward for ENTERING next state.
        Start: uniformly random non-terminal state.
        """
        episode = []
        obs, _  = self.env.reset(seed=None)
        state   = self.env.obs_to_state(obs)
        for _ in range(self.max_steps):
            if self.env.is_terminal(state): break
            action     = self.behavior_policy.select_action(state)
            next_state = self.env.get_next_state(state, action)
            reward     = self.env.get_reward(next_state)  # R(s') entry reward
            episode.append((state, action, reward))
            state = next_state
        return episode

    def train(self, V_vi: Optional[np.ndarray] = None) -> np.ndarray:
        logger.info("MonteCarloAgent.train() start: %d episodes", self.n_episodes)
        np.random.seed(self.random_seed)
        self.Q = np.zeros_like(self.Q)
        self.C = np.zeros_like(self.C)
        self.mae_history.clear()
        log_every = max(1, self.n_episodes // 500)
        t0        = time.perf_counter()

        for ep in range(self.n_episodes):
            episode = self._generate_episode()
            if not episode: continue
            G = 0.0; W = 1.0
            for t in reversed(range(len(episode))):
                St, At, Rt1 = episode[t]
                r, c = St
                G  = self.gamma * G + Rt1                              # step 1
                self.C[r][c][At] += W                                  # step 2
                self.Q[r][c][At] += (                                  # step 3
                    (W / self.C[r][c][At]) * (G - self.Q[r][c][At])
                )
                self.target_policy._pi = (                             # step 4
                    np.argmax(self.Q, axis=2).astype(int)
                )
                if At != int(self.target_policy._pi[r][c]): break      # step 5
                W *= 1.0 / self.behavior_policy.action_probability(St, At)  # step 6

            if V_vi is not None and (ep % log_every == 0):
                V_snap = np.max(self.Q, axis=2)
                mae    = float(np.mean(np.abs(V_snap - V_vi)))
                self.mae_history.append((ep, mae))
                logger.debug("  MC ep=%d MAE=%.4f", ep, mae)

        self.elapsed_ms = (time.perf_counter()-t0)*1000
        self.V = np.max(self.Q, axis=2)
        # Terminal state: V=0 (absorbing, consistent with VI convention)
        gr, gc = self.env.GOAL_STATE
        self.V[gr][gc] = 0.0
        logger.info("MonteCarloAgent.train() done: %.1fms V(goal)=%.1f",
                    self.elapsed_ms, self.V[gr][gc])
        return self.V

    def get_policy_grid(self) -> List[List[str]]:
        sym = self.env.ACTION_SYMBOLS
        out = []
        for r in range(self.env.GRID_SIZE):
            row = []
            for c in range(self.env.GRID_SIZE):
                state = (r, c)
                if self.env.is_terminal(state): row.append("G")
                else: row.append(sym[int(np.argmax(self.Q[r][c]))])
            out.append(row)
        return out

    def episode_length_stats(self, n_sample: int = 1000) -> Dict:
        lengths = [len(self._generate_episode()) for _ in range(n_sample)]
        arr = np.array(lengths)
        stats = {
            "mean":            float(arr.mean()),
            "median":          float(np.median(arr)),
            "max":             int(arr.max()),
            "reached_goal_pct": float(100 * np.mean(arr < self.max_steps)),
        }
        logger.info("Episode stats (n=%d): %s", n_sample, stats)
        return stats
