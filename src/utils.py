"""
utils.py
--------
Utility functions for CSCN 8020 Assignment 1.

Provides:
    - setup_logging      : configure Python logging to file + console
    - ExperimentRunner   : run and benchmark multiple agents
    - Visualiser         : matplotlib gridworld and convergence plots
    - Logger             : structured experiment logger
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib import cm


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(
    log_dir:  str  = "logs",
    log_file: str  = "execution.log",
    level:    int  = logging.DEBUG,
) -> logging.Logger:
    """
    Configure Python logging to write to both console and a log file.

    Log file is stored in log_dir/log_file.
    Returns the root logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    root = logging.getLogger()
    root.info("=" * 70)
    root.info("CSCN 8020 Assignment 1 — Execution Log")
    root.info("Started at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    root.info("Log file  : %s", log_path)
    root.info("=" * 70)
    return root


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Runs and benchmarks multiple RL agents on the same environment.

    Responsibilities:
        - Execute agent.train() with timing
        - Collect results for comparison table
        - Log all parameters and outcomes
    """

    def __init__(self, env) -> None:
        self.env     = env
        self.results: Dict = {}
        self._logger = logging.getLogger(self.__class__.__name__)

    def run(self, agent, label: str, n_timing_runs: int = 500, **train_kwargs):
        """
        Train agent, benchmark timing, store results.

        Args:
            agent      : any agent with a .train() method
            label      : human-readable name for logging / tables
            n_timing_runs : number of timing repetitions
        """
        import timeit

        self._logger.info("--- Running: %s ---", label)
        self._logger.info("Parameters: gamma=%.3f", getattr(agent, "gamma", "N/A"))

        # Single training run for actual values
        V = agent.train(**train_kwargs)

        # Averaged timing
        avg_ms = timeit.timeit(
            lambda: agent.train(), number=n_timing_runs
        ) / n_timing_runs * 1000

        self.results[label] = {
            "agent":    agent,
            "V":        V.copy(),
            "iters":    getattr(agent, "iterations", "N/A"),
            "avg_ms":   avg_ms,
            "elapsed":  agent.elapsed_ms,
        }
        self._logger.info(
            "%s: iters=%s, avg_time=%.4fms",
            label,
            self.results[label]["iters"],
            avg_ms,
        )
        return V

    def print_comparison(self, labels: Optional[List[str]] = None) -> None:
        """Print a formatted comparison table of all registered results."""
        if labels is None:
            labels = list(self.results.keys())
        w = 38
        print("=" * (w + len(labels) * 20))
        header = "PERFORMANCE COMPARISON"
        print(f"{header:^{w + len(labels)*20}}")
        print("=" * (w + len(labels) * 20))
        print("%-38s" % "Metric", end="")
        for lbl in labels:
            print(f"  {lbl:>17}", end="")
        print()
        print("-" * (w + len(labels) * 20))
        rows = ["iters", "avg_ms"]
        names = {"iters": "Iterations to converge",
                 "avg_ms": "Avg time per run (ms)"}
        for key in rows:
            print(f"{'%-38s' % names[key]}", end="")
            for lbl in labels:
                val = self.results[lbl].get(key, "N/A")
                if isinstance(val, float):
                    print(f"  {val:>17.4f}", end="")
                else:
                    print(f"  {str(val):>17}", end="")
            print()
        print("=" * (w + len(labels) * 20))


# ---------------------------------------------------------------------------
# Visualiser
# ---------------------------------------------------------------------------

class Visualiser:
    """
    Matplotlib-based visualisation helper.

    Responsibilities:
        - Gridworld heatmap (V*) + policy overlay (pi*)
        - Convergence delta curves
        - MC MAE convergence over episodes
        - Side-by-side VI vs MC comparison
    """

    GOAL_COLOR = "#27ae60"
    GREY_COLOR = "#95a5a6"
    CMAP       = cm.RdYlGn

    @staticmethod
    def gridworld(
        V: np.ndarray,
        policy_grid: List[List[str]],
        env,
        title: str = "Optimal V* and pi*",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot value function heatmap and policy arrows side by side."""
        grid   = env.GRID_SIZE
        nongv  = [V[r, c] for r in range(grid) for c in range(grid)
                  if (r, c) != env.GOAL_STATE]
        norm   = Normalize(vmin=min(nongv), vmax=max(nongv))
        cmap   = Visualiser.CMAP

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(title, fontsize=13, fontweight="bold")

        for ax_idx, (ax, ttl) in enumerate(
            zip(axes, ["Value Function V*", "Optimal Policy pi*"])
        ):
            ax.set_title(ttl, fontsize=12, fontweight="bold", pad=10)
            ax.set_xlim(0, grid); ax.set_ylim(0, grid)
            ax.set_xticks(range(grid + 1)); ax.set_yticks(range(grid + 1))
            ax.set_xticklabels([]); ax.set_yticklabels([])
            ax.set_aspect("equal")

            for row in range(grid):
                for col in range(grid):
                    state = (row, col)
                    dr    = grid - 1 - row
                    if state == env.GOAL_STATE:
                        color = Visualiser.GOAL_COLOR
                    elif state in env.GREY_STATES:
                        color = Visualiser.GREY_COLOR
                    else:
                        color = cmap(norm(V[row, col]))

                    ax.add_patch(
                        plt.Rectangle((col, dr), 1, 1,
                                      color=color, ec="#333", lw=1.2)
                    )
                    if ax_idx == 0:
                        ax.text(col+0.5, dr+0.65, f"s{row},{col}",
                                ha="center", va="center",
                                fontsize=7, alpha=0.7)
                        ax.text(col+0.5, dr+0.32, f"{V[row,col]:.2f}",
                                ha="center", va="center",
                                fontsize=9, fontweight="bold",
                                color="white" if state == env.GOAL_STATE
                                else "black")
                    else:
                        ax.text(col+0.5, dr+0.5, policy_grid[row][col],
                                ha="center", va="center",
                                fontsize=17, fontweight="bold",
                                color="white" if state == env.GOAL_STATE
                                else "#1a1a6e")

        axes[1].legend(
            handles=[
                mpatches.Patch(color=Visualiser.GOAL_COLOR, label="Goal +10"),
                mpatches.Patch(color=Visualiser.GREY_COLOR, label="Grey -5"),
                mpatches.Patch(color="#f1c40f", label="Regular -1"),
            ],
            loc="upper left", fontsize=9,
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def convergence_delta(
        delta_histories: Dict[str, List[float]],
        theta: float,
        title: str = "Convergence: Delta per Iteration",
        save_path: Optional[str] = None,
    ) -> None:
        """Semilogy plot of max-delta per iteration for VI variants."""
        colors = ["#2980b9", "#e74c3c", "#27ae60"]
        fig, axes = plt.subplots(1, len(delta_histories), figsize=(7 * len(delta_histories), 4))
        if len(delta_histories) == 1:
            axes = [axes]
        fig.suptitle(title, fontsize=13, fontweight="bold")

        for ax, (label, deltas), color in zip(axes, delta_histories.items(), colors):
            ax.semilogy(range(1, len(deltas) + 1), deltas,
                        color=color, lw=2, marker="o", ms=4)
            ax.axhline(theta, color="gray", ls="--", lw=1.2,
                       label=f"theta={theta:.0e}")
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Iteration (sweep)")
            ax.set_ylabel("Max delta (log scale)")
            ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def mc_convergence(
        mae_history: List[Tuple[int, float]],
        n_episodes: int,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot MC global MAE over episodes (linear + log scale)."""
        eps, maes = zip(*mae_history)
        mae_safe  = [max(v, 1e-10) for v in maes]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle(
            f"MC Global Convergence over {n_episodes:,} Episodes",
            fontsize=13, fontweight="bold",
        )
        axes[0].plot(eps, maes, color="#e74c3c", lw=1.5, alpha=0.85,
                     label="MAE (all 25 states vs VI)")
        axes[0].axhline(maes[-1], color="#2980b9", ls="--", lw=1.5,
                        label=f"Final MAE: {maes[-1]:.4f}")
        axes[0].set_xlabel("Episode"); axes[0].set_ylabel("MAE")
        axes[0].set_title("Global MAE vs VI (Linear)")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(eps, mae_safe, color="#8e44ad", lw=1.5)
        axes[1].set_xlabel("Episode"); axes[1].set_ylabel("MAE (log)")
        axes[1].set_title("Global MAE vs VI (Log Scale)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    @staticmethod
    def side_by_side(
        V_vi: np.ndarray,
        V_mc: np.ndarray,
        policy_vi: List[List[str]],
        policy_mc: List[List[str]],
        env,
        n_episodes: int,
        save_path: Optional[str] = None,
    ) -> None:
        """Side-by-side VI vs MC value function and policy."""
        grid  = env.GRID_SIZE
        allv  = np.concatenate([V_vi.ravel(), V_mc.ravel()])
        norm  = Normalize(vmin=allv.min(), vmax=allv.max())
        cmap  = Visualiser.CMAP

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"VI vs Off-Policy MC ({n_episodes:,} episodes)",
            fontsize=13, fontweight="bold",
        )
        pairs = [
            (V_vi, policy_vi, "Standard Value Iteration V*"),
            (V_mc, policy_mc, f"Off-Policy MC V ({n_episodes:,} eps)"),
        ]
        for ax, (V, pol, ttl) in zip(axes, pairs):
            ax.set_title(ttl, fontsize=11, fontweight="bold", pad=8)
            ax.set_xlim(0, grid); ax.set_ylim(0, grid)
            ax.set_xticks(range(grid+1)); ax.set_yticks(range(grid+1))
            ax.set_xticklabels([]); ax.set_yticklabels([])
            ax.set_aspect("equal")
            for row in range(grid):
                for col in range(grid):
                    state = (row, col)
                    dr    = grid - 1 - row
                    if state == env.GOAL_STATE:
                        color = Visualiser.GOAL_COLOR
                    elif state in env.GREY_STATES:
                        color = Visualiser.GREY_COLOR
                    else:
                        color = cmap(norm(V[row, col]))
                    ax.add_patch(
                        plt.Rectangle((col, dr), 1, 1,
                                      color=color, ec="#333", lw=1.0)
                    )
                    ax.text(col+0.5, dr+0.65, f"{V[row,col]:.1f}",
                            ha="center", va="center", fontsize=8,
                            fontweight="bold",
                            color="white" if state == env.GOAL_STATE
                            else "black")
                    ax.text(col+0.5, dr+0.28, pol[row][col],
                            ha="center", va="center", fontsize=14,
                            color="white" if state == env.GOAL_STATE
                            else "#1a1a6e")
        axes[1].legend(
            handles=[
                mpatches.Patch(color=Visualiser.GOAL_COLOR, label="Goal +10"),
                mpatches.Patch(color=Visualiser.GREY_COLOR, label="Grey -5"),
            ],
            loc="upper left", fontsize=9,
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


# ---------------------------------------------------------------------------
# Logger (structured experiment results)
# ---------------------------------------------------------------------------

class Logger:
    """
    Structured logger that writes human-readable experiment results
    to both the Python logging system and an optional summary file.
    """

    def __init__(self, name: str = "ExperimentLogger") -> None:
        self._log = logging.getLogger(name)

    def section(self, title: str) -> None:
        sep = "=" * 60
        self._log.info(sep)
        self._log.info("  %s", title)
        self._log.info(sep)

    def params(self, label: str, **kwargs) -> None:
        self._log.info("[%s] Parameters:", label)
        for k, v in kwargs.items():
            self._log.info("  %-20s = %s", k, v)

    def result(self, label: str, **kwargs) -> None:
        self._log.info("[%s] Results:", label)
        for k, v in kwargs.items():
            if isinstance(v, float):
                self._log.info("  %-20s = %.6f", k, v)
            elif isinstance(v, np.ndarray):
                self._log.info("  %-20s =\n%s", k, np.round(v, 3))
            else:
                self._log.info("  %-20s = %s", k, v)

    def done(self) -> None:
        self._log.info("=" * 60)
        self._log.info("Execution completed at: %s",
                       datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self._log.info("=" * 60)
