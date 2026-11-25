"""Power-law scaling slope & plateau estimation for 350M-scale language model pretraining."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class EvalPoint:
    """Single validation checkpoint measurement."""

    step: int
    tokens_seen: float
    val_loss: float
    is_cooldown: bool = False


@dataclass
class PlateauTracker:
    """Estimate scaling law `L(D) = L_inf + A * D^(-alpha)` and diminishing-returns plateau.

    Designed for 350M-scale models: compares architectures by scaling exponent `alpha`,
    projected asymptote `L_inf`, and empirical loss gain per 2x token doubling.
    """

    plateau_delta_per_doubling: float = 0.02  # Nats gained per 2x tokens below which model is plateauing
    min_points_for_fit: int = 4
    history: list[EvalPoint] = field(default_factory=list)

    def update(self, step: int, tokens_seen: float, val_loss: float, *, is_cooldown: bool = False) -> dict[str, float]:
        """Record a new validation point and return updated scaling/plateau metrics."""
        if tokens_seen > 0 and math.isfinite(val_loss):
            self.history.append(EvalPoint(step=step, tokens_seen=float(tokens_seen), val_loss=float(val_loss), is_cooldown=is_cooldown))
        return self.estimate()

    def _fit_power_law(self, points: list[EvalPoint]) -> tuple[float, float, float, float] | None:
        """Fit `L(D) = L_inf + A * D^(-alpha)` via 1D grid search over `L_inf` + closed-form OLS."""
        if len(points) < self.min_points_for_fit:
            return None

        d = np.array([p.tokens_seen for p in points], dtype=np.float64)
        loss = np.array([p.val_loss for p in points], dtype=np.float64)
        if np.ptp(np.log(d)) < 1e-6:
            return None

        log_d = np.log(d)
        min_loss = float(loss.min())
        # Search candidate irreducible/capacity floor L_inf strictly below min observed loss
        lo = max(0.0, min_loss - 3.0)
        hi = max(lo + 1e-4, min_loss - 0.02)
        candidates = np.linspace(lo, hi, 60)

        best_mse = float("inf")
        best_params: tuple[float, float, float] | None = None

        x_mean = float(log_d.mean())
        x_var = float(((log_d - x_mean) ** 2).sum())
        if x_var <= 1e-12:
            return None

        for l_inf in candidates:
            y = np.log(np.clip(loss - l_inf, 1e-6, None))
            y_mean = float(y.mean())
            slope = float(((log_d - x_mean) * (y - y_mean)).sum()) / x_var
            alpha = -slope
            if alpha <= 0.005 or alpha > 2.0:
                continue
            log_a = y_mean - slope * x_mean
            a = float(math.exp(min(log_a, 50.0)))
            pred = l_inf + a * np.power(d, -alpha)
            mse = float(np.mean((pred - loss) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_params = (float(l_inf), a, float(alpha))

        if best_params is None:
            return None
        l_inf, a, alpha = best_params
        return l_inf, a, alpha, math.sqrt(best_mse)

    def _empirical_gain_per_doubling(self, points: list[EvalPoint]) -> float | None:
        """Estimate loss improvement per 2x tokens from the most recent >=1.5x token span."""
        if len(points) < 2:
            return None
        latest = points[-1]
        for prev in reversed(points[:-1]):
            ratio = latest.tokens_seen / max(prev.tokens_seen, 1.0)
            if ratio >= 1.5:
                doublings = math.log2(ratio)
                return (prev.val_loss - latest.val_loss) / max(doublings, 1e-6)
        return None

    def estimate(self) -> dict[str, float]:
        """Compute scaling exponent `alpha`, projected asymptote `L_inf`, and plateau flag."""
        # Prefer cooldown probes when enough exist (immune to WSD high-LR noise)
        cooldown_pts = [p for p in self.history if p.is_cooldown]
        pts = cooldown_pts if len(cooldown_pts) >= self.min_points_for_fit else self.history
        if len(pts) < 2:
            return {}

        metrics: dict[str, float] = {}
        emp_gain = self._empirical_gain_per_doubling(pts)
        if emp_gain is not None:
            metrics["plateau/delta_per_2x_tokens"] = emp_gain

        fit = self._fit_power_law(pts)
        if fit is not None:
            l_inf, a, alpha, rmse = fit
            d_now = pts[-1].tokens_seen
            pred_now = l_inf + a * (d_now ** (-alpha))
            pred_2x = l_inf + a * ((2.0 * d_now) ** (-alpha))
            proj_gain_2x = pred_now - pred_2x
            metrics.update(
                {
                    "plateau/alpha": alpha,
                    "plateau/l_inf": l_inf,
                    "plateau/fit_rmse": rmse,
                    "plateau/proj_delta_next_2x": proj_gain_2x,
                    "plateau/is_plateauing": float(proj_gain_2x < self.plateau_delta_per_doubling),
                }
            )
        elif emp_gain is not None:
            metrics["plateau/is_plateauing"] = float(emp_gain < self.plateau_delta_per_doubling)

        return metrics
