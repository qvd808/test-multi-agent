"""Backtesting metrics — Sharpe ratio, drawdown, win rate, total return.

These metrics answer the empirical question: "does the strategy actually
perform well?" This is complementary to Rocq formal verification which
answers "does the code do what it claims?"
"""

from __future__ import annotations

import numpy as np


def total_return(portfolio_values: list[float]) -> float:
    """Calculate total return as a percentage.

    Args:
        portfolio_values: Time series of portfolio values.

    Returns:
        Total return percentage (e.g. 0.15 = 15% gain).
    """
    if len(portfolio_values) < 2 or portfolio_values[0] == 0:
        return 0.0
    return (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]


def sharpe_ratio(
    portfolio_values: list[float],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualised Sharpe ratio.

    Args:
        portfolio_values: Time series of portfolio values.
        risk_free_rate: Annual risk-free rate (default 2%).
        periods_per_year: Trading days per year (default 252).

    Returns:
        Annualised Sharpe ratio. Returns 0.0 if insufficient data.
    """
    if len(portfolio_values) < 2:
        return 0.0

    values = np.array(portfolio_values, dtype=np.float64)
    returns = np.diff(values) / values[:-1]

    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    excess_return = np.mean(returns) - risk_free_rate / periods_per_year
    result = float(excess_return / np.std(returns) * np.sqrt(periods_per_year))

    return result if np.isfinite(result) else 0.0


def max_drawdown(portfolio_values: list[float]) -> float:
    """Calculate maximum drawdown (worst peak-to-trough decline).

    Args:
        portfolio_values: Time series of portfolio values.

    Returns:
        Max drawdown as a positive percentage (e.g. 0.20 = 20% decline).
        Returns 0.0 if no drawdown occurred.
    """
    if len(portfolio_values) < 2:
        return 0.0

    values = np.array(portfolio_values, dtype=np.float64)
    peak = np.maximum.accumulate(values)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = (peak - values) / peak

    drawdowns = np.nan_to_num(drawdowns, nan=0.0)
    return float(np.max(drawdowns))


def win_rate(trades: list[dict]) -> float:
    """Calculate win rate from a list of trades.

    Args:
        trades: List of trade dicts. Each must have "action" and "price" keys.
              Buys and sells are matched sequentially.

    Returns:
        Win rate as percentage (e.g. 0.60 = 60% profitable trades).
    """
    if not trades:
        return 0.0

    # Match buy→sell pairs
    buys = [t for t in trades if t.get("action") == "buy"]
    sells = [t for t in trades if t.get("action") == "sell"]

    pairs = min(len(buys), len(sells))
    if pairs == 0:
        return 0.0

    wins = 0
    for i in range(pairs):
        if sells[i]["price"] > buys[i]["price"]:
            wins += 1

    return wins / pairs


def compute_all_metrics(
    portfolio_values: list[float],
    trades: list[dict],
    cumulative_reward: float = 0.0,
) -> dict:
    """Compute all backtesting metrics at once.

    Args:
        portfolio_values: Time series of portfolio values.
        trades: List of trade dicts.
        cumulative_reward: Total RL reward from training.

    Returns:
        Dict with all metric values.
    """
    return {
        "total_return": total_return(portfolio_values),
        "sharpe_ratio": sharpe_ratio(portfolio_values),
        "max_drawdown": max_drawdown(portfolio_values),
        "win_rate": win_rate(trades),
        "cumulative_reward": cumulative_reward,
    }
