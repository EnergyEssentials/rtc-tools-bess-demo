"""Compute clearing/acceptance probabilities for multi-band bid optimisation.

Supports two modes:
- Ensemble mode: empirical CDF from price scenario timeseries
- Normal fallback: Normal(mean=forecast, std=forecast * confidence_pct / 200)

Probabilities are pre-computed constants consumed by the solver's objective
function — they do not participate as decision variables.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


_STD_FLOOR_EUR_MWH = 1.0


def compute_clearing_probabilities(
    band_prices: list[float],
    point_forecast: list[float],
    ensemble_members: list[list[float]] | None = None,
    confidence_pct: float = 10.0,
) -> list[list[float]]:
    """Return P(clearing_price >= band_price[k]) for each PTU and band.

    Used for day-ahead marginal-clearing-price markets:
    - Selling: revenue accrues when clearing price >= offer price
    - Buying: fill occurs when clearing price <= bid price

    Args:
        band_prices: Ascending price levels [n_bands].
        point_forecast: Point price forecast per PTU [n_ptu].
        ensemble_members: Optional list of M scenario series, each [n_ptu].
        confidence_pct: Width of the confidence interval as percentage
            (default 10.0 means +/-10% at 2 sigma).

    Returns:
        Nested list [n_ptu][n_bands] with values in [0, 1].
    """
    n_ptu = len(point_forecast)
    n_bands = len(band_prices)
    band_arr = np.asarray(band_prices, dtype=float)

    if ensemble_members and len(ensemble_members) > 0:
        return _empirical_cdf_ge(band_arr, point_forecast, ensemble_members)

    return _normal_cdf_ge(band_arr, point_forecast, confidence_pct, n_ptu, n_bands)


def compute_acceptance_probabilities(
    offer_prices: list[float],
    clearing_price_forecast: list[float],
    ensemble_members: list[list[float]] | None = None,
    confidence_pct: float = 10.0,
) -> list[list[float]]:
    """Return P(market clearing price >= offer_price[k]) for each PTU and band.

    Used for pay-as-bid markets (FCR, aFRR capacity).
    A bid at price X is accepted iff the market clears at or above X.

    Same interface and fallback logic as compute_clearing_probabilities.

    Args:
        offer_prices: Ascending offer price levels [n_bands].
        clearing_price_forecast: Point forecast of the clearing price per PTU.
        ensemble_members: Optional list of M scenario series, each [n_ptu].
        confidence_pct: Confidence interval width (default 10.0).

    Returns:
        Nested list [n_ptu][n_bands] with values in [0, 1].
    """
    return compute_clearing_probabilities(
        band_prices=offer_prices,
        point_forecast=clearing_price_forecast,
        ensemble_members=ensemble_members,
        confidence_pct=confidence_pct,
    )


# ============================================================================
# Internal helpers
# ============================================================================


def _empirical_cdf_ge(
    band_arr: np.ndarray,
    point_forecast: list[float],
    ensemble_members: list[list[float]],
) -> list[list[float]]:
    """P(price >= X) from empirical CDF of ensemble members."""
    n_ptu = len(point_forecast)
    n_bands = len(band_arr)
    n_members = len(ensemble_members)

    # Build matrix [n_members x n_ptu]
    ensemble_matrix = np.array(ensemble_members, dtype=float)
    if ensemble_matrix.shape[0] < n_ptu:
        ensemble_matrix = ensemble_matrix.T
    # Ensure shape is [n_members x n_ptu]
    if ensemble_matrix.shape[1] != n_ptu and ensemble_matrix.shape[0] == n_ptu:
        ensemble_matrix = ensemble_matrix.T
    n_members = ensemble_matrix.shape[0]

    result: list[list[float]] = []
    for t in range(n_ptu):
        prices_at_t = ensemble_matrix[:, t]
        row: list[float] = []
        for k in range(n_bands):
            prob = float(np.sum(prices_at_t >= band_arr[k])) / n_members
            row.append(prob)
        result.append(row)
    return result


def _normal_cdf_ge(
    band_arr: np.ndarray,
    point_forecast: list[float],
    confidence_pct: float,
    n_ptu: int,
    n_bands: int,
) -> list[list[float]]:
    """P(price >= X) using Normal(mean=forecast, std=forecast*confidence_pct/200).

    The confidence_pct represents the total interval width at 2 sigma.
    E.g. 10% means the 95% CI spans from forecast-5% to forecast+5%,
    so std = forecast * confidence_pct / 200.
    """
    result: list[list[float]] = []
    for t in range(n_ptu):
        mu = float(point_forecast[t])
        # std = half the confidence interval at 2 sigma
        sigma = max(abs(mu) * confidence_pct / 200.0, _STD_FLOOR_EUR_MWH)
        # P(X >= band_price) = 1 - Phi((band_price - mu) / sigma)
        probs = 1.0 - norm.cdf((band_arr - mu) / sigma)
        result.append([float(p) for p in probs])
    return result
