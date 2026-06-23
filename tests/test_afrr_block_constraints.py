"""Tests for aFRR conservative bid constraints.

Covers three requirements added together:

1. Block-equality regression (dummy-timestep off-by-one):
   All PTUs in the same block must produce the same bid value, including the
   last PTU of each block which was previously left unconstrained.

2. Rule 1 — power headroom: capacity_bid + da_dispatch <= max_power per PTU.

3. Rule 2 — single direction per block: no block bids both aFRR up and aFRR
   down simultaneously.

4. Grid-mismatch detection: when aFRR up and down have different block grids,
   the translation layer raises ValueError (HTTP 422).
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from service.translation.pe_to_rtc import translate_scheduling


# ── helpers ──────────────────────────────────────────────────────────

def _qh_timestamps(n: int, base: str = "2025-08-01") -> tuple[list[str], list[str]]:
    """n quarter-hourly intervals starting at base T00:00Z."""
    from datetime import datetime, timedelta, timezone

    t0 = datetime(2025, 8, 1, tzinfo=timezone.utc)
    starts = [
        (t0 + timedelta(minutes=15 * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for i in range(n)
    ]
    ends = [
        (t0 + timedelta(minutes=15 * (i + 1))).strftime("%Y-%m-%dT%H:%M:%SZ")
        for i in range(n)
    ]
    return starts, ends


def _block_ts(
    name: str,
    block_starts: list[str],
    block_ends: list[str],
    values: list[float],
) -> dict[str, Any]:
    return {
        "name": name,
        "interval_start": block_starts,
        "interval_end": block_ends,
        "values": values,
    }


def _afrr_input(
    *,
    n_ptu: int = 8,
    n_blocks: int = 2,
    afrr_up_open: bool = True,
    afrr_down_open: bool = True,
    up_standby_price: float = 20.0,
    down_standby_price: float = 20.0,
    da_prices: list[float] | None = None,
    initial_soc: float = 10.0,
    mismatched_down_blocks: bool = False,
) -> dict[str, Any]:
    """Build a minimal scheduling input with block-structured aFRR markets.

    PTU resolution: 15 min. Block size: n_ptu // n_blocks PTUs per block.
    """
    starts, ends = _qh_timestamps(n_ptu)
    ptu_per_block = n_ptu // n_blocks

    block_starts_up = [starts[i * ptu_per_block] for i in range(n_blocks)]
    block_ends_up = [ends[min((i + 1) * ptu_per_block - 1, n_ptu - 1)] for i in range(n_blocks)]

    if mismatched_down_blocks and n_ptu >= 3:
        # Give down a 3-block grid that differs from the 2-block up grid
        third = n_ptu // 3
        block_starts_down = [starts[0], starts[third], starts[2 * third]]
        block_ends_down = [
            ends[third - 1],
            ends[2 * third - 1],
            ends[n_ptu - 1],
        ]
        down_n_blocks = 3
    else:
        block_starts_down = block_starts_up
        block_ends_down = block_ends_up
        down_n_blocks = n_blocks

    if da_prices is None:
        # Cheap in first half, expensive in second half → incentivise discharge
        da_prices = [30.0] * (n_ptu // 2) + [90.0] * (n_ptu - n_ptu // 2)

    timeseries: list[dict[str, Any]] = [
        {"name": "day_ahead_price", "values": da_prices},
        {"name": "state_of_charge", "values": [initial_soc]},
        {"name": "afrr_activation_fraction", "values": [0.1] * n_ptu},
    ]
    markets: list[dict[str, Any]] = []

    if afrr_up_open:
        timeseries += [
            _block_ts("afrr_up_standby_price", block_starts_up, block_ends_up,
                      [up_standby_price] * n_blocks),
            _block_ts("afrr_up_price", block_starts_up, block_ends_up,
                      [80.0] * n_blocks),
        ]
        markets.append(
            {"name": "afrr_up", "type": "afrr_capacity", "activation_duration": 900}
        )

    if afrr_down_open:
        timeseries += [
            _block_ts("afrr_down_standby_price", block_starts_down, block_ends_down,
                      [down_standby_price] * down_n_blocks),
            _block_ts("afrr_down_price", block_starts_down, block_ends_down,
                      [50.0] * down_n_blocks),
        ]
        markets.append(
            {"name": "afrr_down", "type": "afrr_capacity", "activation_duration": 900}
        )

    return {
        "interval_start": starts,
        "interval_end": ends,
        "timeseries": timeseries,
        "parameters": [
            {"name": "battery_capacity", "value": 20.0},
            {"name": "max_charge_power", "value": 10.0},
            {"name": "max_discharge_power", "value": 10.0},
            {"name": "efficiency_in", "value": 0.95},
            {"name": "efficiency_out", "value": 0.95},
            {"name": "cost_per_cycle", "value": 2.0},
        ],
        "markets": markets,
    }


# ── translation-layer unit tests ──────────────────────────────────────


class TestAFRRTranslation:
    """Validate block structure produced by translate_scheduling."""

    def test_block_grid_creates_correct_block_structure(self) -> None:
        """8 PTUs with 2 blocks → two groups of 4 PTU indices each."""
        cfg = _afrr_input(n_ptu=8, n_blocks=2, afrr_down_open=False)
        result = translate_scheduling(cfg)
        blocks = result.reserve_config["afrr_up"]["blocks"]
        assert len(blocks) == 2
        assert blocks[0] == [0, 1, 2, 3]
        assert blocks[1] == [4, 5, 6, 7]

    def test_grid_mismatch_raises_value_error(self) -> None:
        """Different block grids for up and down raises ValueError (→ HTTP 422)."""
        cfg = _afrr_input(
            n_ptu=12, n_blocks=2, mismatched_down_blocks=True
        )
        with pytest.raises(ValueError, match="same block grid"):
            translate_scheduling(cfg)

    def test_single_market_open_no_mismatch_error(self) -> None:
        """Only aFRR up open → no grid-mismatch check, no error."""
        cfg = _afrr_input(n_ptu=8, n_blocks=2, afrr_down_open=False)
        result = translate_scheduling(cfg)
        assert result.reserve_config["afrr_up"]["open"] is True
        assert result.reserve_config["afrr_down"]["open"] is False


# ── end-to-end API tests ──────────────────────────────────────────────


class TestAFRRConservativeBids:
    """API-level verification of block-equality and conservative bid rules."""

    def test_bids_are_block_constant(self, client: Any) -> None:
        """Block-equality regression for off-by-one fix.

        The API returns one bid value per block (block-resolution). The regression
        check verifies that the bid respects power headroom for EVERY PTU in the
        block, including the last one which was previously left unconstrained.
        Before the fix, bid_up[block 0] = 10 while discharge[PTU 3] = 8.3125,
        summing to 18.3125 > 10.
        """
        cfg = _afrr_input(
            n_ptu=8,
            n_blocks=2,
            afrr_down_open=False,
            up_standby_price=20.0,
        )
        cfg["parameters"].append({"name": "skip_counterfactual_reserves", "value": 1.0})
        resp = client.post(
            "/v1/models/bess_day_ahead/submit_sync",
            json={"model_input_data": cfg, "include_diagnostics": False},
        )
        assert resp.status_code == 200, resp.text
        members = resp.json()["result"]["members"]["default"]
        bid_up = members["bid_afrr_up_total"]["values"]
        discharge = members["day_ahead_power_out"]["values"]
        # bid_up is block-resolution (one value per block), discharge is PTU-resolution
        assert len(bid_up) == 2, f"Expected 2 block values, got {len(bid_up)}"
        assert len(discharge) == 8

        max_power = 10.0
        blocks = [[0, 1, 2, 3], [4, 5, 6, 7]]
        for b, block in enumerate(blocks):
            for t in block:
                assert bid_up[b] + discharge[t] <= max_power + 1e-3, (
                    f"Block {b} off-by-one regression: "
                    f"bid_up={bid_up[b]:.4f} + discharge[{t}]={discharge[t]:.4f} "
                    f"= {bid_up[b]+discharge[t]:.4f} > {max_power}"
                )

    def test_rule1_bid_plus_dispatch_within_max_power(self, client: Any) -> None:
        """bid_afrr_up[t] + discharge[t] <= max_power for every PTU."""
        cfg = _afrr_input(
            n_ptu=8,
            n_blocks=2,
            afrr_down_open=False,
            up_standby_price=20.0,
        )
        cfg["parameters"].append({"name": "skip_counterfactual_reserves", "value": 1.0})
        resp = client.post(
            "/v1/models/bess_day_ahead/submit_sync",
            json={"model_input_data": cfg, "include_diagnostics": False},
        )
        assert resp.status_code == 200, resp.text
        members = resp.json()["result"]["members"]["default"]
        bid_up = members["bid_afrr_up_total"]["values"]
        discharge = members["day_ahead_power_out"]["values"]
        charge = members["day_ahead_power_in"]["values"]
        bid_down = members["bid_afrr_down_total"]["values"]
        max_power = 10.0

        # bid_up/bid_down are block-resolution (2 values); discharge/charge are PTU-resolution (8)
        blocks = [[0, 1, 2, 3], [4, 5, 6, 7]]
        for b, block in enumerate(blocks):
            for t in block:
                assert bid_up[b] + discharge[t] <= max_power + 1e-3, (
                    f"Rule 1 up violated at block {b} PTU {t}: "
                    f"bid_up={bid_up[b]:.4f} discharge={discharge[t]:.4f} sum={bid_up[b]+discharge[t]:.4f}"
                )
                assert bid_down[b] + charge[t] <= max_power + 1e-3, (
                    f"Rule 1 down violated at block {b} PTU {t}: "
                    f"bid_down={bid_down[b]:.4f} charge={charge[t]:.4f} sum={bid_down[b]+charge[t]:.4f}"
                )

    def test_rule2_single_direction_per_block(self, client: Any) -> None:
        """No block bids nonzero on both aFRR up and aFRR down simultaneously."""
        cfg = _afrr_input(
            n_ptu=8,
            n_blocks=2,
            afrr_up_open=True,
            afrr_down_open=True,
            up_standby_price=20.0,
            down_standby_price=20.0,
        )
        cfg["parameters"].append({"name": "skip_counterfactual_reserves", "value": 1.0})
        resp = client.post(
            "/v1/models/bess_day_ahead/submit_sync",
            json={"model_input_data": cfg, "include_diagnostics": False},
        )
        assert resp.status_code == 200, resp.text
        members = resp.json()["result"]["members"]["default"]
        bid_up = members["bid_afrr_up_total"]["values"]
        bid_down = members["bid_afrr_down_total"]["values"]

        # bid_up/bid_down are block-resolution (one value per block)
        THRESHOLD = 1e-3
        for b in range(2):
            has_up = bid_up[b] > THRESHOLD
            has_down = bid_down[b] > THRESHOLD
            assert not (has_up and has_down), (
                f"Rule 2 violated in block {b}: bid_up={bid_up[b]:.4f}, bid_down={bid_down[b]:.4f}"
            )

    def test_grid_mismatch_yields_422(self, client: Any) -> None:
        """Mismatched aFRR up/down block grids return HTTP 422."""
        cfg = _afrr_input(
            n_ptu=12, n_blocks=2, mismatched_down_blocks=True
        )
        resp = client.post(
            "/v1/models/bess_day_ahead/submit_sync",
            json={"model_input_data": cfg, "include_diagnostics": False},
        )
        assert resp.status_code == 422
        assert "same block grid" in resp.json()["detail"]["message"]

    def test_only_afrr_up_open_no_direction_constraint(self, client: Any) -> None:
        """When only aFRR up is open, no direction gating is applied and solver succeeds."""
        cfg = _afrr_input(
            n_ptu=8,
            n_blocks=2,
            afrr_up_open=True,
            afrr_down_open=False,
            up_standby_price=20.0,
        )
        cfg["parameters"].append({"name": "skip_counterfactual_reserves", "value": 1.0})
        resp = client.post(
            "/v1/models/bess_day_ahead/submit_sync",
            json={"model_input_data": cfg, "include_diagnostics": False},
        )
        assert resp.status_code == 200, resp.text
        members = resp.json()["result"]["members"]["default"]
        bid_up = members["bid_afrr_up_total"]["values"]
        assert any(v > 1e-3 for v in bid_up), "Expected nonzero aFRR up bids"
