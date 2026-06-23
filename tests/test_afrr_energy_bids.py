"""Tests for aFRR energy bid pricing (IC orderbook mid-price)."""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd
import pytest

from service.translation.pe_to_rtc import (
    _extract_afrr_energy_market,
    _parse_iso_utc,
    translate_intraday,
)
from service.translation.rtc_to_pe import _compute_afrr_energy_bids


# ── _extract_afrr_energy_market tests ─────────────────────────────────


class TestExtractAfrrEnergyMarket:
    """Input parsing for the aFRR energy bid market."""

    def test_no_market_returns_defaults(self) -> None:
        model_input: dict[str, Any] = {"markets": [], "timeseries": [], "parameters": []}
        ptu_starts = [_parse_iso_utc(f"2025-08-01T{h:02d}:00:00Z") for h in range(4)]
        info: list[str] = []

        up, down, mask, n_bands, grid = _extract_afrr_energy_market(
            model_input, ptu_starts, info
        )

        assert up == [0.0] * 4
        assert down == [0.0] * 4
        assert mask == [False] * 4
        assert n_bands == 0
        assert grid is None
        assert info == []

    def test_market_with_obligations_populates_fields(self) -> None:
        ptu_starts = [_parse_iso_utc(f"2025-08-01T{h:02d}:00:00Z") for h in range(4)]
        model_input: dict[str, Any] = {
            "markets": [
                {"name": "afrr_energy", "type": "afrr_energy_bid", "n_price_bands": 2}
            ],
            "timeseries": [
                {
                    "name": "afrr_up_position",
                    "values": [10.0, 10.0],
                    "interval_start": ["2025-08-01T01:00:00Z", "2025-08-01T02:00:00Z"],
                    "interval_end": ["2025-08-01T02:00:00Z", "2025-08-01T03:00:00Z"],
                },
                {
                    "name": "afrr_down_position",
                    "values": [5.0, 5.0],
                    "interval_start": ["2025-08-01T01:00:00Z", "2025-08-01T02:00:00Z"],
                    "interval_end": ["2025-08-01T02:00:00Z", "2025-08-01T03:00:00Z"],
                },
            ],
            "parameters": [],
        }
        info: list[str] = []

        up, down, mask, n_bands, grid = _extract_afrr_energy_market(
            model_input, ptu_starts, info
        )

        assert up == [0.0, 10.0, 10.0, 0.0]
        assert down == [0.0, 5.0, 5.0, 0.0]
        assert mask == [False, True, True, False]
        assert n_bands == 2
        assert grid is not None
        assert len(grid["interval_start"]) == 2

    def test_split_market_names_afrr_energy_up_down_recognised(self) -> None:
        """Markets named afrr_energy_up / afrr_energy_down with type afrr_energy
        are recognised and afrr_up/down_position drives the obligation."""
        ptu_starts = [_parse_iso_utc(f"2025-08-01T{h:02d}:00:00Z") for h in range(4)]
        model_input: dict[str, Any] = {
            "markets": [
                {"name": "afrr_energy_up",   "type": "afrr_energy", "interval_length_minutes": 60},
                {"name": "afrr_energy_down", "type": "afrr_energy", "interval_length_minutes": 60},
            ],
            "timeseries": [
                {"name": "afrr_up_position",   "values": [0.0, 7.0, 7.0, 0.0]},
                {"name": "afrr_down_position", "values": [0.0, 3.0, 3.0, 0.0]},
            ],
            "parameters": [],
        }
        info: list[str] = []

        up, down, mask, n_bands, grid = _extract_afrr_energy_market(
            model_input, ptu_starts, info
        )

        assert up == [0.0, 7.0, 7.0, 0.0]
        assert down == [0.0, 3.0, 3.0, 0.0]
        assert mask == [False, True, True, False]
        assert n_bands == 1
        assert any("afrr_energy" in e for e in info)

    def test_translate_intraday_includes_afrr_energy_fields(
        self, intraday_input: dict[str, Any]
    ) -> None:
        inp = copy.deepcopy(intraday_input)
        n = len(inp["interval_start"])

        inp["markets"].append(
            {"name": "afrr_energy", "type": "afrr_energy_bid", "n_price_bands": 1}
        )
        inp["timeseries"].append(
            {"name": "afrr_up_position", "values": [8.0] * n}
        )
        inp["timeseries"].append(
            {"name": "afrr_down_position", "values": [6.0] * n}
        )

        result = translate_intraday(inp)

        assert result.afrr_energy_obligation_up == [8.0] * n
        assert result.afrr_energy_obligation_down == [6.0] * n
        assert result.afrr_energy_open_mask == [True] * n
        assert result.afrr_energy_n_bands == 1
        assert not hasattr(result, "afrr_energy_markup")


# ── _compute_afrr_energy_bids tests ──────────────────────────────────


class TestComputeAfrrEnergyBids:
    """IC mid-price computation for aFRR energy bid prices."""

    def _make_dfs(
        self,
        n: int,
        bid_price: float = 40.0,
        ask_price: float = 50.0,
        bid_volume: float = 5.0,
        ask_volume: float = 5.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_output = pd.DataFrame({
            "soc": [10.0] * n,
            "charge_power": [0.0] * n,
            "discharge_power": [0.0] * n,
        })
        df_input = pd.DataFrame({
            "bid_prices[1]": [bid_price] * n,
            "ask_prices[1]": [ask_price] * n,
            "bid_volumes[1]": [bid_volume] * n,
            "ask_volumes[1]": [ask_volume] * n,
        })
        return df_output, df_input

    def test_mid_price_up(self) -> None:
        """Up-direction price equals (bid + ask) / 2."""
        n = 4
        df_out, df_in = self._make_dfs(n, bid_price=40.0, ask_price=50.0)
        info: list[str] = []

        members = _compute_afrr_energy_bids(
            df_out, df_in, [],
            [10.0] * n, [0.0] * n, [True] * n,
            1, None, info,
        )

        prices = members["afrr_energy_up_price[1]"]["values"]
        assert prices == [pytest.approx(45.0)] * n

    def test_mid_price_down_equals_up(self) -> None:
        """Down-direction price equals the same IC mid (same source)."""
        n = 4
        df_out, df_in = self._make_dfs(n, bid_price=40.0, ask_price=50.0)
        info: list[str] = []

        members = _compute_afrr_energy_bids(
            df_out, df_in, [],
            [0.0] * n, [8.0] * n, [True] * n,
            1, None, info,
        )

        up = members["afrr_energy_up_price[1]"]["values"]
        down = members["afrr_energy_down_price[1]"]["values"]
        assert up == down

    def test_negative_mid_price_propagates(self) -> None:
        """Negative IC mid (cheap power) passes through unchanged."""
        n = 2
        df_out, df_in = self._make_dfs(n, bid_price=-20.0, ask_price=-10.0)
        info: list[str] = []

        members = _compute_afrr_energy_bids(
            df_out, df_in, [],
            [10.0] * n, [10.0] * n, [True] * n,
            1, None, info,
        )

        prices = members["afrr_energy_up_price[1]"]["values"]
        assert prices == [pytest.approx(-15.0)] * n

    def test_fallback_to_da_price(self) -> None:
        """No IC depth → fall back to day-ahead price."""
        n = 3
        df_out, df_in = self._make_dfs(n, bid_volume=0.0, ask_volume=0.0)
        info: list[str] = []
        da_prices = [25.0, 30.0, 35.0]

        members = _compute_afrr_energy_bids(
            df_out, df_in, da_prices,
            [10.0] * n, [10.0] * n, [True] * n,
            1, None, info,
        )

        prices = members["afrr_energy_up_price[1]"]["values"]
        assert prices == [25.0, 30.0, 35.0]
        # Source label appears in info
        assert any("da_price_fallback" in line for line in info)

    def test_fallback_to_zero(self) -> None:
        """No IC depth and no DA → zero."""
        n = 2
        df_out, df_in = self._make_dfs(n, bid_volume=0.0, ask_volume=0.0)
        info: list[str] = []

        members = _compute_afrr_energy_bids(
            df_out, df_in, [],
            [10.0] * n, [10.0] * n, [True] * n,
            1, None, info,
        )

        prices = members["afrr_energy_up_price[1]"]["values"]
        assert prices == [0.0, 0.0]
        assert any("zero_fallback" in line for line in info)

    def test_volumes_equal_obligations(self) -> None:
        """Output volumes equal the obligation inputs on open PTUs."""
        n = 3
        df_out, df_in = self._make_dfs(n)
        info: list[str] = []

        members = _compute_afrr_energy_bids(
            df_out, df_in, [],
            [10.0, 0.0, 5.0], [7.0, 3.0, 0.0], [True, True, True],
            1, None, info,
        )

        assert members["afrr_energy_up_volume[1]"]["values"] == [10.0, 0.0, 5.0]
        assert members["afrr_energy_down_volume[1]"]["values"] == [7.0, 3.0, 0.0]

    def test_closed_ptus_have_zero(self) -> None:
        """PTUs where open_mask is False emit zero price and zero volume."""
        n = 4
        df_out, df_in = self._make_dfs(n)
        info: list[str] = []

        members = _compute_afrr_energy_bids(
            df_out, df_in, [],
            [10.0] * n, [5.0] * n, [False, True, False, True],
            1, None, info,
        )

        prices_up = members["afrr_energy_up_price[1]"]["values"]
        volumes_up = members["afrr_energy_up_volume[1]"]["values"]
        assert prices_up[0] == 0.0 and prices_up[2] == 0.0
        assert prices_up[1] != 0.0 and prices_up[3] != 0.0
        assert volumes_up[0] == 0.0 and volumes_up[2] == 0.0
        assert volumes_up[1] == 10.0 and volumes_up[3] == 10.0

    def test_all_bands_carry_full_value(self) -> None:
        """Bands 2..N carry the same price and same volume as band 1."""
        n = 2
        df_out, df_in = self._make_dfs(n, bid_price=40.0, ask_price=50.0)
        info: list[str] = []

        members = _compute_afrr_energy_bids(
            df_out, df_in, [],
            [10.0] * n, [5.0] * n, [True] * n,
            3, None, info,
        )

        for k in (1, 2, 3):
            assert members[f"afrr_energy_up_price[{k}]"]["values"] == [
                pytest.approx(45.0)
            ] * n
            assert members[f"afrr_energy_up_volume[{k}]"]["values"] == [10.0] * n
            assert members[f"afrr_energy_down_price[{k}]"]["values"] == [
                pytest.approx(45.0)
            ] * n
            assert members[f"afrr_energy_down_volume[{k}]"]["values"] == [5.0] * n

    def test_info_transparency_note(self) -> None:
        """Info entries name the price source and the pay-as-cleared rationale."""
        n = 2
        df_out, df_in = self._make_dfs(n)
        info: list[str] = []

        _compute_afrr_energy_bids(
            df_out, df_in, [],
            [10.0] * n, [5.0] * n, [True, False],
            1, None, info,
        )

        bid_info = [i for i in info if i.startswith("afrr_energy_bid_")]
        assert len(bid_info) == 2
        for line in bid_info:
            assert "ic_orderbook_mid_price" in line
            assert "pay-as-cleared activation" in line

    def test_no_open_ptus_returns_empty(self) -> None:
        n = 3
        df_out, df_in = self._make_dfs(n)
        info: list[str] = []

        members = _compute_afrr_energy_bids(
            df_out, df_in, [],
            [0.0] * n, [0.0] * n, [False] * n,
            1, None, info,
        )

        assert members == {}

    def test_grid_shaping(self) -> None:
        """Output values are collapsed onto the provided grid blocks."""
        n = 4
        df_out, df_in = self._make_dfs(n)
        grid = {
            "interval_start": ["2025-08-01T01:00:00Z", "2025-08-01T02:00:00Z"],
            "interval_end": ["2025-08-01T02:00:00Z", "2025-08-01T03:00:00Z"],
            "blocks": [[1], [2]],
        }
        info: list[str] = []

        members = _compute_afrr_energy_bids(
            df_out, df_in, [],
            [0.0, 10.0, 10.0, 0.0],
            [0.0, 5.0, 5.0, 0.0],
            [False, True, True, False],
            1, grid, info,
        )

        assert len(members["afrr_energy_up_price[1]"]["values"]) == 2
        assert len(members["afrr_energy_up_volume[1]"]["values"]) == 2
        assert "interval_start" in members["afrr_energy_up_price[1]"]
        assert "interval_end" in members["afrr_energy_up_price[1]"]
