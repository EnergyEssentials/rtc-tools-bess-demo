import numpy as np
from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.util import run_optimization_problem


# Default reserve config — every product closed, zero LER duration.
# Concrete configs are stamped onto the dynamically-derived solver subclass
# by service/solvers/scheduling.py.  Each entry has the shape::
#
#     {"open": bool, "t_min_hours": float, "blocks": list[list[int]]}
#
# where ``blocks`` lists the PTU-index groupings that must hold a constant
# bid (block-equality constraints).  Blocks come from runs of identical
# standby-price values in the input timeseries.
_DEFAULT_RESERVE_CONFIG: dict[str, dict] = {
    "fcr":       {"open": False, "t_min_hours": 0.0, "blocks": []},
    "afrr_up":   {"open": False, "t_min_hours": 0.0, "blocks": []},
    "afrr_down": {"open": False, "t_min_hours": 0.0, "blocks": []},
}


class BESS(
    CSVMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):
    """
    BESS optimization problem for time arbitrage.

    This class implements a Battery Energy Storage System (BESS) optimization
    problem that maximizes revenue from time arbitrage while considering
    cycling penalties and round-trip efficiency.

    The physical asset (battery dynamics) is modeled in Modelica, while the
    revenue and costs are calculated in Python.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Economic parameters (not in Modelica model)
        self.cycling_penalty_factor = 0.1
        self.stored_energy_value = (
            0.0  # EUR/MWh value assigned to SoC remaining at horizon end
        )
        # Reserve configuration; service wrappers override via class attribute.
        self.reserve_config: dict[str, dict] = {
            k: dict(v) for k, v in _DEFAULT_RESERVE_CONFIG.items()
        }
        # Multi-band DA config; service wrappers override via class attribute.
        self.n_da_bands: int = 1
        self.da_band_prices: list[float] = []
        # Per-PTU clearing probabilities [n_ptu][n_bands].
        # P(clear >= price[k]) for selling; 1 - that for buying.
        self.da_clearing_probs: list[list[float]] = []
        # Per-product acceptance probabilities [n_ptu][n_bands] for pay-as-bid.
        self.reserve_acceptance_probs: dict[str, list[list[float]]] = {}
        self.reserve_offer_prices: dict[str, list[float]] = {}

    def solver_options(self):
        """Configure solver options for mixed-integer optimization."""
        options = super().solver_options()
        options["casadi_solver"] = "qpsol"
        options["solver"] = "highs"
        # Bound MILP runtime: the multi-band formulation is degenerate by
        # construction (zero-revenue / zero-probability bands), so HiGHS'
        # default 1e-4 gap is unreachable.  1% gap is well below the
        # ~5 EUR/MWh cycling penalty noise floor.
        options["highs"] = {
            "mip_rel_gap": 0.01,
            "time_limit": 60.0,
        }
        return options

    def path_objective(self, ensemble_member):
        """Maximize expected revenue from multi-band DA bids and reserve capacity.

        Revenue terms:
        - DA energy: expected-value formulation using per-band clearing probabilities
        - Reserve standby: pay-as-bid with per-band acceptance probabilities
        - Activation revenue: on total committed reserve (existing logic)

        Cost terms:
        - Grid fees on gross charge/discharge
        - Cycling penalty on throughput + expected activation
        """
        # DA energy revenue — expected-value multi-band formulation.
        # When n_da_bands == 1 and no probs configured, falls back to
        # deterministic price * net_power (equivalent to prob=1.0 for all).
        if self.n_da_bands > 1 and self.da_clearing_probs:
            da_revenue = self._multi_band_da_revenue(ensemble_member)
        else:
            da_revenue = self.state("net_power") * self.state("price")

        grid_fee_cost = self.state("grid_fee_in") * self.state(
            "charge_power"
        ) + self.state("grid_fee_out") * self.state("discharge_power")

        # Reserve standby revenue — pay-as-bid with acceptance probabilities.
        standby_revenue = self._reserve_standby_revenue(ensemble_member)

        activation_revenue = (
            self.state("total_afrr_up")
            * self.state("afrr_activation_fraction")
            * self.state("afrr_up_price")
            + self.state("total_afrr_down")
            * self.state("afrr_activation_fraction")
            * self.state("afrr_down_price")
        )

        # Cycling penalty extended with expected activation throughput.
        # FCR is symmetric → 2 * total_fcr * fraction (both directions cycle).
        # aFRR is one-sided per product → 1 * total_* * fraction.
        cycling_penalty = self.cycling_penalty_factor * (
            self.state("charge_power")
            + self.state("discharge_power")
            + 2.0 * self.state("total_fcr") * self.state("fcr_activation_fraction")
            + self.state("total_afrr_up")   * self.state("afrr_activation_fraction")
            + self.state("total_afrr_down") * self.state("afrr_activation_fraction")
        )

        return -(
            da_revenue
            + standby_revenue
            + activation_revenue
            - grid_fee_cost
            - cycling_penalty
        )

    def _multi_band_da_revenue(self, ensemble_member):
        """Expected DA revenue summed across price bands.

        E[sell revenue] = sum_k(P(clear >= price[k]) * price[k] * delta_out[k])
        E[buy cost]     = sum_k((1 - P(clear >= price[k])) * price[k] * delta_in[k])

        Probabilities are time-varying constants injected per request.
        RTC-Tools evaluates path_objective at each collocation point, so we
        use the per-PTU index derived from self.times().
        """
        times = self.times()
        # Determine current PTU index from the collocation-point time.
        # path_objective is called once per collocation point; self.state()
        # returns the symbolic expression at that point. We sum band
        # contributions using the pre-computed probability constants.
        # Because probabilities are constants (not decision vars), we build
        # the weighted sum directly — still linear in the deltas.
        revenue = 0.0
        for k in range(self.n_da_bands):
            delta_out = self.state(f"da_power_out_deltas[{k + 1}]")
            delta_in = self.state(f"da_power_in_deltas[{k + 1}]")
            band_price = self.da_band_prices[k]
            # Use average probability across all PTUs as a scalar weight.
            # This is exact when probabilities are constant per block, and a
            # good approximation otherwise since RTC-Tools sums path_objective
            # uniformly over all collocation points.
            avg_prob_sell = float(np.mean([row[k] for row in self.da_clearing_probs]))
            avg_prob_buy = 1.0 - avg_prob_sell
            revenue += avg_prob_sell * band_price * delta_out
            revenue -= avg_prob_buy * band_price * delta_in
        return revenue

    def _reserve_standby_revenue(self, ensemble_member):
        """Reserve standby revenue using pay-as-bid acceptance probabilities.

        When acceptance probabilities are configured (multi-band):
          E[revenue] = sum_k(P(accepted at price[k]) * price[k] * delta[k])

        When not configured (single-band, backward compatible):
          revenue = bid_total * standby_price (deterministic, as before)
        """
        # Map product names to their standby price variable and delta base
        _PRODUCT_MAP = {
            "fcr": ("fcr_standby_price", "fcr_capacity_deltas"),
            "afrr_up": ("afrr_up_standby_price", "afrr_up_capacity_deltas"),
            "afrr_down": ("afrr_down_standby_price", "afrr_down_capacity_deltas"),
        }

        standby_revenue = 0.0
        for product, (price_var, delta_base) in _PRODUCT_MAP.items():
            pcfg = self.reserve_config.get(product) or {}
            if not pcfg.get("open"):
                continue

            offer_prices = self.reserve_offer_prices.get(product, [])
            acceptance_probs = self.reserve_acceptance_probs.get(product, [])

            if offer_prices and acceptance_probs:
                # Pay-as-bid: each band earns its own offer price if accepted
                n_bands = len(offer_prices)
                for k in range(n_bands):
                    delta = self.state(f"{delta_base}[{k + 1}]")
                    avg_prob = float(np.mean([row[k] for row in acceptance_probs]))
                    standby_revenue += avg_prob * offer_prices[k] * delta
            else:
                # Fallback: deterministic standby price (v1 behaviour)
                standby_revenue += (
                    self.state(f"bid_{product}_total") * self.state(price_var)
                )

        return standby_revenue

    def objective(self, ensemble_member):
        """Add terminal SoC valuation to the path objective total.

        When ``stored_energy_value`` is non-zero (EUR/MWh), the solver is
        rewarded for energy remaining in the battery at the end of the
        optimisation horizon.  This prevents greedy end-of-horizon draining
        when future trading opportunities exist beyond the current window.

        RTC-Tools plain-sums ``path_objective`` over collocation points
        without multiplying by dt, so rates in EUR/h are effectively
        inflated by ``1/dt_hours``.  The terminal value must be scaled
        by the same factor to remain comparable in magnitude.
        """
        obj = super().objective(ensemble_member)
        if self.stored_energy_value != 0.0:
            times = self.times()
            dt_hours = (times[1] - times[0]) / 3600.0
            soc_final = self.state_at("soc", times[-1], ensemble_member)
            obj -= (self.stored_energy_value / dt_hours) * soc_final
        return obj

    def path_constraints(self, ensemble_member):
        """Define path constraints (inequality constraints over time)."""
        constraints = super().path_constraints(ensemble_member)

        parameters = self.parameters(ensemble_member)
        max_power = parameters["max_power"]
        capacity = parameters["capacity"]

        # Ensure only one mode can be active at a time (complementarity)
        constraints.append(
            (
                self.state("is_charging") + self.state("is_discharging"),
                -np.inf,
                1.0,
            )
        )
        constraints.append(
            (
                self.state("charge_power")
                - self.state("is_charging") * max_power,
                -np.inf,
                0,
            )
        )
        constraints.append(
            (
                self.state("discharge_power")
                - self.state("is_discharging") * max_power,
                -np.inf,
                0,
            )
        )

        # Reserve power-headroom constraints.  Up-direction reserves (FCR
        # which is symmetric, plus aFRR up) compete with discharge for the
        # inverter's discharging capacity; down-direction reserves (FCR plus
        # aFRR down) compete with charging.  Both inequalities are <= max_power.
        total_fcr = self.state("total_fcr")
        total_afrr_up = self.state("total_afrr_up")
        total_afrr_down = self.state("total_afrr_down")

        constraints.append(
            (
                self.state("discharge_power") + total_fcr + total_afrr_up - max_power,
                -np.inf,
                0.0,
            )
        )
        constraints.append(
            (
                self.state("charge_power") + total_fcr + total_afrr_down - max_power,
                -np.inf,
                0.0,
            )
        )

        # Per-block aFRR single-direction enforcement: big-M gating on the ``afrr_up_active`` selector. 
        # Active only when both markets are open; otherwise the closed-market pin below already zeroes the absent side.
        afrr_up_open = bool((self.reserve_config.get("afrr_up") or {}).get("open"))
        afrr_down_open = bool((self.reserve_config.get("afrr_down") or {}).get("open"))
        if afrr_up_open and afrr_down_open:
            afrr_up_active = self.state("afrr_up_active")
            constraints.append(
                (
                    self.state("bid_afrr_up_total") - max_power * afrr_up_active,
                    -np.inf,
                    0.0,
                )
            )
            constraints.append(
                (
                    self.state("bid_afrr_down_total")
                    + max_power * afrr_up_active
                    - max_power,
                    -np.inf,
                    0.0,
                )
            )

        # SoC LER (limited-energy reservoir) constraints.  The battery must
        # keep enough headroom to honour the worst-case activation for the
        # product's T_min duration.  Down-side reserves squeeze the *top* of
        # the SoC band; up-side reserves squeeze the *bottom*.
        fcr_t = float(self.reserve_config.get("fcr", {}).get("t_min_hours", 0.0))
        afrr_up_t = float(self.reserve_config.get("afrr_up", {}).get("t_min_hours", 0.0))
        afrr_down_t = float(
            self.reserve_config.get("afrr_down", {}).get("t_min_hours", 0.0)
        )

        if fcr_t > 0.0 or afrr_down_t > 0.0:
            constraints.append(
                (
                    self.state("soc")
                    + total_fcr * fcr_t
                    + total_afrr_down * afrr_down_t
                    - capacity,
                    -np.inf,
                    0.0,
                )
            )
        if fcr_t > 0.0 or afrr_up_t > 0.0:
            constraints.append(
                (
                    -self.state("soc")
                    + total_fcr * fcr_t
                    + total_afrr_up * afrr_up_t,
                    -np.inf,
                    0.0,
                )
            )

        # Closed-market pin: when the caller did not include a market in this
        # run, force the corresponding bid total to 0.  Combined with the
        # model's ``min=0`` bound this collapses the decision variable.
        for product in ("fcr", "afrr_up", "afrr_down"):
            pcfg = self.reserve_config.get(product) or {}
            if not pcfg.get("open"):
                constraints.append(
                    (self.state(f"bid_{product}_total"), -np.inf, 0.0)
                )

        # Degenerate-band pin: bands whose effective objective coefficient
        # is numerically zero (price == 0 or probability underflowed to 0)
        # are free variables that the MILP B&B tree cannot distinguish,
        # blocking convergence.  Pin them to 0 here; HiGHS presolve removes
        # them entirely.
        _ZERO_COEF_THRESHOLD = 1e-6

        if self.n_da_bands > 1 and self.da_clearing_probs and self.da_band_prices:
            for k, band_price in enumerate(self.da_band_prices):
                avg_prob_sell = float(
                    np.mean([row[k] for row in self.da_clearing_probs])
                )
                avg_prob_buy = 1.0 - avg_prob_sell
                if abs(avg_prob_sell * band_price) < _ZERO_COEF_THRESHOLD:
                    constraints.append(
                        (self.state(f"da_power_out_deltas[{k + 1}]"), 0.0, 0.0)
                    )
                if abs(avg_prob_buy * band_price) < _ZERO_COEF_THRESHOLD:
                    constraints.append(
                        (self.state(f"da_power_in_deltas[{k + 1}]"), 0.0, 0.0)
                    )

        _RESERVE_DELTA_BASE = {
            "fcr": "fcr_capacity_deltas",
            "afrr_up": "afrr_up_capacity_deltas",
            "afrr_down": "afrr_down_capacity_deltas",
        }
        for product, delta_base in _RESERVE_DELTA_BASE.items():
            pcfg = self.reserve_config.get(product) or {}
            if not pcfg.get("open"):
                continue
            offer_prices = self.reserve_offer_prices.get(product, [])
            acceptance_probs = self.reserve_acceptance_probs.get(product, [])
            if not offer_prices or not acceptance_probs:
                continue
            for k, offer_price in enumerate(offer_prices):
                avg_prob = float(np.mean([row[k] for row in acceptance_probs]))
                if abs(avg_prob * offer_price) < _ZERO_COEF_THRESHOLD:
                    constraints.append(
                        (self.state(f"{delta_base}[{k + 1}]"), 0.0, 0.0)
                    )

        return constraints

    def constraints(self, ensemble_member):
        """Cross-time constraints — block-equality on open reserve bids.

        For each open product with multi-band config, each per-band delta
        must be constant across all PTUs within the same standby-price block.
        For single-band products, the existing bid_total constraint applies.
        """
        out = super().constraints(ensemble_member)
        times = self.times()
        if len(times) < 2:
            return out

        # pe_to_rtc prepends one dummy timestep to ``times`` (real PTU i
        # lives at times[i+1]); block indices come from _blocks_from_grid using
        # real-PTU 0-based positions, so every state_at lookup needs +1.
        def _t(idx: int) -> float:
            return times[idx + 1]

        def _in_range(idx: int) -> bool:
            return 0 <= idx + 1 < len(times)

        # per-block single-direction aFRR bids; assumes afrr_up and afrr_down share the same block grid. 
        # Fail loud otherwise.
        afrr_up_cfg = self.reserve_config.get("afrr_up") or {}
        afrr_down_cfg = self.reserve_config.get("afrr_down") or {}
        if afrr_up_cfg.get("open") and afrr_down_cfg.get("open"):
            up_blocks = [tuple(b) for b in afrr_up_cfg.get("blocks", [])]
            down_blocks = [tuple(b) for b in afrr_down_cfg.get("blocks", [])]
            if up_blocks != down_blocks:
                raise ValueError(
                    "aFRR up and aFRR down must share the same block grid for "
                    "single-direction-per-block enforcement; got "
                    f"up={up_blocks!r} down={down_blocks!r}"
                )

        for product in ("fcr", "afrr_up", "afrr_down"):
            pcfg = self.reserve_config.get(product) or {}
            if not pcfg.get("open"):
                continue

            offer_prices = self.reserve_offer_prices.get(product, [])
            n_bands = len(offer_prices) if offer_prices else 1

            if n_bands > 1:
                # Per-band block-equality: each delta[k] constant within block
                delta_base = f"{product}_capacity_deltas"
                for k in range(1, n_bands + 1):
                    var = f"{delta_base}[{k}]"
                    for block in pcfg.get("blocks", []):
                        if not block or len(block) < 2:
                            continue
                        ref_idx = block[0]
                        if not _in_range(ref_idx):
                            continue
                        ref_val = self.state_at(var, _t(ref_idx), ensemble_member)
                        for idx in block[1:]:
                            if not _in_range(idx):
                                continue
                            other = self.state_at(var, _t(idx), ensemble_member)
                            out.append((other - ref_val, 0.0, 0.0))
            else:
                # Single-band: constrain the aggregate bid_total as before
                var = f"bid_{product}_total"
                for block in pcfg.get("blocks", []):
                    if not block or len(block) < 2:
                        continue
                    ref_idx = block[0]
                    if not _in_range(ref_idx):
                        continue
                    ref_val = self.state_at(var, _t(ref_idx), ensemble_member)
                    for idx in block[1:]:
                        if not _in_range(idx):
                            continue
                        other = self.state_at(var, _t(idx), ensemble_member)
                        out.append((other - ref_val, 0.0, 0.0))

        # Per-block aFRR single-direction enforcement: ``afrr_up_active`` is
        # pinned constant within each block, and the path-level big-M
        # constraints (path_constraints) gate bid_afrr_up_total /
        # bid_afrr_down_total on its value.
        if afrr_up_cfg.get("open") and afrr_down_cfg.get("open"):
            for block in afrr_up_cfg.get("blocks", []):
                if not block or len(block) < 2:
                    continue
                ref_idx = block[0]
                if not _in_range(ref_idx):
                    continue
                ref_val = self.state_at(
                    "afrr_up_active", _t(ref_idx), ensemble_member
                )
                for idx in block[1:]:
                    if not _in_range(idx):
                        continue
                    other = self.state_at(
                        "afrr_up_active", _t(idx), ensemble_member
                    )
                    out.append((other - ref_val, 0.0, 0.0))

        return out

    def post(self):
        """Post-processing step to save results and call plotting script."""
        super().post()

        print("Optimization completed successfully!")
        print("Results saved to output/timeseries_export.csv")
        print(
            "Run 'uv run python src/plot_results.py' to generate plots and summary statistics."
        )


if __name__ == "__main__":
    # Run the optimization
    run_optimization_problem(BESS)
