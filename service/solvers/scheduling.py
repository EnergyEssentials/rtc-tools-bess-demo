"""Configurable BESS solver for day-ahead scheduling.

Inherits the demo's ``BESS`` class but accepts runtime-configurable
``cycling_penalty_factor`` and ``stored_energy_value`` via class
attributes set per request.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the scheduling source directory to the path so we can import BESS
_scheduling_src = str(
    Path(__file__).resolve().parent.parent.parent / "scheduling" / "src"
)
if _scheduling_src not in sys.path:
    sys.path.insert(0, _scheduling_src)

from bess import BESS  # noqa: E402


class ConfigurableBESS(BESS):
    """BESS solver with runtime-configurable economic parameters.

    Battery parameters (``capacity``, ``max_power``, ``efficiency``) are
    overridden via ``parameters.csv``.  Economic config (cycling penalty,
    stored energy value, reserve config, multi-band probabilities) needs
    Python-level overrides because they are not Modelica parameters.

    Per-request values are injected via class attributes on a dynamically
    created subclass before instantiation.
    """

    _cycling_penalty: float = 2.0
    _stored_energy_value: float = 0.0
    _reserve_config: dict | None = None
    _n_da_bands: int = 1
    _da_band_prices: list = []
    _da_clearing_probs: list = []
    _reserve_acceptance_probs: dict = {}
    _reserve_offer_prices: dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cycling_penalty_factor = self.__class__._cycling_penalty
        self.stored_energy_value = self.__class__._stored_energy_value
        if self.__class__._reserve_config is not None:
            # Deep-copy so concurrent requests can't mutate each other's state
            self.reserve_config = {
                k: dict(v) for k, v in self.__class__._reserve_config.items()
            }
        self.n_da_bands = self.__class__._n_da_bands
        self.da_band_prices = list(self.__class__._da_band_prices)
        self.da_clearing_probs = list(self.__class__._da_clearing_probs)
        self.reserve_acceptance_probs = {
            k: list(v) for k, v in self.__class__._reserve_acceptance_probs.items()
        }
        self.reserve_offer_prices = {
            k: list(v) for k, v in self.__class__._reserve_offer_prices.items()
        }

    def post(self):
        # Skip the demo's print statements — we read CSV output directly
        # Call the grandparent's post() to ensure CSV export happens
        super(BESS, self).post()
