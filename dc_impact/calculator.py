"""Core computation logic for the dc_impact environmental footprint calculator.

This module exposes a single high-level entry point, ``calculate_impact``,
as well as several lower-level helpers that can be used and tested
independently. The calculation pipeline is:

1. Validate all input parameters.
2. Look up hardware TDP (W) from :mod:`dc_impact.data`.
3. Multiply by number of accelerators and utilization to get raw power draw.
4. Apply the data-centre PUE for the given region to get total facility power.
5. Multiply by duration (hours) to get energy consumption (kWh).
6. Multiply energy by the regional carbon intensity (gCO2eq/kWh) to get
   CO2e emissions (kg).
7. Multiply energy by the regional WUE (L/kWh) to get water usage (litres).

All lookup keys follow the conventions defined in :mod:`dc_impact.data`.

Typical usage::

    from dc_impact.calculator import calculate_impact

    result = calculate_impact(
        hardware="A100_80GB",
        region="us-east-1",
        duration_hours=72.0,
        num_accelerators=8,
        workload_type="training",
        utilization=1.0,
    )
    print(result.co2e_kg)   # kg CO2e
    print(result.energy_kwh)  # kWh
    print(result.water_liters)  # litres
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from dc_impact import data as _data

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Grams per kilogram — used to convert gCO2e to kgCO2e.
_G_PER_KG: Final[float] = 1_000.0

#: Watts per kilowatt — used to convert W to kW.
_W_PER_KW: Final[float] = 1_000.0

#: Recognised workload types.
_VALID_WORKLOAD_TYPES: Final[frozenset[str]] = frozenset({"training", "inference"})


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class CalculatorError(ValueError):
    """Raised when input parameters are invalid or a lookup key is unknown.

    Inherits from :class:`ValueError` so callers can catch it with a broad
    ``except ValueError`` if they prefer not to import this specific class.
    """


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpactResult:
    """Immutable result of an environmental impact calculation.

    All numeric fields are rounded to six decimal places by
    :func:`calculate_impact` before being stored here.

    Attributes:
        energy_kwh: Total energy consumed by the workload (kWh), including
            the data-centre PUE overhead.
        co2e_kg: Carbon dioxide equivalent emissions (kg CO2e).
        water_liters: Estimated water consumption (litres).
        hardware: Hardware preset key used for the calculation.
        region: Cloud region key used for the calculation.
        duration_hours: Wall-clock duration of the workload (hours).
        num_accelerators: Number of accelerator units involved.
        utilization: Fractional GPU/TPU utilisation (0–1].
        workload_type: One of ``"training"`` or ``"inference"``.
        pue: Power Usage Effectiveness applied.
        wue: Water Usage Effectiveness applied (L/kWh).
        carbon_intensity_g_per_kwh: Grid carbon intensity used (gCO2e/kWh).
        tdp_watts: Hardware TDP (W) used in the calculation.
    """

    energy_kwh: float
    co2e_kg: float
    water_liters: float
    hardware: str
    region: str
    duration_hours: float
    num_accelerators: int
    utilization: float
    workload_type: str
    pue: float
    wue: float
    carbon_intensity_g_per_kwh: float
    tdp_watts: float

    def to_dict(self) -> dict[str, object]:
        """Serialize the result to a plain dictionary.

        Returns:
            Dictionary mapping field names to their values. All values are
            JSON-serialisable primitives (str, int, float).
        """
        return {
            "energy_kwh": self.energy_kwh,
            "co2e_kg": self.co2e_kg,
            "water_liters": self.water_liters,
            "hardware": self.hardware,
            "region": self.region,
            "duration_hours": self.duration_hours,
            "num_accelerators": self.num_accelerators,
            "utilization": self.utilization,
            "workload_type": self.workload_type,
            "pue": self.pue,
            "wue": self.wue,
            "carbon_intensity_g_per_kwh": self.carbon_intensity_g_per_kwh,
            "tdp_watts": self.tdp_watts,
        }


# ---------------------------------------------------------------------------
# Primary public API
# ---------------------------------------------------------------------------


def calculate_impact(
    hardware: str,
    region: str,
    duration_hours: float,
    num_accelerators: int = 1,
    workload_type: str = "training",
    utilization: float = 1.0,
) -> ImpactResult:
    """Compute the full environmental impact for a given AI workload.

    This is the primary public API for the calculation engine. It validates
    all parameters, performs lookups in the static data tables, and returns
    a fully populated :class:`ImpactResult`.

    The calculation pipeline:

    1. Validate all inputs via :func:`_validate_inputs`.
    2. Look up TDP, carbon intensity, PUE, and WUE from :mod:`dc_impact.data`.
    3. Compute PUE-adjusted energy in kWh via :func:`compute_energy_kwh`.
    4. Compute CO2e emissions in kg via :func:`compute_co2e_kg`.
    5. Compute water consumption in litres via :func:`compute_water_liters`.
    6. Package everything into a frozen :class:`ImpactResult`.

    Args:
        hardware: Hardware preset key (e.g. ``"A100_80GB"``, ``"H100_SXM"``).
            Must be a key in :data:`dc_impact.data.HARDWARE_TDP_WATTS`.
        region: Cloud region key (e.g. ``"us-east-1"``, ``"europe-west4"``).
            Must be a key in :data:`dc_impact.data.CARBON_INTENSITY_G_PER_KWH`.
        duration_hours: Positive wall-clock duration of the workload in hours.
        num_accelerators: Number of accelerator devices used (must be >= 1).
            Defaults to ``1``.
        workload_type: Type of workload — either ``"training"`` or
            ``"inference"``. Defaults to ``"training"``.
        utilization: Fractional device utilisation in the range ``(0, 1]``.
            A value of ``1.0`` means the accelerator runs at full TDP.
            Defaults to ``1.0``.

    Returns:
        A fully populated :class:`ImpactResult` containing energy (kWh),
        CO2e (kg), and water (litres) figures together with all input
        parameters and lookup factors used.

    Raises:
        CalculatorError: If any parameter is out of range or an unrecognised
            lookup key is provided.

    Examples:
        Calculate the footprint of training on 8× A100s for 72 hours in
        AWS Oregon (low-carbon grid)::

            result = calculate_impact(
                hardware="A100_80GB",
                region="us-west-2",
                duration_hours=72.0,
                num_accelerators=8,
                workload_type="training",
                utilization=1.0,
            )
            # energy = 8 * 400 W * 1.0 * 72 h * 1.20 PUE / 1000 = 276.48 kWh
            # co2e  = 276.48 * 136 / 1000 = 37.601 kg
            # water = 276.48 * 0.18       = 49.766 L
    """
    _validate_inputs(
        hardware=hardware,
        region=region,
        duration_hours=duration_hours,
        num_accelerators=num_accelerators,
        workload_type=workload_type,
        utilization=utilization,
    )

    tdp_watts = _lookup_hardware_tdp(hardware)
    carbon_intensity = _lookup_carbon_intensity(region)
    pue = _lookup_pue(region)
    wue = _lookup_wue(region)

    energy_kwh = compute_energy_kwh(
        tdp_watts=tdp_watts,
        num_accelerators=num_accelerators,
        utilization=utilization,
        duration_hours=duration_hours,
        pue=pue,
    )
    co2e_kg = compute_co2e_kg(
        energy_kwh=energy_kwh,
        carbon_intensity_g_per_kwh=carbon_intensity,
    )
    water_liters = compute_water_liters(
        energy_kwh=energy_kwh,
        wue=wue,
    )

    return ImpactResult(
        energy_kwh=round(energy_kwh, 6),
        co2e_kg=round(co2e_kg, 6),
        water_liters=round(water_liters, 6),
        hardware=hardware,
        region=region,
        duration_hours=duration_hours,
        num_accelerators=num_accelerators,
        utilization=utilization,
        workload_type=workload_type,
        pue=pue,
        wue=wue,
        carbon_intensity_g_per_kwh=carbon_intensity,
        tdp_watts=tdp_watts,
    )


# ---------------------------------------------------------------------------
# Lower-level computation helpers (public, independently testable)
# ---------------------------------------------------------------------------


def compute_energy_kwh(
    tdp_watts: float,
    num_accelerators: int,
    utilization: float,
    duration_hours: float,
    pue: float,
) -> float:
    """Calculate PUE-adjusted energy consumption in kilowatt-hours.

    The formula is::

        raw_power_kw = (tdp_watts * num_accelerators * utilization) / 1000
        energy_kwh   = raw_power_kw * duration_hours * pue

    Args:
        tdp_watts: Thermal design power of a single accelerator in watts.
            Must be strictly positive.
        num_accelerators: Number of accelerator devices (>= 1).
        utilization: Fractional device utilisation in the range ``(0, 1]``.
        duration_hours: Duration of the workload in hours. Must be positive.
        pue: Power Usage Effectiveness of the data centre (>= 1.0). A PUE
            of 1.0 means no overhead; 1.2 means 20 % extra power for cooling
            and other facility loads.

    Returns:
        Total facility energy consumed in kWh.

    Raises:
        CalculatorError: If any argument is outside its valid range.

    Examples:
        Single A100 at full utilisation for 1 hour with PUE 1.2::

            energy = compute_energy_kwh(
                tdp_watts=400.0,
                num_accelerators=1,
                utilization=1.0,
                duration_hours=1.0,
                pue=1.2,
            )
            # 400 / 1000 * 1.0 * 1.2 = 0.48 kWh
    """
    if not isinstance(tdp_watts, (int, float)) or tdp_watts <= 0:
        raise CalculatorError(
            f"tdp_watts must be a positive number, got {tdp_watts!r}."
        )
    if not isinstance(num_accelerators, int) or num_accelerators < 1:
        raise CalculatorError(
            f"num_accelerators must be a positive integer, got {num_accelerators!r}."
        )
    if not isinstance(utilization, (int, float)) or not (0.0 < utilization <= 1.0):
        raise CalculatorError(
            f"utilization must be a number in (0, 1], got {utilization!r}."
        )
    if not isinstance(duration_hours, (int, float)) or duration_hours <= 0:
        raise CalculatorError(
            f"duration_hours must be a positive number, got {duration_hours!r}."
        )
    if not isinstance(pue, (int, float)) or pue < 1.0:
        raise CalculatorError(
            f"pue must be a number >= 1.0, got {pue!r}."
        )

    raw_power_kw = (tdp_watts * num_accelerators * utilization) / _W_PER_KW
    return raw_power_kw * duration_hours * pue


def compute_co2e_kg(
    energy_kwh: float,
    carbon_intensity_g_per_kwh: float,
) -> float:
    """Convert energy consumption to CO2-equivalent emissions in kilograms.

    The formula is::

        co2e_kg = (energy_kwh * carbon_intensity_g_per_kwh) / 1000

    Args:
        energy_kwh: Energy consumed in kilowatt-hours. Must be >= 0.
        carbon_intensity_g_per_kwh: Grid carbon intensity in gCO2e per kWh.
            Must be >= 0.

    Returns:
        CO2-equivalent emissions in kilograms.

    Raises:
        CalculatorError: If either argument is negative.

    Examples:
        100 kWh consumed from a grid with 415 gCO2e/kWh::

            co2e = compute_co2e_kg(100.0, 415.0)
            # 100 * 415 / 1000 = 41.5 kg CO2e
    """
    if not isinstance(energy_kwh, (int, float)) or energy_kwh < 0:
        raise CalculatorError(
            f"energy_kwh must be a non-negative number, got {energy_kwh!r}."
        )
    if (
        not isinstance(carbon_intensity_g_per_kwh, (int, float))
        or carbon_intensity_g_per_kwh < 0
    ):
        raise CalculatorError(
            f"carbon_intensity_g_per_kwh must be a non-negative number, "
            f"got {carbon_intensity_g_per_kwh!r}."
        )
    return (energy_kwh * carbon_intensity_g_per_kwh) / _G_PER_KG


def compute_water_liters(
    energy_kwh: float,
    wue: float,
) -> float:
    """Estimate onsite water consumption in litres.

    Uses the Water Usage Effectiveness (WUE) metric::

        water_liters = energy_kwh * wue

    A WUE of 0 indicates a waterless cooling system (e.g. 100 % air-side
    economisation); most data centres fall between 0.1 and 2.5 L/kWh.

    Args:
        energy_kwh: Energy consumed in kilowatt-hours. Must be >= 0.
        wue: Water Usage Effectiveness of the data centre in litres per kWh.
            Must be >= 0.

    Returns:
        Estimated water consumption in litres.

    Raises:
        CalculatorError: If either argument is negative.

    Examples:
        276.48 kWh consumed at a facility with WUE = 0.18 L/kWh::

            water = compute_water_liters(276.48, 0.18)
            # 276.48 * 0.18 = 49.7664 litres
    """
    if not isinstance(energy_kwh, (int, float)) or energy_kwh < 0:
        raise CalculatorError(
            f"energy_kwh must be a non-negative number, got {energy_kwh!r}."
        )
    if not isinstance(wue, (int, float)) or wue < 0:
        raise CalculatorError(
            f"wue must be a non-negative number, got {wue!r}."
        )
    return energy_kwh * wue


def get_available_hardware() -> list[str]:
    """Return a sorted list of all supported hardware preset keys.

    Returns:
        Sorted list of hardware preset keys from
        :data:`dc_impact.data.HARDWARE_TDP_WATTS`.
    """
    return sorted(_data.HARDWARE_TDP_WATTS.keys())


def get_available_regions() -> list[str]:
    """Return a sorted list of all supported cloud region keys.

    Returns:
        Sorted list of region keys from
        :data:`dc_impact.data.CARBON_INTENSITY_G_PER_KWH`.
    """
    return sorted(_data.CARBON_INTENSITY_G_PER_KWH.keys())


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    *,
    hardware: str,
    region: str,
    duration_hours: float,
    num_accelerators: int,
    workload_type: str,
    utilization: float,
) -> None:
    """Validate all high-level inputs, raising :class:`CalculatorError` on
    the first detected problem.

    This function is intentionally strict: it checks both type and value
    constraints so that downstream lookup and computation functions receive
    clean inputs.

    Args:
        hardware: Hardware preset key — must be a non-empty string.
        region: Cloud region key — must be a non-empty string.
        duration_hours: Workload wall-clock duration (hours) — must be a
            positive finite number.
        num_accelerators: Number of accelerator devices — must be a positive
            integer (>= 1).
        workload_type: Must be exactly ``"training"`` or ``"inference"``.
        utilization: Fractional device utilisation — must be a number in
            the range ``(0, 1]``.

    Raises:
        CalculatorError: On any invalid parameter, with a descriptive message
            indicating which parameter failed and what was received.
    """
    # --- hardware ---
    if not isinstance(hardware, str) or not hardware.strip():
        raise CalculatorError(
            f"hardware must be a non-empty string, got {hardware!r}."
        )

    # --- region ---
    if not isinstance(region, str) or not region.strip():
        raise CalculatorError(
            f"region must be a non-empty string, got {region!r}."
        )

    # --- duration_hours ---
    if not isinstance(duration_hours, (int, float)) or isinstance(duration_hours, bool):
        raise CalculatorError(
            f"duration_hours must be a positive number, got {duration_hours!r}."
        )
    if duration_hours <= 0:
        raise CalculatorError(
            f"duration_hours must be positive, got {duration_hours}."
        )

    # --- num_accelerators ---
    if isinstance(num_accelerators, bool) or not isinstance(num_accelerators, int):
        raise CalculatorError(
            f"num_accelerators must be a positive integer, "
            f"got {num_accelerators!r}."
        )
    if num_accelerators < 1:
        raise CalculatorError(
            f"num_accelerators must be >= 1, got {num_accelerators}."
        )

    # --- workload_type ---
    if workload_type not in _VALID_WORKLOAD_TYPES:
        valid = "', '".join(sorted(_VALID_WORKLOAD_TYPES))
        raise CalculatorError(
            f"workload_type must be one of '{valid}', got {workload_type!r}."
        )

    # --- utilization ---
    if isinstance(utilization, bool) or not isinstance(utilization, (int, float)):
        raise CalculatorError(
            f"utilization must be a number in (0, 1], got {utilization!r}."
        )
    if not (0.0 < utilization <= 1.0):
        raise CalculatorError(
            f"utilization must be in the range (0, 1], got {utilization}."
        )


def _lookup_hardware_tdp(hardware: str) -> float:
    """Look up the TDP (W) for a hardware preset key.

    Args:
        hardware: Hardware preset key (e.g. ``"A100_80GB"``).

    Returns:
        TDP in watts as a float.

    Raises:
        CalculatorError: If *hardware* is not found in
            :data:`dc_impact.data.HARDWARE_TDP_WATTS`, with a message
            listing all available options.
    """
    table: dict[str, float] = _data.HARDWARE_TDP_WATTS
    if hardware not in table:
        available = ", ".join(sorted(table.keys()))
        raise CalculatorError(
            f"Unknown hardware preset {hardware!r}. "
            f"Available options: {available}."
        )
    return float(table[hardware])


def _lookup_carbon_intensity(region: str) -> float:
    """Look up the grid carbon intensity (gCO2e/kWh) for a cloud region.

    Args:
        region: Cloud region key (e.g. ``"us-east-1"``).

    Returns:
        Carbon intensity in gCO2e/kWh as a float.

    Raises:
        CalculatorError: If *region* is not found in
            :data:`dc_impact.data.CARBON_INTENSITY_G_PER_KWH`, with a
            message listing all available options.
    """
    table: dict[str, float] = _data.CARBON_INTENSITY_G_PER_KWH
    if region not in table:
        available = ", ".join(sorted(table.keys()))
        raise CalculatorError(
            f"Unknown region {region!r}. Available options: {available}."
        )
    return float(table[region])


def _lookup_pue(region: str) -> float:
    """Look up the Power Usage Effectiveness for a cloud region.

    Falls back to :data:`dc_impact.data.DEFAULT_PUE` (the global industry
    average of 1.58) if *region* is not explicitly listed in
    :data:`dc_impact.data.PUE_BY_REGION`.

    Args:
        region: Cloud region key.

    Returns:
        PUE value as a float (dimensionless, >= 1.0).
    """
    return float(_data.PUE_BY_REGION.get(region, _data.DEFAULT_PUE))


def _lookup_wue(region: str) -> float:
    """Look up the Water Usage Effectiveness (L/kWh) for a cloud region.

    Falls back to :data:`dc_impact.data.DEFAULT_WUE` (the global industry
    average) if *region* is not explicitly listed in
    :data:`dc_impact.data.WUE_BY_REGION`.

    Args:
        region: Cloud region key.

    Returns:
        WUE value in litres per kWh as a float.
    """
    return float(_data.WUE_BY_REGION.get(region, _data.DEFAULT_WUE))
