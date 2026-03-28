"""Core computation logic for the dc_impact environmental footprint calculator.

This module exposes a single high-level entry point, ``calculate_impact``,
as well as several lower-level helpers that can be used and tested
independently. The calculation pipeline is:

1. Look up hardware TDP (W) from :mod:`dc_impact.data`.
2. Multiply by number of accelerators and utilization to get raw power draw.
3. Apply the data-centre PUE for the given region to get total facility power.
4. Multiply by duration (hours) to get energy consumption (kWh).
5. Multiply energy by the regional carbon intensity (gCO2eq/kWh) to get
   CO2e emissions (kg).
6. Multiply energy by the regional WUE (L/kWh) to get water usage (litres).

All lookup keys follow the conventions defined in :mod:`dc_impact.data`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

# Placeholder imports — data module will be fully populated in phase 2.
from dc_impact import data as _data

# Grams per kilogram conversion factor.
_G_PER_KG: Final[float] = 1_000.0

# Watts per kilowatt conversion factor.
_W_PER_KW: Final[float] = 1_000.0


class CalculatorError(ValueError):
    """Raised when input parameters are invalid or a lookup key is unknown."""


@dataclass(frozen=True)
class ImpactResult:
    """Immutable result of an environmental impact calculation.

    Attributes:
        energy_kwh: Total energy consumed by the workload (kWh), including
            the data-centre PUE overhead.
        co2e_kg: Carbon dioxide equivalent emissions (kg CO2e).
        water_liters: Estimated water consumption (litres).
        hardware: Hardware preset key used for the calculation.
        region: Cloud region key used for the calculation.
        duration_hours: Wall-clock duration of the workload (hours).
        num_accelerators: Number of accelerator units involved.
        utilization: Fractional GPU/TPU utilisation (0–1).
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
            Dictionary mapping field names to their values.
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

    Args:
        hardware: Hardware preset key (e.g. ``"A100_80GB"``, ``"H100"``).
            Must be a key in :data:`dc_impact.data.HARDWARE_TDP_WATTS`.
        region: Cloud region key (e.g. ``"us-east-1"``, ``"europe-west4"``).
            Must be a key in :data:`dc_impact.data.CARBON_INTENSITY_G_PER_KWH`.
        duration_hours: Positive wall-clock duration of the workload in hours.
        num_accelerators: Number of accelerator devices used (must be >= 1).
        workload_type: Type of workload — either ``"training"`` or
            ``"inference"``.
        utilization: Fractional device utilisation in the range (0, 1].
            Defaults to ``1.0`` (100 %).

    Returns:
        An :class:`ImpactResult` containing energy, CO2e, and water figures.

    Raises:
        CalculatorError: If any parameter is out of range or an unrecognised
            lookup key is provided.
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


def compute_energy_kwh(
    tdp_watts: float,
    num_accelerators: int,
    utilization: float,
    duration_hours: float,
    pue: float,
) -> float:
    """Calculate PUE-adjusted energy consumption in kilowatt-hours.

    The formula is::

        energy_kwh = (tdp_watts * num_accelerators * utilization / 1000)
                     * duration_hours * pue

    Args:
        tdp_watts: Thermal design power of a single accelerator (W).
        num_accelerators: Number of accelerator devices.
        utilization: Fractional device utilisation (0, 1].
        duration_hours: Duration of the workload (hours).
        pue: Power Usage Effectiveness of the data centre (>= 1.0).

    Returns:
        Total facility energy in kWh.

    Raises:
        CalculatorError: If any parameter is out of its valid range.
    """
    if tdp_watts <= 0:
        raise CalculatorError(f"tdp_watts must be positive, got {tdp_watts}.")
    if num_accelerators < 1:
        raise CalculatorError(
            f"num_accelerators must be >= 1, got {num_accelerators}."
        )
    if not (0.0 < utilization <= 1.0):
        raise CalculatorError(
            f"utilization must be in (0, 1], got {utilization}."
        )
    if duration_hours <= 0:
        raise CalculatorError(
            f"duration_hours must be positive, got {duration_hours}."
        )
    if pue < 1.0:
        raise CalculatorError(f"pue must be >= 1.0, got {pue}.")

    raw_power_kw = (tdp_watts * num_accelerators * utilization) / _W_PER_KW
    return raw_power_kw * duration_hours * pue


def compute_co2e_kg(
    energy_kwh: float,
    carbon_intensity_g_per_kwh: float,
) -> float:
    """Convert energy consumption to CO2-equivalent emissions in kilograms.

    Args:
        energy_kwh: Energy consumed (kWh).
        carbon_intensity_g_per_kwh: Grid carbon intensity (gCO2e/kWh).

    Returns:
        CO2e emissions in kilograms.

    Raises:
        CalculatorError: If either argument is negative.
    """
    if energy_kwh < 0:
        raise CalculatorError(f"energy_kwh must be >= 0, got {energy_kwh}.")
    if carbon_intensity_g_per_kwh < 0:
        raise CalculatorError(
            f"carbon_intensity_g_per_kwh must be >= 0, "
            f"got {carbon_intensity_g_per_kwh}."
        )
    return (energy_kwh * carbon_intensity_g_per_kwh) / _G_PER_KG


def compute_water_liters(
    energy_kwh: float,
    wue: float,
) -> float:
    """Estimate water consumption in litres.

    Uses the Water Usage Effectiveness (WUE) metric::

        water_liters = energy_kwh * wue

    Args:
        energy_kwh: Energy consumed (kWh).
        wue: Water Usage Effectiveness of the data centre (L/kWh).

    Returns:
        Estimated water consumption in litres.

    Raises:
        CalculatorError: If either argument is negative.
    """
    if energy_kwh < 0:
        raise CalculatorError(f"energy_kwh must be >= 0, got {energy_kwh}.")
    if wue < 0:
        raise CalculatorError(f"wue must be >= 0, got {wue}.")
    return energy_kwh * wue


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

    Args:
        hardware: Hardware preset key.
        region: Cloud region key.
        duration_hours: Workload wall-clock duration (hours).
        num_accelerators: Number of accelerator devices.
        workload_type: ``"training"`` or ``"inference"``.
        utilization: Fractional device utilisation.

    Raises:
        CalculatorError: On any invalid parameter.
    """
    if not isinstance(hardware, str) or not hardware.strip():
        raise CalculatorError("hardware must be a non-empty string.")
    if not isinstance(region, str) or not region.strip():
        raise CalculatorError("region must be a non-empty string.")
    if not isinstance(duration_hours, (int, float)) or duration_hours <= 0:
        raise CalculatorError(
            f"duration_hours must be a positive number, got {duration_hours!r}."
        )
    if not isinstance(num_accelerators, int) or num_accelerators < 1:
        raise CalculatorError(
            f"num_accelerators must be a positive integer, "
            f"got {num_accelerators!r}."
        )
    if workload_type not in {"training", "inference"}:
        raise CalculatorError(
            f"workload_type must be 'training' or 'inference', "
            f"got {workload_type!r}."
        )
    if not isinstance(utilization, (int, float)) or not (0.0 < utilization <= 1.0):
        raise CalculatorError(
            f"utilization must be a number in (0, 1], got {utilization!r}."
        )


def _lookup_hardware_tdp(hardware: str) -> float:
    """Look up the TDP (W) for a hardware preset.

    Args:
        hardware: Hardware preset key.

    Returns:
        TDP in watts.

    Raises:
        CalculatorError: If the key is not found in the lookup table.
    """
    table: dict[str, float] = _data.HARDWARE_TDP_WATTS
    if hardware not in table:
        available = ", ".join(sorted(table.keys()))
        raise CalculatorError(
            f"Unknown hardware preset {hardware!r}. "
            f"Available options: {available}."
        )
    return table[hardware]


def _lookup_carbon_intensity(region: str) -> float:
    """Look up the grid carbon intensity (gCO2e/kWh) for a region.

    Args:
        region: Cloud region key.

    Returns:
        Carbon intensity in gCO2e/kWh.

    Raises:
        CalculatorError: If the key is not found in the lookup table.
    """
    table: dict[str, float] = _data.CARBON_INTENSITY_G_PER_KWH
    if region not in table:
        available = ", ".join(sorted(table.keys()))
        raise CalculatorError(
            f"Unknown region {region!r}. Available options: {available}."
        )
    return table[region]


def _lookup_pue(region: str) -> float:
    """Look up the Power Usage Effectiveness for a region.

    Falls back to a global average of 1.58 if the region is not explicitly
    listed in the PUE table.

    Args:
        region: Cloud region key.

    Returns:
        PUE value (dimensionless, >= 1.0).
    """
    return _data.PUE_BY_REGION.get(region, _data.DEFAULT_PUE)


def _lookup_wue(region: str) -> float:
    """Look up the Water Usage Effectiveness (L/kWh) for a region.

    Falls back to a global average if the region is not explicitly listed.

    Args:
        region: Cloud region key.

    Returns:
        WUE value in litres per kWh.
    """
    return _data.WUE_BY_REGION.get(region, _data.DEFAULT_WUE)
