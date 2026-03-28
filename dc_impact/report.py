"""Report generation for dc_impact.

This module exposes two public functions:

* :func:`generate_json_report` — converts an
  :class:`~dc_impact.calculator.ImpactResult` into a structured,
  serialisable dictionary suitable for use as a JSON API response or for
  embedding in model cards.
* :func:`generate_html_report` — renders the same result as an embeddable
  HTML snippet using Jinja2 and the ``report_card.html`` template.

The JSON report schema (version ``"1.0"``) has the following top-level keys:

:``schema_version``: Report format version string (``"1.0"``).
:``generated_at``:   ISO 8601 UTC timestamp at generation time.
:``inputs``:         Input parameters supplied by the caller.
:``metrics``:        Computed environmental impact metrics.
:``factors``:        Data-centre and grid factors applied during computation.
:``comparisons``:    Human-readable equivalence comparisons for the metrics.

Typical usage::

    from dc_impact.calculator import calculate_impact
    from dc_impact.report import generate_json_report, generate_html_report

    result = calculate_impact(
        hardware="A100_80GB",
        region="us-east-1",
        duration_hours=72.0,
        num_accelerators=8,
    )

    report_dict = generate_json_report(result)
    html_snippet = generate_html_report(result)
"""

from __future__ import annotations

import datetime
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from dc_impact.calculator import ImpactResult

# ---------------------------------------------------------------------------
# Report schema version — bump this when the JSON structure changes.
# ---------------------------------------------------------------------------
_SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Equivalence comparison constants
# These help readers contextualise abstract numbers.
# ---------------------------------------------------------------------------

#: Average CO2e emitted per km driven in a petrol/gasoline passenger car
#: (source: UK DEFRA 2023 — ~0.170 kg CO2e/km).
_CO2_KG_PER_CAR_KM: float = 0.170

#: Average CO2e emitted per tree sequestered per year
#: (source: US EPA — ~21.77 kg CO2/tree/year).
_CO2_KG_PER_TREE_YEAR: float = 21.77

#: Average kWh consumed by a US household per day
#: (source: US EIA 2022 — 10,500 kWh/year ≈ 28.77 kWh/day).
_KWH_PER_US_HOUSEHOLD_DAY: float = 28.77

#: Average litres of water used per shower (8-minute shower, 8 L/min).
_LITERS_PER_SHOWER: float = 64.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_json_report(result: ImpactResult) -> dict[str, Any]:
    """Build a structured JSON-serialisable report dictionary.

    The returned mapping includes all computed metrics, contextual metadata
    (hardware, region, workload type, etc.), the lookup factors applied, and
    a set of human-readable equivalence comparisons that contextualise the
    raw numbers for readers unfamiliar with energy/carbon units.

    Args:
        result: A fully populated :class:`~dc_impact.calculator.ImpactResult`
            returned by :func:`~dc_impact.calculator.calculate_impact`.

    Returns:
        A plain dictionary with the following top-level keys:

        - ``"schema_version"`` (str): Report schema version (``"1.0"``).
        - ``"generated_at"`` (str): ISO 8601 UTC timestamp.
        - ``"inputs"`` (dict): Input parameters used for the calculation.
        - ``"metrics"`` (dict): Computed impact metrics (energy, CO2e, water).
        - ``"factors"`` (dict): Lookup factors applied (TDP, PUE, WUE, CI).
        - ``"comparisons"`` (dict): Human-readable equivalence figures.

    Example::

        report = generate_json_report(result)
        # report["metrics"]["co2e_kg"] -> float
        # report["comparisons"]["car_km_equivalent"] -> float
    """
    return {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "inputs": _build_inputs_section(result),
        "metrics": _build_metrics_section(result),
        "factors": _build_factors_section(result),
        "comparisons": _build_comparisons_section(result),
    }


def generate_html_report(result: ImpactResult) -> str:
    """Render an embeddable HTML report card snippet.

    Uses Jinja2 with the package's ``templates/report_card.html`` template
    to produce a self-contained HTML ``<div>`` fragment suitable for inclusion
    in GitHub READMEs (as raw HTML blocks) or Hugging Face model cards.

    The template receives two context variables:

    * ``result`` — the :class:`~dc_impact.calculator.ImpactResult` instance.
    * ``report`` — the full JSON report dictionary produced by
      :func:`generate_json_report`, giving templates access to formatted
      comparisons and metadata.

    Args:
        result: A fully populated :class:`~dc_impact.calculator.ImpactResult`
            returned by :func:`~dc_impact.calculator.calculate_impact`.

    Returns:
        Rendered HTML string — a ``<div class="report-card">`` fragment,
        **not** a full ``<!DOCTYPE html>`` document.

    Raises:
        jinja2.TemplateNotFound: If ``report_card.html`` cannot be located
            within the package template directory.
    """
    env = _get_jinja_env()
    template = env.get_template("report_card.html")
    report_data = generate_json_report(result)
    return template.render(result=result, report=report_data)


# ---------------------------------------------------------------------------
# Private helpers — report section builders
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string with a 'Z' suffix.

    Returns:
        Timestamp string, e.g. ``"2024-01-15T12:34:56.789012Z"``.
    """
    return datetime.datetime.utcnow().isoformat() + "Z"


def _build_inputs_section(result: ImpactResult) -> dict[str, Any]:
    """Build the ``inputs`` section of the JSON report.

    Args:
        result: The calculation result to extract inputs from.

    Returns:
        Dictionary of input parameters with descriptive keys.
    """
    return {
        "hardware": result.hardware,
        "region": result.region,
        "duration_hours": result.duration_hours,
        "num_accelerators": result.num_accelerators,
        "workload_type": result.workload_type,
        "utilization": result.utilization,
        "utilization_pct": round(result.utilization * 100, 1),
    }


def _build_metrics_section(result: ImpactResult) -> dict[str, Any]:
    """Build the ``metrics`` section of the JSON report.

    Includes the three primary environmental metrics plus convenient unit
    conversions (e.g. grams alongside kg for small values, MWh alongside
    kWh for large values).

    Args:
        result: The calculation result to extract metrics from.

    Returns:
        Dictionary of computed metrics with multiple unit representations.
    """
    energy_kwh = result.energy_kwh
    co2e_kg = result.co2e_kg
    water_liters = result.water_liters

    return {
        "energy_kwh": energy_kwh,
        "energy_mwh": round(energy_kwh / 1_000, 6),
        "co2e_kg": co2e_kg,
        "co2e_g": round(co2e_kg * 1_000, 3),
        "co2e_tonnes": round(co2e_kg / 1_000, 6),
        "water_liters": water_liters,
        "water_ml": round(water_liters * 1_000, 3),
    }


def _build_factors_section(result: ImpactResult) -> dict[str, Any]:
    """Build the ``factors`` section of the JSON report.

    Contains the lookup values applied during the calculation so that
    consumers can reproduce the result independently.

    Args:
        result: The calculation result to extract factors from.

    Returns:
        Dictionary of data-centre and grid factors with descriptive keys.
    """
    return {
        "tdp_watts": result.tdp_watts,
        "pue": result.pue,
        "wue_l_per_kwh": result.wue,
        "carbon_intensity_g_per_kwh": result.carbon_intensity_g_per_kwh,
        # Derived factor: effective power per accelerator including PUE overhead
        "effective_power_per_accelerator_w": round(
            result.tdp_watts * result.utilization * result.pue, 3
        ),
    }


def _build_comparisons_section(result: ImpactResult) -> dict[str, Any]:
    """Build the ``comparisons`` section with human-readable equivalences.

    Converts abstract energy/carbon/water numbers into relatable analogies
    that help non-technical readers understand the scale of the impact.

    Args:
        result: The calculation result to build comparisons from.

    Returns:
        Dictionary with human-readable equivalence values and descriptions.

    Note:
        All comparison constants are rough approximations and are labelled
        as such in the returned dictionary.
    """
    co2e_kg = result.co2e_kg
    energy_kwh = result.energy_kwh
    water_liters = result.water_liters

    # CO2e comparisons
    car_km = round(co2e_kg / _CO2_KG_PER_CAR_KM, 2) if co2e_kg > 0 else 0.0
    tree_months = round(
        (co2e_kg / _CO2_KG_PER_TREE_YEAR) * 12, 2
    ) if co2e_kg > 0 else 0.0

    # Energy comparisons
    household_days = round(
        energy_kwh / _KWH_PER_US_HOUSEHOLD_DAY, 2
    ) if energy_kwh > 0 else 0.0

    # Water comparisons
    showers = round(
        water_liters / _LITERS_PER_SHOWER, 2
    ) if water_liters > 0 else 0.0

    return {
        "co2e": {
            "car_km_equivalent": car_km,
            "car_km_equivalent_description": (
                f"Equivalent to driving a petrol car approx. "
                f"{car_km:,.1f} km"
            ),
            "tree_sequestration_months": tree_months,
            "tree_sequestration_months_description": (
                f"Would take one tree approx. "
                f"{tree_months:,.1f} months to sequester"
            ),
        },
        "energy": {
            "us_household_days_equivalent": household_days,
            "us_household_days_equivalent_description": (
                f"Equivalent to approx. {household_days:,.1f} days of average "
                f"US household electricity consumption"
            ),
        },
        "water": {
            "showers_equivalent": showers,
            "showers_equivalent_description": (
                f"Equivalent to approx. {showers:,.1f} eight-minute showers"
            ),
        },
        "disclaimer": (
            "Comparisons are approximate and intended for illustrative "
            "purposes only. See report factors for calculation methodology."
        ),
    }


# ---------------------------------------------------------------------------
# Private helper — Jinja2 environment
# ---------------------------------------------------------------------------


def _get_jinja_env() -> Environment:
    """Construct a Jinja2 :class:`~jinja2.Environment` configured to load
    templates from the ``dc_impact`` package's ``templates`` directory.

    The environment is configured with:

    * ``PackageLoader`` pointing at ``dc_impact/templates/``.
    * HTML autoescaping enabled for ``.html`` files (XSS prevention).
    * ``trim_blocks=True`` and ``lstrip_blocks=True`` for cleaner output.
    * Custom filters: ``format_number``, ``co2_badge_colour``.

    Returns:
        A configured :class:`~jinja2.Environment` instance ready for
        ``get_template()`` calls.
    """
    env = Environment(
        loader=PackageLoader("dc_impact", "templates"),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Register custom Jinja2 filters
    env.filters["format_number"] = _filter_format_number
    env.filters["co2_badge_colour"] = _filter_co2_badge_colour
    env.filters["energy_badge_colour"] = _filter_energy_badge_colour

    return env


# ---------------------------------------------------------------------------
# Custom Jinja2 filters
# ---------------------------------------------------------------------------


def _filter_format_number(value: float, precision: int = 3) -> str:
    """Jinja2 filter: format a float with adaptive precision.

    For values >= 1 the number is rounded to *precision* decimal places.
    For very small values (< 0.001) scientific notation is used.

    Args:
        value: The numeric value to format.
        precision: Number of decimal places to show for normal values.

    Returns:
        Formatted string representation.

    Example::

        {{ result.co2e_kg | format_number(3) }}
    """
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return str(value)

    if fval == 0.0:
        return "0"
    if abs(fval) < 0.001:
        return f"{fval:.2e}"
    if abs(fval) >= 1_000_000:
        return f"{fval / 1_000_000:.3f}M"
    if abs(fval) >= 1_000:
        return f"{fval:,.{precision}f}"
    return f"{fval:.{precision}f}"


def _filter_co2_badge_colour(co2e_kg: float) -> str:
    """Jinja2 filter: return a CSS colour class based on CO2e magnitude.

    Colours provide a traffic-light indication of environmental impact:

    * Green  (``"badge--green"``)  — < 1 kg CO2e (very low).
    * Yellow (``"badge--yellow"``) — 1–100 kg CO2e (moderate).
    * Orange (``"badge--orange"``) — 100–1 000 kg CO2e (significant).
    * Red    (``"badge--red"``)    — > 1 000 kg CO2e (high).

    Args:
        co2e_kg: CO2-equivalent emissions in kilograms.

    Returns:
        CSS modifier class string.

    Example::

        <div class="report-card__header {{ result.co2e_kg | co2_badge_colour }}">
    """
    try:
        val = float(co2e_kg)
    except (TypeError, ValueError):
        return "badge--grey"

    if val < 1.0:
        return "badge--green"
    if val < 100.0:
        return "badge--yellow"
    if val < 1_000.0:
        return "badge--orange"
    return "badge--red"


def _filter_energy_badge_colour(energy_kwh: float) -> str:
    """Jinja2 filter: return a CSS colour class based on energy magnitude.

    * Green  — < 1 kWh.
    * Yellow — 1–100 kWh.
    * Orange — 100–10 000 kWh.
    * Red    — > 10 000 kWh.

    Args:
        energy_kwh: Energy consumption in kilowatt-hours.

    Returns:
        CSS modifier class string.
    """
    try:
        val = float(energy_kwh)
    except (TypeError, ValueError):
        return "badge--grey"

    if val < 1.0:
        return "badge--green"
    if val < 100.0:
        return "badge--yellow"
    if val < 10_000.0:
        return "badge--orange"
    return "badge--red"
