"""Report generation for dc_impact.

This module exposes two public functions:

* :func:`generate_json_report` — converts an :class:`~dc_impact.calculator.ImpactResult`
  into a structured, serialisable dictionary suitable for use as a JSON API
  response or for embedding in model cards.
* :func:`generate_html_report` — renders the same result as an embeddable
  HTML snippet using Jinja2 and the ``report_card.html`` template.

Both functions are implemented as stubs in phase 1 and will be fully
implemented in phase 4.
"""

from __future__ import annotations

import datetime
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from dc_impact.calculator import ImpactResult


def generate_json_report(result: ImpactResult) -> dict[str, Any]:
    """Build a structured JSON-serialisable report dictionary.

    The returned mapping includes all computed metrics plus contextual
    metadata (hardware, region, workload type, etc.) in a format ready
    for API responses or embedding in Hugging Face model cards.

    Args:
        result: A fully populated :class:`~dc_impact.calculator.ImpactResult`
            returned by :func:`~dc_impact.calculator.calculate_impact`.

    Returns:
        A plain dictionary with the following top-level keys:

        - ``"schema_version"`` (str): Report schema version.
        - ``"generated_at"`` (str): ISO 8601 UTC timestamp.
        - ``"inputs"`` (dict): Input parameters used for the calculation.
        - ``"metrics"`` (dict): Computed impact metrics.
        - ``"factors"`` (dict): Lookup factors applied (PUE, WUE, CI).
    """
    return {
        "schema_version": "1.0",
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "inputs": {
            "hardware": result.hardware,
            "region": result.region,
            "duration_hours": result.duration_hours,
            "num_accelerators": result.num_accelerators,
            "workload_type": result.workload_type,
            "utilization": result.utilization,
        },
        "metrics": {
            "energy_kwh": result.energy_kwh,
            "co2e_kg": result.co2e_kg,
            "water_liters": result.water_liters,
        },
        "factors": {
            "tdp_watts": result.tdp_watts,
            "pue": result.pue,
            "wue": result.wue,
            "carbon_intensity_g_per_kwh": result.carbon_intensity_g_per_kwh,
        },
    }


def generate_html_report(result: ImpactResult) -> str:
    """Render an embeddable HTML report card snippet.

    Uses Jinja2 with the package's ``templates/report_card.html`` template
    to produce a self-contained HTML fragment suitable for inclusion in
    GitHub READMEs or Hugging Face model cards.

    Args:
        result: A fully populated :class:`~dc_impact.calculator.ImpactResult`
            returned by :func:`~dc_impact.calculator.calculate_impact`.

    Returns:
        Rendered HTML string (a ``<div>`` fragment, not a full document).
    """
    env = _get_jinja_env()
    template = env.get_template("report_card.html")
    report_data = generate_json_report(result)
    return template.render(result=result, report=report_data)


def _get_jinja_env() -> Environment:
    """Construct a Jinja2 :class:`~jinja2.Environment` configured to load
    templates from the ``dc_impact`` package.

    Returns:
        A :class:`~jinja2.Environment` instance with autoescaping enabled
        for HTML files.
    """
    env = Environment(
        loader=PackageLoader("dc_impact", "templates"),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env
