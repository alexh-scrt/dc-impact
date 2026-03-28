"""Smoke tests verifying that the phase-1 scaffold imports correctly and
exposes the expected public API surface.

These tests do not exercise business logic — they simply confirm that all
modules can be imported without errors and that key names are accessible,
guarding against typos and circular-import issues.
"""

from __future__ import annotations

import importlib

import pytest


def test_package_importable() -> None:
    """dc_impact package can be imported."""
    import dc_impact  # noqa: F401


def test_create_app_exposed() -> None:
    """create_app is importable from the top-level package."""
    from dc_impact import create_app

    assert callable(create_app)


def test_app_module_importable() -> None:
    """dc_impact.app module imports without error."""
    import dc_impact.app  # noqa: F401


def test_calculator_module_importable() -> None:
    """dc_impact.calculator module imports without error."""
    import dc_impact.calculator  # noqa: F401


def test_data_module_importable() -> None:
    """dc_impact.data module imports without error."""
    import dc_impact.data  # noqa: F401


def test_report_module_importable() -> None:
    """dc_impact.report module imports without error."""
    import dc_impact.report  # noqa: F401


def test_data_hardware_table_not_empty() -> None:
    """HARDWARE_TDP_WATTS lookup table is a non-empty dict."""
    from dc_impact.data import HARDWARE_TDP_WATTS

    assert isinstance(HARDWARE_TDP_WATTS, dict)
    assert len(HARDWARE_TDP_WATTS) > 0


def test_data_carbon_intensity_table_not_empty() -> None:
    """CARBON_INTENSITY_G_PER_KWH lookup table is a non-empty dict."""
    from dc_impact.data import CARBON_INTENSITY_G_PER_KWH

    assert isinstance(CARBON_INTENSITY_G_PER_KWH, dict)
    assert len(CARBON_INTENSITY_G_PER_KWH) > 0


def test_data_pue_table_not_empty() -> None:
    """PUE_BY_REGION lookup table is a non-empty dict."""
    from dc_impact.data import PUE_BY_REGION

    assert isinstance(PUE_BY_REGION, dict)
    assert len(PUE_BY_REGION) > 0


def test_data_wue_table_not_empty() -> None:
    """WUE_BY_REGION lookup table is a non-empty dict."""
    from dc_impact.data import WUE_BY_REGION

    assert isinstance(WUE_BY_REGION, dict)
    assert len(WUE_BY_REGION) > 0


def test_default_pue_gte_one() -> None:
    """DEFAULT_PUE must be >= 1.0 (physically meaningful)."""
    from dc_impact.data import DEFAULT_PUE

    assert DEFAULT_PUE >= 1.0


def test_default_wue_gte_zero() -> None:
    """DEFAULT_WUE must be non-negative."""
    from dc_impact.data import DEFAULT_WUE

    assert DEFAULT_WUE >= 0.0


def test_create_app_returns_flask_app() -> None:
    """create_app() returns a Flask application instance."""
    from flask import Flask

    from dc_impact import create_app

    app = create_app({"TESTING": True})
    assert isinstance(app, Flask)


def test_calculator_impact_result_importable() -> None:
    """ImpactResult dataclass is importable from calculator module."""
    from dc_impact.calculator import ImpactResult

    assert ImpactResult is not None


def test_calculator_error_importable() -> None:
    """CalculatorError exception is importable from calculator module."""
    from dc_impact.calculator import CalculatorError

    assert issubclass(CalculatorError, ValueError)


def test_report_functions_importable() -> None:
    """Public functions from report module are importable."""
    from dc_impact.report import generate_html_report, generate_json_report

    assert callable(generate_html_report)
    assert callable(generate_json_report)
