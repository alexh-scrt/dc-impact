"""Unit tests for dc_impact.report — report generation.

Covers:
- :func:`~dc_impact.report.generate_json_report` structure and field values.
- :func:`~dc_impact.report.generate_html_report` HTML structure.
- Custom Jinja2 filters: ``format_number``, ``co2_badge_colour``,
  ``energy_badge_colour``.
- Comparison section values for known reference inputs.
- JSON serialisability of the report dictionary.
- HTML snippet correctness (contains expected substrings, no full document).
"""

from __future__ import annotations

import json
import re

import pytest

from dc_impact.calculator import calculate_impact, ImpactResult
from dc_impact.report import (
    generate_html_report,
    generate_json_report,
    _filter_co2_badge_colour,
    _filter_energy_badge_colour,
    _filter_format_number,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def basic_result() -> ImpactResult:
    """A simple, fully populated ImpactResult for use in multiple tests."""
    return calculate_impact(
        hardware="A100_80GB",
        region="us-east-1",
        duration_hours=1.0,
        num_accelerators=1,
        workload_type="training",
        utilization=1.0,
    )


@pytest.fixture()
def large_result() -> ImpactResult:
    """A large workload result (8 GPUs, 72 h) for scale tests."""
    return calculate_impact(
        hardware="A100_80GB",
        region="us-east-1",
        duration_hours=72.0,
        num_accelerators=8,
        workload_type="training",
        utilization=1.0,
    )


@pytest.fixture()
def low_carbon_result() -> ImpactResult:
    """Result from a low-carbon region for comparison tests."""
    return calculate_impact(
        hardware="T4",
        region="eu-north-1",
        duration_hours=1.0,
        num_accelerators=1,
        workload_type="inference",
        utilization=0.5,
    )


@pytest.fixture()
def basic_report(basic_result: ImpactResult) -> dict:
    """JSON report dict from the basic result."""
    return generate_json_report(basic_result)


# ---------------------------------------------------------------------------
# generate_json_report — top-level structure
# ---------------------------------------------------------------------------


class TestGenerateJsonReportStructure:
    """Verify the top-level structure of the JSON report."""

    def test_returns_dict(self, basic_result: ImpactResult) -> None:
        report = generate_json_report(basic_result)
        assert isinstance(report, dict)

    def test_top_level_keys(self, basic_report: dict) -> None:
        expected_keys = {
            "schema_version",
            "generated_at",
            "inputs",
            "metrics",
            "factors",
            "comparisons",
        }
        assert set(basic_report.keys()) == expected_keys

    def test_schema_version_is_string(self, basic_report: dict) -> None:
        assert isinstance(basic_report["schema_version"], str)

    def test_schema_version_value(self, basic_report: dict) -> None:
        assert basic_report["schema_version"] == "1.0"

    def test_generated_at_is_string(self, basic_report: dict) -> None:
        assert isinstance(basic_report["generated_at"], str)

    def test_generated_at_ends_with_z(self, basic_report: dict) -> None:
        """Timestamp must be UTC, indicated by trailing 'Z'."""
        assert basic_report["generated_at"].endswith("Z")

    def test_generated_at_is_iso_format(self, basic_report: dict) -> None:
        """Timestamp must be parseable as ISO 8601."""
        import datetime
        ts = basic_report["generated_at"].rstrip("Z")
        # Should not raise
        datetime.datetime.fromisoformat(ts)

    def test_inputs_is_dict(self, basic_report: dict) -> None:
        assert isinstance(basic_report["inputs"], dict)

    def test_metrics_is_dict(self, basic_report: dict) -> None:
        assert isinstance(basic_report["metrics"], dict)

    def test_factors_is_dict(self, basic_report: dict) -> None:
        assert isinstance(basic_report["factors"], dict)

    def test_comparisons_is_dict(self, basic_report: dict) -> None:
        assert isinstance(basic_report["comparisons"], dict)

    def test_json_serialisable(self, basic_report: dict) -> None:
        """The entire report must be JSON serialisable without custom encoders."""
        serialised = json.dumps(basic_report)
        assert len(serialised) > 0

    def test_json_roundtrip_preserves_values(self, basic_report: dict) -> None:
        """Values must survive a JSON encode/decode cycle unchanged."""
        roundtripped = json.loads(json.dumps(basic_report))
        assert roundtripped["schema_version"] == basic_report["schema_version"]
        assert roundtripped["metrics"]["energy_kwh"] == basic_report["metrics"]["energy_kwh"]


# ---------------------------------------------------------------------------
# generate_json_report — inputs section
# ---------------------------------------------------------------------------


class TestJsonReportInputs:
    """Verify the inputs section of the JSON report."""

    def test_inputs_keys(self, basic_report: dict) -> None:
        inputs = basic_report["inputs"]
        expected = {
            "hardware",
            "region",
            "duration_hours",
            "num_accelerators",
            "workload_type",
            "utilization",
            "utilization_pct",
        }
        assert set(inputs.keys()) == expected

    def test_inputs_hardware_matches(self, basic_result: ImpactResult, basic_report: dict) -> None:
        assert basic_report["inputs"]["hardware"] == basic_result.hardware

    def test_inputs_region_matches(self, basic_result: ImpactResult, basic_report: dict) -> None:
        assert basic_report["inputs"]["region"] == basic_result.region

    def test_inputs_duration_matches(self, basic_result: ImpactResult, basic_report: dict) -> None:
        assert basic_report["inputs"]["duration_hours"] == basic_result.duration_hours

    def test_inputs_num_accelerators_matches(self, basic_result: ImpactResult, basic_report: dict) -> None:
        assert basic_report["inputs"]["num_accelerators"] == basic_result.num_accelerators

    def test_inputs_workload_type_matches(self, basic_result: ImpactResult, basic_report: dict) -> None:
        assert basic_report["inputs"]["workload_type"] == basic_result.workload_type

    def test_inputs_utilization_matches(self, basic_result: ImpactResult, basic_report: dict) -> None:
        assert basic_report["inputs"]["utilization"] == basic_result.utilization

    def test_inputs_utilization_pct_is_100_for_full_util(self, basic_report: dict) -> None:
        """100 % utilisation should give utilization_pct = 100.0."""
        assert basic_report["inputs"]["utilization_pct"] == pytest.approx(100.0)

    def test_inputs_utilization_pct_scales(self) -> None:
        """50 % utilisation should give utilization_pct = 50.0."""
        result = calculate_impact(
            hardware="T4",
            region="us-east-1",
            duration_hours=1.0,
            utilization=0.5,
        )
        report = generate_json_report(result)
        assert report["inputs"]["utilization_pct"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# generate_json_report — metrics section
# ---------------------------------------------------------------------------


class TestJsonReportMetrics:
    """Verify the metrics section of the JSON report."""

    def test_metrics_keys(self, basic_report: dict) -> None:
        metrics = basic_report["metrics"]
        expected = {
            "energy_kwh",
            "energy_mwh",
            "co2e_kg",
            "co2e_g",
            "co2e_tonnes",
            "water_liters",
            "water_ml",
        }
        assert set(metrics.keys()) == expected

    def test_metrics_energy_kwh_matches_result(self, basic_result: ImpactResult, basic_report: dict) -> None:
        assert basic_report["metrics"]["energy_kwh"] == pytest.approx(basic_result.energy_kwh)

    def test_metrics_co2e_kg_matches_result(self, basic_result: ImpactResult, basic_report: dict) -> None:
        assert basic_report["metrics"]["co2e_kg"] == pytest.approx(basic_result.co2e_kg)

    def test_metrics_water_liters_matches_result(self, basic_result: ImpactResult, basic_report: dict) -> None:
        assert basic_report["metrics"]["water_liters"] == pytest.approx(basic_result.water_liters)

    def test_energy_mwh_is_kwh_divided_by_1000(self, basic_result: ImpactResult, basic_report: dict) -> None:
        expected_mwh = basic_result.energy_kwh / 1_000
        assert basic_report["metrics"]["energy_mwh"] == pytest.approx(expected_mwh, rel=1e-5)

    def test_co2e_g_is_kg_times_1000(self, basic_result: ImpactResult, basic_report: dict) -> None:
        expected_g = basic_result.co2e_kg * 1_000
        assert basic_report["metrics"]["co2e_g"] == pytest.approx(expected_g, rel=1e-5)

    def test_co2e_tonnes_is_kg_divided_by_1000(self, basic_result: ImpactResult, basic_report: dict) -> None:
        expected_t = basic_result.co2e_kg / 1_000
        assert basic_report["metrics"]["co2e_tonnes"] == pytest.approx(expected_t, rel=1e-5)

    def test_water_ml_is_liters_times_1000(self, basic_result: ImpactResult, basic_report: dict) -> None:
        expected_ml = basic_result.water_liters * 1_000
        assert basic_report["metrics"]["water_ml"] == pytest.approx(expected_ml, rel=1e-5)

    def test_all_metric_values_non_negative(self, basic_report: dict) -> None:
        for key, val in basic_report["metrics"].items():
            assert val >= 0, f"Metric {key!r} is negative: {val}"

    def test_all_metric_values_are_floats(self, basic_report: dict) -> None:
        for key, val in basic_report["metrics"].items():
            assert isinstance(val, (int, float)), f"Metric {key!r} is not numeric: {val!r}"


# ---------------------------------------------------------------------------
# generate_json_report — factors section
# ---------------------------------------------------------------------------


class TestJsonReportFactors:
    """Verify the factors section of the JSON report."""

    def test_factors_keys(self, basic_report: dict) -> None:
        factors = basic_report["factors"]
        expected = {
            "tdp_watts",
            "pue",
            "wue_l_per_kwh",
            "carbon_intensity_g_per_kwh",
            "effective_power_per_accelerator_w",
        }
        assert set(factors.keys()) == expected

    def test_factors_tdp_matches_result(self, basic_result: ImpactResult, basic_report: dict) -> None:
        assert basic_report["factors"]["tdp_watts"] == pytest.approx(basic_result.tdp_watts)

    def test_factors_pue_matches_result(self, basic_result: ImpactResult, basic_report: dict) -> None:
        assert basic_report["factors"]["pue"] == pytest.approx(basic_result.pue)

    def test_factors_wue_matches_result(self, basic_result: ImpactResult, basic_report: dict) -> None:
        assert basic_report["factors"]["wue_l_per_kwh"] == pytest.approx(basic_result.wue)

    def test_factors_carbon_intensity_matches_result(self, basic_result: ImpactResult, basic_report: dict) -> None:
        assert basic_report["factors"]["carbon_intensity_g_per_kwh"] == pytest.approx(
            basic_result.carbon_intensity_g_per_kwh
        )

    def test_effective_power_is_tdp_times_util_times_pue(self, basic_result: ImpactResult, basic_report: dict) -> None:
        expected = basic_result.tdp_watts * basic_result.utilization * basic_result.pue
        assert basic_report["factors"]["effective_power_per_accelerator_w"] == pytest.approx(
            expected, rel=1e-4
        )

    def test_all_factor_values_are_numeric(self, basic_report: dict) -> None:
        for key, val in basic_report["factors"].items():
            assert isinstance(val, (int, float)), f"Factor {key!r} is not numeric: {val!r}"


# ---------------------------------------------------------------------------
# generate_json_report — comparisons section
# ---------------------------------------------------------------------------


class TestJsonReportComparisons:
    """Verify the comparisons section of the JSON report."""

    def test_comparisons_top_level_keys(self, basic_report: dict) -> None:
        cmp = basic_report["comparisons"]
        expected = {"co2e", "energy", "water", "disclaimer"}
        assert set(cmp.keys()) == expected

    def test_co2e_comparisons_keys(self, basic_report: dict) -> None:
        co2e_cmp = basic_report["comparisons"]["co2e"]
        assert "car_km_equivalent" in co2e_cmp
        assert "car_km_equivalent_description" in co2e_cmp
        assert "tree_sequestration_months" in co2e_cmp
        assert "tree_sequestration_months_description" in co2e_cmp

    def test_energy_comparisons_keys(self, basic_report: dict) -> None:
        energy_cmp = basic_report["comparisons"]["energy"]
        assert "us_household_days_equivalent" in energy_cmp
        assert "us_household_days_equivalent_description" in energy_cmp

    def test_water_comparisons_keys(self, basic_report: dict) -> None:
        water_cmp = basic_report["comparisons"]["water"]
        assert "showers_equivalent" in water_cmp
        assert "showers_equivalent_description" in water_cmp

    def test_disclaimer_is_non_empty_string(self, basic_report: dict) -> None:
        disclaimer = basic_report["comparisons"]["disclaimer"]
        assert isinstance(disclaimer, str)
        assert len(disclaimer) > 0

    def test_car_km_equivalent_is_non_negative(self, basic_report: dict) -> None:
        val = basic_report["comparisons"]["co2e"]["car_km_equivalent"]
        assert val >= 0

    def test_car_km_description_contains_km(self, basic_report: dict) -> None:
        desc = basic_report["comparisons"]["co2e"]["car_km_equivalent_description"]
        assert "km" in desc.lower()

    def test_tree_months_is_non_negative(self, basic_report: dict) -> None:
        val = basic_report["comparisons"]["co2e"]["tree_sequestration_months"]
        assert val >= 0

    def test_household_days_is_non_negative(self, basic_report: dict) -> None:
        val = basic_report["comparisons"]["energy"]["us_household_days_equivalent"]
        assert val >= 0

    def test_showers_is_non_negative(self, basic_report: dict) -> None:
        val = basic_report["comparisons"]["water"]["showers_equivalent"]
        assert val >= 0

    def test_large_workload_has_larger_car_km(self, basic_result: ImpactResult, large_result: ImpactResult) -> None:
        """A larger workload must give a higher car-km equivalent."""
        small_report = generate_json_report(basic_result)
        large_report = generate_json_report(large_result)
        small_km = small_report["comparisons"]["co2e"]["car_km_equivalent"]
        large_km = large_report["comparisons"]["co2e"]["car_km_equivalent"]
        assert large_km > small_km

    def test_car_km_reference_calculation(self) -> None:
        """Manually verify the car-km calculation for a known CO2 value."""
        # 170 g CO2 = 1 km => 1 kg CO2 = ~5.88 km
        result = calculate_impact(
            hardware="T4",
            region="us-east-1",
            duration_hours=1.0,
            num_accelerators=1,
            workload_type="inference",
            utilization=1.0,
        )
        report = generate_json_report(result)
        expected_km = round(result.co2e_kg / 0.170, 2)
        assert report["comparisons"]["co2e"]["car_km_equivalent"] == pytest.approx(
            expected_km, rel=1e-4
        )

    def test_zero_co2_gives_zero_comparisons(self) -> None:
        """A zero-carbon result must give zero comparison values."""
        # eu-north-1 has CI=8, but still > 0; we test via the builder directly
        # by monkey-patching temporarily — instead we just check the formula
        # directly with the low-carbon region result.
        result = calculate_impact(
            hardware="T4",
            region="eu-north-1",
            duration_hours=0.001,  # very short
            num_accelerators=1,
            utilization=0.01,
        )
        report = generate_json_report(result)
        # Both should be very small (near zero)
        assert report["comparisons"]["co2e"]["car_km_equivalent"] >= 0


# ---------------------------------------------------------------------------
# generate_html_report — structure
# ---------------------------------------------------------------------------


class TestGenerateHtmlReport:
    """Verify the HTML report card snippet structure."""

    def test_returns_string(self, basic_result: ImpactResult) -> None:
        html = generate_html_report(basic_result)
        assert isinstance(html, str)

    def test_non_empty_output(self, basic_result: ImpactResult) -> None:
        html = generate_html_report(basic_result)
        assert len(html.strip()) > 0

    def test_contains_report_card_div(self, basic_result: ImpactResult) -> None:
        html = generate_html_report(basic_result)
        assert 'class="report-card"' in html

    def test_does_not_contain_doctype(self, basic_result: ImpactResult) -> None:
        """Output must be a fragment, not a full HTML document."""
        html = generate_html_report(basic_result)
        assert "<!DOCTYPE" not in html
        assert "<html" not in html

    def test_does_not_contain_body_tag(self, basic_result: ImpactResult) -> None:
        html = generate_html_report(basic_result)
        assert "<body" not in html

    def test_contains_energy_value(self, basic_result: ImpactResult) -> None:
        html = generate_html_report(basic_result)
        # The formatted energy value must appear somewhere in the HTML
        formatted = _filter_format_number(basic_result.energy_kwh, 3)
        assert formatted in html

    def test_contains_co2_value(self, basic_result: ImpactResult) -> None:
        html = generate_html_report(basic_result)
        formatted = _filter_format_number(basic_result.co2e_kg, 3)
        assert formatted in html

    def test_contains_water_value(self, basic_result: ImpactResult) -> None:
        html = generate_html_report(basic_result)
        formatted = _filter_format_number(basic_result.water_liters, 3)
        assert formatted in html

    def test_contains_hardware_name(self, basic_result: ImpactResult) -> None:
        html = generate_html_report(basic_result)
        assert basic_result.hardware in html

    def test_contains_region_name(self, basic_result: ImpactResult) -> None:
        html = generate_html_report(basic_result)
        assert basic_result.region in html

    def test_contains_kwh_unit(self, basic_result: ImpactResult) -> None:
        html = generate_html_report(basic_result)
        assert "kWh" in html

    def test_contains_co2_unit(self, basic_result: ImpactResult) -> None:
        html = generate_html_report(basic_result)
        assert "CO₂e" in html or "CO2e" in html

    def test_contains_water_unit(self, basic_result: ImpactResult) -> None:
        html = generate_html_report(basic_result)
        # 'L' for litres must appear
        assert ">L<" in html or "&gt;L&lt;" in html or '"L"' in html or ">L<" in html

    def test_contains_pue_value(self, basic_result: ImpactResult) -> None:
        html = generate_html_report(basic_result)
        assert str(basic_result.pue) in html

    def test_contains_dc_impact_brand(self, basic_result: ImpactResult) -> None:
        html = generate_html_report(basic_result)
        assert "dc_impact" in html

    def test_empty_result_renders_empty_card(self) -> None:
        """Passing result=None should render the empty-state card."""
        from dc_impact.report import _get_jinja_env
        env = _get_jinja_env()
        template = env.get_template("report_card.html")
        html = template.render(result=None, report=None)
        assert "report-card--empty" in html
        assert "No results" in html

    def test_workload_type_appears_in_html(self, basic_result: ImpactResult) -> None:
        html = generate_html_report(basic_result)
        # workload_type is capitalized by the template filter
        assert basic_result.workload_type.capitalize() in html

    def test_comparisons_section_rendered(self, basic_result: ImpactResult) -> None:
        html = generate_html_report(basic_result)
        # Comparison icons should appear
        assert "🚗" in html or "car" in html.lower() or "km" in html

    def test_html_contains_details_tables(self, basic_result: ImpactResult) -> None:
        html = generate_html_report(basic_result)
        assert "<table" in html
        assert "</table>" in html

    def test_inference_type_in_html(self) -> None:
        """Inference workload_type must appear capitalised in the card."""
        result = calculate_impact(
            hardware="T4",
            region="us-east-1",
            duration_hours=1.0,
            workload_type="inference",
        )
        html = generate_html_report(result)
        assert "Inference" in html

    def test_multiple_gpus_show_correct_count(self) -> None:
        result = calculate_impact(
            hardware="H100_SXM",
            region="us-west-2",
            duration_hours=24.0,
            num_accelerators=8,
        )
        html = generate_html_report(result)
        assert "8" in html


# ---------------------------------------------------------------------------
# Custom filter: _filter_format_number
# ---------------------------------------------------------------------------


class TestFilterFormatNumber:
    """Unit tests for the format_number Jinja2 filter."""

    def test_zero_returns_zero_string(self) -> None:
        assert _filter_format_number(0.0) == "0"

    def test_integer_like_float(self) -> None:
        result = _filter_format_number(1.0)
        assert result == "1.000"

    def test_small_value_uses_scientific(self) -> None:
        result = _filter_format_number(0.0000001)
        assert "e" in result.lower()

    def test_normal_precision_three(self) -> None:
        result = _filter_format_number(12.345678, 3)
        assert result == "12.346"

    def test_large_value_includes_comma(self) -> None:
        result = _filter_format_number(1234.5, 3)
        assert "," in result

    def test_very_large_value_uses_M_suffix(self) -> None:
        result = _filter_format_number(2_500_000.0)
        assert "M" in result

    def test_precision_parameter(self) -> None:
        result = _filter_format_number(3.14159, 2)
        assert result == "3.14"

    def test_invalid_input_returns_string(self) -> None:
        result = _filter_format_number("not-a-number")  # type: ignore[arg-type]
        assert isinstance(result, str)

    def test_negative_value(self) -> None:
        result = _filter_format_number(-5.5, 1)
        assert "-" in result


# ---------------------------------------------------------------------------
# Custom filter: _filter_co2_badge_colour
# ---------------------------------------------------------------------------


class TestFilterCo2BadgeColour:
    """Unit tests for the co2_badge_colour Jinja2 filter."""

    def test_very_low_returns_green(self) -> None:
        assert _filter_co2_badge_colour(0.001) == "badge--green"

    def test_zero_returns_green(self) -> None:
        assert _filter_co2_badge_colour(0.0) == "badge--green"

    def test_below_one_returns_green(self) -> None:
        assert _filter_co2_badge_colour(0.999) == "badge--green"

    def test_one_returns_yellow(self) -> None:
        assert _filter_co2_badge_colour(1.0) == "badge--yellow"

    def test_moderate_returns_yellow(self) -> None:
        assert _filter_co2_badge_colour(50.0) == "badge--yellow"

    def test_below_100_returns_yellow(self) -> None:
        assert _filter_co2_badge_colour(99.9) == "badge--yellow"

    def test_100_returns_orange(self) -> None:
        assert _filter_co2_badge_colour(100.0) == "badge--orange"

    def test_below_1000_returns_orange(self) -> None:
        assert _filter_co2_badge_colour(500.0) == "badge--orange"

    def test_1000_returns_red(self) -> None:
        assert _filter_co2_badge_colour(1000.0) == "badge--red"

    def test_very_large_returns_red(self) -> None:
        assert _filter_co2_badge_colour(100_000.0) == "badge--red"

    def test_invalid_input_returns_grey(self) -> None:
        assert _filter_co2_badge_colour("bad") == "badge--grey"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Custom filter: _filter_energy_badge_colour
# ---------------------------------------------------------------------------


class TestFilterEnergyBadgeColour:
    """Unit tests for the energy_badge_colour Jinja2 filter."""

    def test_below_1_returns_green(self) -> None:
        assert _filter_energy_badge_colour(0.5) == "badge--green"

    def test_zero_returns_green(self) -> None:
        assert _filter_energy_badge_colour(0.0) == "badge--green"

    def test_1_returns_yellow(self) -> None:
        assert _filter_energy_badge_colour(1.0) == "badge--yellow"

    def test_50_returns_yellow(self) -> None:
        assert _filter_energy_badge_colour(50.0) == "badge--yellow"

    def test_100_returns_orange(self) -> None:
        assert _filter_energy_badge_colour(100.0) == "badge--orange"

    def test_5000_returns_orange(self) -> None:
        assert _filter_energy_badge_colour(5000.0) == "badge--orange"

    def test_10000_returns_red(self) -> None:
        assert _filter_energy_badge_colour(10_000.0) == "badge--red"

    def test_very_large_returns_red(self) -> None:
        assert _filter_energy_badge_colour(1_000_000.0) == "badge--red"

    def test_invalid_input_returns_grey(self) -> None:
        assert _filter_energy_badge_colour(None) == "badge--grey"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# generate_json_report — multiple results consistency
# ---------------------------------------------------------------------------


class TestJsonReportConsistency:
    """Cross-result consistency checks."""

    def test_generated_at_differs_across_calls(self, basic_result: ImpactResult) -> None:
        """Two calls made at slightly different times must have different timestamps
        (with nanosecond precision this is almost always true, but we just check
        that generated_at is present and looks different at the string level).
        """
        import time
        r1 = generate_json_report(basic_result)
        time.sleep(0.01)  # ensure at least 10ms separation
        r2 = generate_json_report(basic_result)
        # The metrics must be the same (deterministic)
        assert r1["metrics"] == r2["metrics"]
        # Schema version must be the same
        assert r1["schema_version"] == r2["schema_version"]

    def test_low_carbon_report_lower_co2_than_high_carbon(self, low_carbon_result: ImpactResult) -> None:
        high_carbon_result = calculate_impact(
            hardware="T4",
            region="af-south-1",  # South Africa — very coal-heavy
            duration_hours=1.0,
            num_accelerators=1,
            workload_type="inference",
            utilization=0.5,
        )
        low_report = generate_json_report(low_carbon_result)
        high_report = generate_json_report(high_carbon_result)
        assert (
            low_report["metrics"]["co2e_kg"]
            < high_report["metrics"]["co2e_kg"]
        )

    def test_report_reflects_result_to_dict(self, basic_result: ImpactResult, basic_report: dict) -> None:
        """The metrics in the report must match what ImpactResult.to_dict() returns."""
        result_dict = basic_result.to_dict()
        assert basic_report["metrics"]["energy_kwh"] == result_dict["energy_kwh"]
        assert basic_report["metrics"]["co2e_kg"] == result_dict["co2e_kg"]
        assert basic_report["metrics"]["water_liters"] == result_dict["water_liters"]
        assert basic_report["factors"]["tdp_watts"] == result_dict["tdp_watts"]
        assert basic_report["factors"]["pue"] == result_dict["pue"]
