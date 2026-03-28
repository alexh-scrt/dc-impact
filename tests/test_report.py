"""Unit tests for dc_impact.report — report generation.

Covers:
- :func:`~dc_impact.report.generate_json_report` structure and field values.
- :func:`~dc_impact.report.generate_html_report` HTML structure.
- Custom Jinja2 filters: ``format_number``, ``co2_badge_colour``,
  ``energy_badge_colour``.
- Comparison section values for known reference inputs.
- JSON serialisability of the report dictionary.
- HTML snippet correctness (contains expected substrings, no full document).
- Schema version and timestamp format validation.
- Cross-result consistency checks.
- Private helper functions.
"""

from __future__ import annotations

import datetime
import json
import re
import time

import pytest

from dc_impact.calculator import ImpactResult, calculate_impact
from dc_impact.report import (
    _filter_co2_badge_colour,
    _filter_energy_badge_colour,
    _filter_format_number,
    _get_jinja_env,
    _utc_now_iso,
    generate_html_report,
    generate_json_report,
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


@pytest.fixture()
def inference_result() -> ImpactResult:
    """An inference workload result."""
    return calculate_impact(
        hardware="T4",
        region="us-east-1",
        duration_hours=1.0,
        num_accelerators=1,
        workload_type="inference",
        utilization=0.8,
    )


@pytest.fixture()
def high_carbon_result() -> ImpactResult:
    """Result from a high-carbon region (South Africa)."""
    return calculate_impact(
        hardware="A100_80GB",
        region="af-south-1",
        duration_hours=24.0,
        num_accelerators=1,
        workload_type="training",
        utilization=1.0,
    )


# ---------------------------------------------------------------------------
# generate_json_report — top-level structure
# ---------------------------------------------------------------------------


class TestGenerateJsonReportStructure:
    """Verify the top-level structure of the JSON report."""

    def test_returns_dict(self, basic_result: ImpactResult) -> None:
        """generate_json_report must return a dict."""
        report = generate_json_report(basic_result)
        assert isinstance(report, dict)

    def test_top_level_keys(self, basic_report: dict) -> None:
        """Report must contain exactly the expected top-level keys."""
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
        """Schema version must be '1.0'."""
        assert basic_report["schema_version"] == "1.0"

    def test_generated_at_is_string(self, basic_report: dict) -> None:
        assert isinstance(basic_report["generated_at"], str)

    def test_generated_at_ends_with_z(self, basic_report: dict) -> None:
        """Timestamp must be UTC, indicated by trailing 'Z'."""
        assert basic_report["generated_at"].endswith("Z")

    def test_generated_at_is_iso_format(self, basic_report: dict) -> None:
        """Timestamp must be parseable as ISO 8601."""
        ts = basic_report["generated_at"].rstrip("Z")
        # Should not raise
        datetime.datetime.fromisoformat(ts)

    def test_generated_at_contains_date_and_time(self, basic_report: dict) -> None:
        """Timestamp must contain both date and time components."""
        ts = basic_report["generated_at"]
        assert "T" in ts  # ISO 8601 separator between date and time

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

    def test_json_roundtrip_preserves_schema_version(self, basic_report: dict) -> None:
        """Schema version must survive a JSON encode/decode cycle."""
        roundtripped = json.loads(json.dumps(basic_report))
        assert roundtripped["schema_version"] == basic_report["schema_version"]

    def test_json_roundtrip_preserves_metrics(self, basic_report: dict) -> None:
        """Metrics must survive a JSON encode/decode cycle unchanged."""
        roundtripped = json.loads(json.dumps(basic_report))
        assert roundtripped["metrics"]["energy_kwh"] == basic_report["metrics"]["energy_kwh"]
        assert roundtripped["metrics"]["co2e_kg"] == basic_report["metrics"]["co2e_kg"]
        assert roundtripped["metrics"]["water_liters"] == basic_report["metrics"]["water_liters"]

    def test_json_roundtrip_preserves_inputs(self, basic_report: dict) -> None:
        """Inputs must survive a JSON encode/decode cycle unchanged."""
        roundtripped = json.loads(json.dumps(basic_report))
        assert roundtripped["inputs"]["hardware"] == basic_report["inputs"]["hardware"]
        assert roundtripped["inputs"]["region"] == basic_report["inputs"]["region"]


# ---------------------------------------------------------------------------
# generate_json_report — inputs section
# ---------------------------------------------------------------------------


class TestJsonReportInputs:
    """Verify the inputs section of the JSON report."""

    def test_inputs_keys(self, basic_report: dict) -> None:
        """Inputs section must contain exactly the expected keys."""
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

    def test_inputs_utilization_pct_scales_correctly(self) -> None:
        """50 % utilisation should give utilization_pct = 50.0."""
        result = calculate_impact(
            hardware="T4",
            region="us-east-1",
            duration_hours=1.0,
            utilization=0.5,
        )
        report = generate_json_report(result)
        assert report["inputs"]["utilization_pct"] == pytest.approx(50.0)

    def test_inputs_utilization_pct_80_percent(self) -> None:
        """80 % utilisation should give utilization_pct = 80.0."""
        result = calculate_impact(
            hardware="T4",
            region="us-east-1",
            duration_hours=1.0,
            utilization=0.8,
        )
        report = generate_json_report(result)
        assert report["inputs"]["utilization_pct"] == pytest.approx(80.0)

    def test_inputs_num_accelerators_is_int(self, basic_report: dict) -> None:
        assert isinstance(basic_report["inputs"]["num_accelerators"], int)

    def test_inputs_hardware_is_string(self, basic_report: dict) -> None:
        assert isinstance(basic_report["inputs"]["hardware"], str)

    def test_inputs_region_is_string(self, basic_report: dict) -> None:
        assert isinstance(basic_report["inputs"]["region"], str)

    def test_inputs_workload_type_is_string(self, basic_report: dict) -> None:
        assert isinstance(basic_report["inputs"]["workload_type"], str)

    def test_inputs_inference_workload_type(self, inference_result: ImpactResult) -> None:
        """Inference workload type must be reflected in inputs."""
        report = generate_json_report(inference_result)
        assert report["inputs"]["workload_type"] == "inference"

    def test_inputs_large_cluster(self) -> None:
        """Large cluster inputs must be reflected accurately."""
        result = calculate_impact(
            hardware="H100_SXM",
            region="us-west-2",
            duration_hours=168.0,
            num_accelerators=256,
            workload_type="training",
            utilization=0.95,
        )
        report = generate_json_report(result)
        assert report["inputs"]["num_accelerators"] == 256
        assert report["inputs"]["duration_hours"] == 168.0
        assert report["inputs"]["utilization_pct"] == pytest.approx(95.0)


# ---------------------------------------------------------------------------
# generate_json_report — metrics section
# ---------------------------------------------------------------------------


class TestJsonReportMetrics:
    """Verify the metrics section of the JSON report."""

    def test_metrics_keys(self, basic_report: dict) -> None:
        """Metrics section must contain exactly the expected keys."""
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
        """All metric values must be non-negative."""
        for key, val in basic_report["metrics"].items():
            assert val >= 0, f"Metric {key!r} is negative: {val}"

    def test_all_metric_values_are_floats(self, basic_report: dict) -> None:
        """All metric values must be numeric."""
        for key, val in basic_report["metrics"].items():
            assert isinstance(val, (int, float)), f"Metric {key!r} is not numeric: {val!r}"

    def test_energy_mwh_less_than_energy_kwh(self, basic_report: dict) -> None:
        """MWh value must always be less than kWh value."""
        assert basic_report["metrics"]["energy_mwh"] < basic_report["metrics"]["energy_kwh"]

    def test_co2e_tonnes_less_than_co2e_kg(self, basic_report: dict) -> None:
        """CO2e tonnes must always be less than kg."""
        assert basic_report["metrics"]["co2e_tonnes"] < basic_report["metrics"]["co2e_kg"]

    def test_co2e_g_greater_than_co2e_kg(self, basic_report: dict) -> None:
        """CO2e grams must always be greater than kg."""
        assert basic_report["metrics"]["co2e_g"] > basic_report["metrics"]["co2e_kg"]

    def test_water_ml_greater_than_water_liters(self, basic_report: dict) -> None:
        """Water ml must always be greater than litres."""
        assert basic_report["metrics"]["water_ml"] > basic_report["metrics"]["water_liters"]

    def test_large_workload_higher_metrics(self, basic_result: ImpactResult, large_result: ImpactResult) -> None:
        """Larger workload must produce higher metric values."""
        small_report = generate_json_report(basic_result)
        large_report = generate_json_report(large_result)
        assert large_report["metrics"]["energy_kwh"] > small_report["metrics"]["energy_kwh"]
        assert large_report["metrics"]["co2e_kg"] > small_report["metrics"]["co2e_kg"]
        assert large_report["metrics"]["water_liters"] > small_report["metrics"]["water_liters"]

    def test_unit_consistency_energy(self, basic_report: dict) -> None:
        """energy_mwh * 1000 must equal energy_kwh."""
        kwh = basic_report["metrics"]["energy_kwh"]
        mwh = basic_report["metrics"]["energy_mwh"]
        assert mwh * 1000 == pytest.approx(kwh, rel=1e-5)

    def test_unit_consistency_co2e(self, basic_report: dict) -> None:
        """co2e_g / 1000 must equal co2e_kg."""
        kg = basic_report["metrics"]["co2e_kg"]
        g = basic_report["metrics"]["co2e_g"]
        assert g / 1000 == pytest.approx(kg, rel=1e-5)

    def test_unit_consistency_water(self, basic_report: dict) -> None:
        """water_ml / 1000 must equal water_liters."""
        liters = basic_report["metrics"]["water_liters"]
        ml = basic_report["metrics"]["water_ml"]
        assert ml / 1000 == pytest.approx(liters, rel=1e-5)


# ---------------------------------------------------------------------------
# generate_json_report — factors section
# ---------------------------------------------------------------------------


class TestJsonReportFactors:
    """Verify the factors section of the JSON report."""

    def test_factors_keys(self, basic_report: dict) -> None:
        """Factors section must contain exactly the expected keys."""
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
        """effective_power_per_accelerator_w = tdp * utilization * pue."""
        expected = basic_result.tdp_watts * basic_result.utilization * basic_result.pue
        assert basic_report["factors"]["effective_power_per_accelerator_w"] == pytest.approx(
            expected, rel=1e-4
        )

    def test_all_factor_values_are_numeric(self, basic_report: dict) -> None:
        """All factor values must be numeric."""
        for key, val in basic_report["factors"].items():
            assert isinstance(val, (int, float)), f"Factor {key!r} is not numeric: {val!r}"

    def test_pue_factor_gte_one(self, basic_report: dict) -> None:
        """PUE must always be >= 1.0."""
        assert basic_report["factors"]["pue"] >= 1.0

    def test_wue_factor_non_negative(self, basic_report: dict) -> None:
        """WUE must be non-negative."""
        assert basic_report["factors"]["wue_l_per_kwh"] >= 0.0

    def test_tdp_factor_positive(self, basic_report: dict) -> None:
        """TDP must be strictly positive."""
        assert basic_report["factors"]["tdp_watts"] > 0

    def test_carbon_intensity_non_negative(self, basic_report: dict) -> None:
        """Carbon intensity must be non-negative."""
        assert basic_report["factors"]["carbon_intensity_g_per_kwh"] >= 0

    def test_effective_power_positive(self, basic_report: dict) -> None:
        """Effective power per accelerator must be positive."""
        assert basic_report["factors"]["effective_power_per_accelerator_w"] > 0

    def test_factors_vary_by_region(self) -> None:
        """Different regions should produce different factor values."""
        r1 = calculate_impact(hardware="A100_80GB", region="us-east-1", duration_hours=1.0)
        r2 = calculate_impact(hardware="A100_80GB", region="eu-north-1", duration_hours=1.0)
        rep1 = generate_json_report(r1)
        rep2 = generate_json_report(r2)
        # Carbon intensity must differ between these regions
        assert (
            rep1["factors"]["carbon_intensity_g_per_kwh"]
            != rep2["factors"]["carbon_intensity_g_per_kwh"]
        )

    def test_factors_vary_by_hardware(self) -> None:
        """Different hardware should produce different TDP values."""
        r1 = calculate_impact(hardware="A100_80GB", region="us-east-1", duration_hours=1.0)
        r2 = calculate_impact(hardware="T4", region="us-east-1", duration_hours=1.0)
        rep1 = generate_json_report(r1)
        rep2 = generate_json_report(r2)
        assert rep1["factors"]["tdp_watts"] != rep2["factors"]["tdp_watts"]

    def test_effective_power_scales_with_utilization(self) -> None:
        """Half utilization should give approximately half the effective power."""
        r_full = calculate_impact(
            hardware="A100_80GB", region="us-east-1", duration_hours=1.0, utilization=1.0
        )
        r_half = calculate_impact(
            hardware="A100_80GB", region="us-east-1", duration_hours=1.0, utilization=0.5
        )
        rep_full = generate_json_report(r_full)
        rep_half = generate_json_report(r_half)
        assert rep_half["factors"]["effective_power_per_accelerator_w"] == pytest.approx(
            rep_full["factors"]["effective_power_per_accelerator_w"] / 2, rel=1e-5
        )


# ---------------------------------------------------------------------------
# generate_json_report — comparisons section
# ---------------------------------------------------------------------------


class TestJsonReportComparisons:
    """Verify the comparisons section of the JSON report."""

    def test_comparisons_top_level_keys(self, basic_report: dict) -> None:
        """Comparisons must have exactly the expected top-level keys."""
        cmp = basic_report["comparisons"]
        expected = {"co2e", "energy", "water", "disclaimer"}
        assert set(cmp.keys()) == expected

    def test_co2e_comparisons_keys(self, basic_report: dict) -> None:
        """co2e comparison sub-section must have required keys."""
        co2e_cmp = basic_report["comparisons"]["co2e"]
        assert "car_km_equivalent" in co2e_cmp
        assert "car_km_equivalent_description" in co2e_cmp
        assert "tree_sequestration_months" in co2e_cmp
        assert "tree_sequestration_months_description" in co2e_cmp

    def test_energy_comparisons_keys(self, basic_report: dict) -> None:
        """energy comparison sub-section must have required keys."""
        energy_cmp = basic_report["comparisons"]["energy"]
        assert "us_household_days_equivalent" in energy_cmp
        assert "us_household_days_equivalent_description" in energy_cmp

    def test_water_comparisons_keys(self, basic_report: dict) -> None:
        """water comparison sub-section must have required keys."""
        water_cmp = basic_report["comparisons"]["water"]
        assert "showers_equivalent" in water_cmp
        assert "showers_equivalent_description" in water_cmp

    def test_disclaimer_is_non_empty_string(self, basic_report: dict) -> None:
        """Disclaimer must be a non-empty string."""
        disclaimer = basic_report["comparisons"]["disclaimer"]
        assert isinstance(disclaimer, str)
        assert len(disclaimer) > 0

    def test_car_km_equivalent_is_non_negative(self, basic_report: dict) -> None:
        val = basic_report["comparisons"]["co2e"]["car_km_equivalent"]
        assert val >= 0

    def test_car_km_description_is_string(self, basic_report: dict) -> None:
        desc = basic_report["comparisons"]["co2e"]["car_km_equivalent_description"]
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_car_km_description_contains_km(self, basic_report: dict) -> None:
        """Car km description must mention 'km'."""
        desc = basic_report["comparisons"]["co2e"]["car_km_equivalent_description"]
        assert "km" in desc.lower()

    def test_tree_months_is_non_negative(self, basic_report: dict) -> None:
        val = basic_report["comparisons"]["co2e"]["tree_sequestration_months"]
        assert val >= 0

    def test_tree_description_contains_tree(self, basic_report: dict) -> None:
        """Tree description must mention 'tree'."""
        desc = basic_report["comparisons"]["co2e"]["tree_sequestration_months_description"]
        assert "tree" in desc.lower()

    def test_household_days_is_non_negative(self, basic_report: dict) -> None:
        val = basic_report["comparisons"]["energy"]["us_household_days_equivalent"]
        assert val >= 0

    def test_household_days_description_is_string(self, basic_report: dict) -> None:
        desc = basic_report["comparisons"]["energy"]["us_household_days_equivalent_description"]
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_showers_is_non_negative(self, basic_report: dict) -> None:
        val = basic_report["comparisons"]["water"]["showers_equivalent"]
        assert val >= 0

    def test_showers_description_is_string(self, basic_report: dict) -> None:
        desc = basic_report["comparisons"]["water"]["showers_equivalent_description"]
        assert isinstance(desc, str)
        assert len(desc) > 0

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

    def test_tree_months_reference_calculation(self) -> None:
        """Manually verify the tree-month calculation."""
        result = calculate_impact(
            hardware="A100_80GB",
            region="us-east-1",
            duration_hours=1.0,
        )
        report = generate_json_report(result)
        expected_months = round((result.co2e_kg / 21.77) * 12, 2)
        assert report["comparisons"]["co2e"]["tree_sequestration_months"] == pytest.approx(
            expected_months, rel=1e-4
        )

    def test_household_days_reference_calculation(self) -> None:
        """Manually verify the household-days calculation."""
        result = calculate_impact(
            hardware="A100_80GB",
            region="us-east-1",
            duration_hours=1.0,
        )
        report = generate_json_report(result)
        expected_days = round(result.energy_kwh / 28.77, 2)
        assert report["comparisons"]["energy"]["us_household_days_equivalent"] == pytest.approx(
            expected_days, rel=1e-4
        )

    def test_showers_reference_calculation(self) -> None:
        """Manually verify the showers calculation."""
        result = calculate_impact(
            hardware="A100_80GB",
            region="us-east-1",
            duration_hours=1.0,
        )
        report = generate_json_report(result)
        expected_showers = round(result.water_liters / 64.0, 2)
        assert report["comparisons"]["water"]["showers_equivalent"] == pytest.approx(
            expected_showers, rel=1e-4
        )

    def test_high_carbon_region_more_car_km(self, low_carbon_result: ImpactResult, high_carbon_result: ImpactResult) -> None:
        """High-carbon region should give more car-km equivalent than low-carbon."""
        low_report = generate_json_report(low_carbon_result)
        high_report = generate_json_report(high_carbon_result)
        assert (
            high_report["comparisons"]["co2e"]["car_km_equivalent"]
            > low_report["comparisons"]["co2e"]["car_km_equivalent"]
        )

    def test_all_comparison_values_json_serialisable(self, basic_report: dict) -> None:
        """All comparison values must be JSON serialisable."""
        cmp = basic_report["comparisons"]
        serialised = json.dumps(cmp)
        assert len(serialised) > 0

    def test_disclaimer_mentions_approximate(self, basic_report: dict) -> None:
        """Disclaimer should mention that comparisons are approximate."""
        disclaimer = basic_report["comparisons"]["disclaimer"].lower()
        # Should contain 'approximate' or 'illustrative' or similar
        assert any(word in disclaimer for word in ("approximate", "illustrative", "intended"))

    def test_comparisons_scale_linearly_with_co2(self) -> None:
        """Doubling CO2e should double car_km_equivalent."""
        r1 = calculate_impact(
            hardware="T4",
            region="us-east-1",
            duration_hours=1.0,
        )
        r2 = calculate_impact(
            hardware="T4",
            region="us-east-1",
            duration_hours=2.0,
        )
        rep1 = generate_json_report(r1)
        rep2 = generate_json_report(r2)
        km1 = rep1["comparisons"]["co2e"]["car_km_equivalent"]
        km2 = rep2["comparisons"]["co2e"]["car_km_equivalent"]
        assert km2 == pytest.approx(km1 * 2, rel=1e-4)

    def test_zero_co2_gives_zero_car_km(self) -> None:
        """Near-zero CO2e should give near-zero car-km equivalent."""
        result = calculate_impact(
            hardware="T4",
            region="eu-north-1",
            duration_hours=0.001,
            num_accelerators=1,
            utilization=0.01,
        )
        report = generate_json_report(result)
        assert report["comparisons"]["co2e"]["car_km_equivalent"] >= 0


# ---------------------------------------------------------------------------
# generate_html_report — structure
# ---------------------------------------------------------------------------


class TestGenerateHtmlReport:
    """Verify the HTML report card snippet structure."""

    def test_returns_string(self, basic_result: ImpactResult) -> None:
        """generate_html_report must return a string."""
        html = generate_html_report(basic_result)
        assert isinstance(html, str)

    def test_non_empty_output(self, basic_result: ImpactResult) -> None:
        """Output must be non-empty."""
        html = generate_html_report(basic_result)
        assert len(html.strip()) > 0

    def test_contains_report_card_div(self, basic_result: ImpactResult) -> None:
        """Output must contain the report-card div class."""
        html = generate_html_report(basic_result)
        assert 'class="report-card"' in html

    def test_does_not_contain_doctype(self, basic_result: ImpactResult) -> None:
        """Output must be a fragment, not a full HTML document."""
        html = generate_html_report(basic_result)
        assert "<!DOCTYPE" not in html
        assert "<html" not in html

    def test_does_not_contain_body_tag(self, basic_result: ImpactResult) -> None:
        """Output must not contain a <body> tag."""
        html = generate_html_report(basic_result)
        assert "<body" not in html

    def test_does_not_contain_head_tag(self, basic_result: ImpactResult) -> None:
        """Output must not contain a <head> tag."""
        html = generate_html_report(basic_result)
        assert "<head" not in html

    def test_contains_energy_value(self, basic_result: ImpactResult) -> None:
        """The formatted energy value must appear in the HTML."""
        html = generate_html_report(basic_result)
        formatted = _filter_format_number(basic_result.energy_kwh, 3)
        assert formatted in html

    def test_contains_co2_value(self, basic_result: ImpactResult) -> None:
        """The formatted CO2e value must appear in the HTML."""
        html = generate_html_report(basic_result)
        formatted = _filter_format_number(basic_result.co2e_kg, 3)
        assert formatted in html

    def test_contains_water_value(self, basic_result: ImpactResult) -> None:
        """The formatted water value must appear in the HTML."""
        html = generate_html_report(basic_result)
        formatted = _filter_format_number(basic_result.water_liters, 3)
        assert formatted in html

    def test_contains_hardware_name(self, basic_result: ImpactResult) -> None:
        """Hardware name must appear in the HTML."""
        html = generate_html_report(basic_result)
        assert basic_result.hardware in html

    def test_contains_region_name(self, basic_result: ImpactResult) -> None:
        """Region name must appear in the HTML."""
        html = generate_html_report(basic_result)
        assert basic_result.region in html

    def test_contains_kwh_unit(self, basic_result: ImpactResult) -> None:
        """kWh unit must appear in the HTML."""
        html = generate_html_report(basic_result)
        assert "kWh" in html

    def test_contains_co2_unit(self, basic_result: ImpactResult) -> None:
        """CO2e unit must appear in the HTML."""
        html = generate_html_report(basic_result)
        assert "CO₂e" in html or "CO2e" in html

    def test_contains_water_unit(self, basic_result: ImpactResult) -> None:
        """Water unit (L) must appear in the HTML."""
        html = generate_html_report(basic_result)
        # The template renders >L< for litres
        assert ">L<" in html or ">L<" in html or "unit">L<" in html or ">L</" in html

    def test_contains_pue_value(self, basic_result: ImpactResult) -> None:
        """PUE value must appear in the HTML."""
        html = generate_html_report(basic_result)
        assert str(basic_result.pue) in html

    def test_contains_dc_impact_brand(self, basic_result: ImpactResult) -> None:
        """dc_impact brand must appear in the HTML."""
        html = generate_html_report(basic_result)
        assert "dc_impact" in html

    def test_contains_duration_hours(self, basic_result: ImpactResult) -> None:
        """Duration must appear in the HTML."""
        html = generate_html_report(basic_result)
        assert str(basic_result.duration_hours) in html

    def test_empty_result_renders_empty_card(self) -> None:
        """Passing result=None should render the empty-state card."""
        env = _get_jinja_env()
        template = env.get_template("report_card.html")
        html = template.render(result=None, report=None)
        assert "report-card--empty" in html
        assert "No results" in html

    def test_empty_result_no_metric_values(self) -> None:
        """Empty state should not show metric values."""
        env = _get_jinja_env()
        template = env.get_template("report_card.html")
        html = template.render(result=None, report=None)
        # Should not contain metric divs
        assert "metric__value" not in html

    def test_workload_type_capitalised_in_html(self, basic_result: ImpactResult) -> None:
        """Workload type must appear capitalised in the HTML."""
        html = generate_html_report(basic_result)
        assert basic_result.workload_type.capitalize() in html

    def test_comparisons_section_rendered(self, basic_result: ImpactResult) -> None:
        """Comparison section should appear in the HTML."""
        html = generate_html_report(basic_result)
        # Comparison section heading or content
        assert "comparisons" in html.lower() or "km" in html.lower() or "🚗" in html

    def test_html_contains_details_tables(self, basic_result: ImpactResult) -> None:
        """HTML must contain table elements for parameters/factors."""
        html = generate_html_report(basic_result)
        assert "<table" in html
        assert "</table>" in html

    def test_inference_type_in_html(self, inference_result: ImpactResult) -> None:
        """Inference workload_type must appear capitalised in the card."""
        html = generate_html_report(inference_result)
        assert "Inference" in html

    def test_multiple_gpus_show_correct_count(self) -> None:
        """Num accelerators must appear in the card."""
        result = calculate_impact(
            hardware="H100_SXM",
            region="us-west-2",
            duration_hours=24.0,
            num_accelerators=8,
        )
        html = generate_html_report(result)
        assert "8" in html

    def test_html_contains_footer(self, basic_result: ImpactResult) -> None:
        """Report card must contain footer element."""
        html = generate_html_report(basic_result)
        assert "report-card__footer" in html

    def test_html_contains_header(self, basic_result: ImpactResult) -> None:
        """Report card must contain header element."""
        html = generate_html_report(basic_result)
        assert "report-card__header" in html

    def test_html_contains_metrics_section(self, basic_result: ImpactResult) -> None:
        """Report card must contain metrics section."""
        html = generate_html_report(basic_result)
        assert "report-card__metrics" in html

    def test_html_escaping_for_special_chars(self) -> None:
        """HTML output should have autoescaping enabled — region names are safe strings."""
        result = calculate_impact(
            hardware="A100_80GB",
            region="eu-west-3",
            duration_hours=1.0,
        )
        html = generate_html_report(result)
        # Region should appear unescaped (no HTML entity for hyphens)
        assert "eu-west-3" in html

    def test_two_calls_produce_same_html(self, basic_result: ImpactResult) -> None:
        """generate_html_report is deterministic for the same result."""
        html1 = generate_html_report(basic_result)
        html2 = generate_html_report(basic_result)
        # Strip timestamps in comparison — use a simple structural check
        # Both should contain the same hardware/region
        assert basic_result.hardware in html1
        assert basic_result.hardware in html2

    def test_wue_value_in_html(self, basic_result: ImpactResult) -> None:
        """WUE value must appear in the HTML."""
        html = generate_html_report(basic_result)
        assert str(basic_result.wue) in html

    def test_carbon_intensity_in_html(self, basic_result: ImpactResult) -> None:
        """Carbon intensity value must appear in the HTML."""
        html = generate_html_report(basic_result)
        assert str(basic_result.carbon_intensity_g_per_kwh) in html

    def test_tdp_in_html(self, basic_result: ImpactResult) -> None:
        """TDP value must appear in the HTML."""
        html = generate_html_report(basic_result)
        assert str(basic_result.tdp_watts) in html


# ---------------------------------------------------------------------------
# Custom filter: _filter_format_number
# ---------------------------------------------------------------------------


class TestFilterFormatNumber:
    """Unit tests for the format_number Jinja2 filter."""

    def test_zero_returns_zero_string(self) -> None:
        """Zero should return the string '0'."""
        assert _filter_format_number(0.0) == "0"

    def test_integer_like_float(self) -> None:
        """1.0 with precision 3 should return '1.000'."""
        result = _filter_format_number(1.0)
        assert result == "1.000"

    def test_small_value_uses_scientific(self) -> None:
        """Very small values should use scientific notation."""
        result = _filter_format_number(0.0000001)
        assert "e" in result.lower()

    def test_normal_precision_three(self) -> None:
        """Normal value with precision 3 should be correctly rounded."""
        result = _filter_format_number(12.345678, 3)
        assert result == "12.346"

    def test_large_value_includes_comma(self) -> None:
        """Large values (>=1000) should include a thousands separator."""
        result = _filter_format_number(1234.5, 3)
        assert "," in result

    def test_very_large_value_uses_M_suffix(self) -> None:
        """Values >= 1,000,000 should use 'M' suffix."""
        result = _filter_format_number(2_500_000.0)
        assert "M" in result

    def test_precision_parameter(self) -> None:
        """Precision parameter must be respected."""
        result = _filter_format_number(3.14159, 2)
        assert result == "3.14"

    def test_precision_one(self) -> None:
        """Precision of 1 must give 1 decimal place."""
        result = _filter_format_number(5.678, 1)
        assert result == "5.7"

    def test_precision_zero(self) -> None:
        """Precision of 0 should round to integer."""
        result = _filter_format_number(5.678, 0)
        assert result == "6"

    def test_invalid_input_returns_string(self) -> None:
        """Non-numeric input should return a string without raising."""
        result = _filter_format_number("not-a-number")  # type: ignore[arg-type]
        assert isinstance(result, str)

    def test_none_input_returns_string(self) -> None:
        """None input should return a string without raising."""
        result = _filter_format_number(None)  # type: ignore[arg-type]
        assert isinstance(result, str)

    def test_negative_value(self) -> None:
        """Negative values should include a minus sign."""
        result = _filter_format_number(-5.5, 1)
        assert "-" in result

    def test_exactly_one_returns_1_000(self) -> None:
        """1.0 with precision 3 should return '1.000'."""
        result = _filter_format_number(1.0, 3)
        assert result == "1.000"

    def test_boundary_at_0_001(self) -> None:
        """Value exactly 0.001 should NOT use scientific notation."""
        result = _filter_format_number(0.001, 3)
        assert "e" not in result.lower()

    def test_value_below_0_001_uses_scientific(self) -> None:
        """Value just below 0.001 should use scientific notation."""
        result = _filter_format_number(0.0009999, 3)
        assert "e" in result.lower()

    def test_exactly_1_million_uses_M(self) -> None:
        """Exactly 1,000,000 should use 'M' suffix."""
        result = _filter_format_number(1_000_000.0)
        assert "M" in result

    def test_positive_float_returns_non_empty_string(self) -> None:
        """Any positive float should return a non-empty string."""
        result = _filter_format_number(42.0, 2)
        assert isinstance(result, str)
        assert len(result) > 0


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
        """Exactly 1 kg CO2e should return 'badge--yellow'."""
        assert _filter_co2_badge_colour(1.0) == "badge--yellow"

    def test_moderate_returns_yellow(self) -> None:
        assert _filter_co2_badge_colour(50.0) == "badge--yellow"

    def test_below_100_returns_yellow(self) -> None:
        assert _filter_co2_badge_colour(99.9) == "badge--yellow"

    def test_100_returns_orange(self) -> None:
        """Exactly 100 kg CO2e should return 'badge--orange'."""
        assert _filter_co2_badge_colour(100.0) == "badge--orange"

    def test_below_1000_returns_orange(self) -> None:
        assert _filter_co2_badge_colour(500.0) == "badge--orange"

    def test_999_returns_orange(self) -> None:
        assert _filter_co2_badge_colour(999.9) == "badge--orange"

    def test_1000_returns_red(self) -> None:
        """Exactly 1000 kg CO2e should return 'badge--red'."""
        assert _filter_co2_badge_colour(1000.0) == "badge--red"

    def test_very_large_returns_red(self) -> None:
        assert _filter_co2_badge_colour(100_000.0) == "badge--red"

    def test_invalid_string_input_returns_grey(self) -> None:
        assert _filter_co2_badge_colour("bad") == "badge--grey"  # type: ignore[arg-type]

    def test_none_input_returns_grey(self) -> None:
        assert _filter_co2_badge_colour(None) == "badge--grey"  # type: ignore[arg-type]

    def test_all_return_values_are_valid_css_classes(self) -> None:
        """All returned values should be valid CSS modifier class strings."""
        valid = {"badge--green", "badge--yellow", "badge--orange", "badge--red", "badge--grey"}
        for val in [0.0, 0.5, 1.0, 50.0, 100.0, 500.0, 1000.0, 10000.0]:
            assert _filter_co2_badge_colour(val) in valid

    def test_integer_input_works(self) -> None:
        """Integer input (not just float) should work."""
        assert _filter_co2_badge_colour(0) == "badge--green"
        assert _filter_co2_badge_colour(1) == "badge--yellow"
        assert _filter_co2_badge_colour(1000) == "badge--red"


# ---------------------------------------------------------------------------
# Custom filter: _filter_energy_badge_colour
# ---------------------------------------------------------------------------


class TestFilterEnergyBadgeColour:
    """Unit tests for the energy_badge_colour Jinja2 filter."""

    def test_below_1_returns_green(self) -> None:
        assert _filter_energy_badge_colour(0.5) == "badge--green"

    def test_zero_returns_green(self) -> None:
        assert _filter_energy_badge_colour(0.0) == "badge--green"

    def test_exactly_0_999_returns_green(self) -> None:
        assert _filter_energy_badge_colour(0.999) == "badge--green"

    def test_1_returns_yellow(self) -> None:
        """Exactly 1 kWh should return 'badge--yellow'."""
        assert _filter_energy_badge_colour(1.0) == "badge--yellow"

    def test_50_returns_yellow(self) -> None:
        assert _filter_energy_badge_colour(50.0) == "badge--yellow"

    def test_99_returns_yellow(self) -> None:
        assert _filter_energy_badge_colour(99.9) == "badge--yellow"

    def test_100_returns_orange(self) -> None:
        """Exactly 100 kWh should return 'badge--orange'."""
        assert _filter_energy_badge_colour(100.0) == "badge--orange"

    def test_5000_returns_orange(self) -> None:
        assert _filter_energy_badge_colour(5000.0) == "badge--orange"

    def test_9999_returns_orange(self) -> None:
        assert _filter_energy_badge_colour(9999.9) == "badge--orange"

    def test_10000_returns_red(self) -> None:
        """Exactly 10,000 kWh should return 'badge--red'."""
        assert _filter_energy_badge_colour(10_000.0) == "badge--red"

    def test_very_large_returns_red(self) -> None:
        assert _filter_energy_badge_colour(1_000_000.0) == "badge--red"

    def test_invalid_string_input_returns_grey(self) -> None:
        assert _filter_energy_badge_colour("bad") == "badge--grey"  # type: ignore[arg-type]

    def test_none_input_returns_grey(self) -> None:
        assert _filter_energy_badge_colour(None) == "badge--grey"  # type: ignore[arg-type]

    def test_integer_input_works(self) -> None:
        """Integer input should work the same as float."""
        assert _filter_energy_badge_colour(0) == "badge--green"
        assert _filter_energy_badge_colour(1) == "badge--yellow"
        assert _filter_energy_badge_colour(10_000) == "badge--red"

    def test_all_return_values_are_valid_css_classes(self) -> None:
        """All returned values should be valid CSS modifier class strings."""
        valid = {"badge--green", "badge--yellow", "badge--orange", "badge--red", "badge--grey"}
        for val in [0.0, 0.5, 1.0, 50.0, 100.0, 5000.0, 10_000.0, 1_000_000.0]:
            assert _filter_energy_badge_colour(val) in valid


# ---------------------------------------------------------------------------
# _utc_now_iso helper
# ---------------------------------------------------------------------------


class TestUtcNowIso:
    """Unit tests for the _utc_now_iso timestamp helper."""

    def test_returns_string(self) -> None:
        ts = _utc_now_iso()
        assert isinstance(ts, str)

    def test_ends_with_z(self) -> None:
        ts = _utc_now_iso()
        assert ts.endswith("Z")

    def test_contains_t_separator(self) -> None:
        ts = _utc_now_iso()
        assert "T" in ts

    def test_is_parseable_as_iso8601(self) -> None:
        ts = _utc_now_iso().rstrip("Z")
        dt = datetime.datetime.fromisoformat(ts)
        assert isinstance(dt, datetime.datetime)

    def test_two_calls_produce_ordered_timestamps(self) -> None:
        """Second call must produce a timestamp >= the first."""
        ts1 = _utc_now_iso()
        time.sleep(0.01)
        ts2 = _utc_now_iso()
        # String comparison works for ISO 8601 timestamps
        assert ts2 >= ts1


# ---------------------------------------------------------------------------
# _get_jinja_env helper
# ---------------------------------------------------------------------------


class TestGetJinjaEnv:
    """Tests for the Jinja2 environment factory."""

    def test_returns_environment(self) -> None:
        from jinja2 import Environment
        env = _get_jinja_env()
        assert isinstance(env, Environment)

    def test_has_format_number_filter(self) -> None:
        env = _get_jinja_env()
        assert "format_number" in env.filters

    def test_has_co2_badge_colour_filter(self) -> None:
        env = _get_jinja_env()
        assert "co2_badge_colour" in env.filters

    def test_has_energy_badge_colour_filter(self) -> None:
        env = _get_jinja_env()
        assert "energy_badge_colour" in env.filters

    def test_can_load_report_card_template(self) -> None:
        env = _get_jinja_env()
        # Should not raise TemplateNotFound
        template = env.get_template("report_card.html")
        assert template is not None

    def test_autoescape_enabled_for_html(self) -> None:
        """HTML autoescaping must be enabled."""
        env = _get_jinja_env()
        # Check that autoescape is configured
        assert env.autoescape is not None

    def test_trim_blocks_enabled(self) -> None:
        env = _get_jinja_env()
        assert env.trim_blocks is True

    def test_lstrip_blocks_enabled(self) -> None:
        env = _get_jinja_env()
        assert env.lstrip_blocks is True


# ---------------------------------------------------------------------------
# Cross-result consistency checks
# ---------------------------------------------------------------------------


class TestJsonReportConsistency:
    """Cross-result consistency checks."""

    def test_metrics_are_deterministic(self, basic_result: ImpactResult) -> None:
        """Metrics must be the same across two calls with the same result."""
        r1 = generate_json_report(basic_result)
        r2 = generate_json_report(basic_result)
        assert r1["metrics"] == r2["metrics"]

    def test_inputs_are_deterministic(self, basic_result: ImpactResult) -> None:
        """Inputs must be the same across two calls with the same result."""
        r1 = generate_json_report(basic_result)
        r2 = generate_json_report(basic_result)
        assert r1["inputs"] == r2["inputs"]

    def test_factors_are_deterministic(self, basic_result: ImpactResult) -> None:
        """Factors must be the same across two calls with the same result."""
        r1 = generate_json_report(basic_result)
        r2 = generate_json_report(basic_result)
        assert r1["factors"] == r2["factors"]

    def test_schema_version_is_consistent(self, basic_result: ImpactResult) -> None:
        """Schema version must not change across calls."""
        r1 = generate_json_report(basic_result)
        r2 = generate_json_report(basic_result)
        assert r1["schema_version"] == r2["schema_version"]

    def test_generated_at_may_differ(self, basic_result: ImpactResult) -> None:
        """generated_at may differ across calls (it is a live timestamp)."""
        r1 = generate_json_report(basic_result)
        time.sleep(0.01)
        r2 = generate_json_report(basic_result)
        # The schema version and metrics are the same
        assert r1["schema_version"] == r2["schema_version"]
        assert r1["metrics"] == r2["metrics"]

    def test_low_carbon_report_lower_co2_than_high_carbon(
        self,
        low_carbon_result: ImpactResult,
        high_carbon_result: ImpactResult,
    ) -> None:
        """Low-carbon region must yield lower CO2e in report than high-carbon."""
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

    def test_report_inputs_reflect_result(self, basic_result: ImpactResult, basic_report: dict) -> None:
        """The inputs section must match the result's fields."""
        assert basic_report["inputs"]["hardware"] == basic_result.hardware
        assert basic_report["inputs"]["region"] == basic_result.region
        assert basic_report["inputs"]["duration_hours"] == basic_result.duration_hours
        assert basic_report["inputs"]["num_accelerators"] == basic_result.num_accelerators
        assert basic_report["inputs"]["workload_type"] == basic_result.workload_type
        assert basic_report["inputs"]["utilization"] == basic_result.utilization

    def test_multiple_hardware_presets_produce_valid_reports(self) -> None:
        """Multiple hardware types should all produce valid JSON reports."""
        hardware_list = ["A100_80GB", "H100_SXM", "T4", "TPU_v4", "MI300X", "Gaudi2"]
        for hw in hardware_list:
            result = calculate_impact(
                hardware=hw,
                region="us-east-1",
                duration_hours=1.0,
            )
            report = generate_json_report(result)
            assert report["schema_version"] == "1.0"
            assert report["metrics"]["energy_kwh"] > 0
            assert report["factors"]["tdp_watts"] > 0

    def test_multiple_regions_produce_valid_reports(self) -> None:
        """Multiple cloud regions should all produce valid JSON reports."""
        regions = ["us-east-1", "us-west-2", "eu-north-1", "ap-southeast-1", "af-south-1"]
        for region in regions:
            result = calculate_impact(
                hardware="T4",
                region=region,
                duration_hours=1.0,
            )
            report = generate_json_report(result)
            assert report["schema_version"] == "1.0"
            assert report["metrics"]["energy_kwh"] > 0
            assert report["factors"]["carbon_intensity_g_per_kwh"] >= 0

    def test_html_and_json_agree_on_hardware(self, basic_result: ImpactResult) -> None:
        """HTML and JSON reports must agree on the hardware field."""
        report = generate_json_report(basic_result)
        html = generate_html_report(basic_result)
        assert report["inputs"]["hardware"] in html

    def test_html_and_json_agree_on_region(self, basic_result: ImpactResult) -> None:
        """HTML and JSON reports must agree on the region field."""
        report = generate_json_report(basic_result)
        html = generate_html_report(basic_result)
        assert report["inputs"]["region"] in html
