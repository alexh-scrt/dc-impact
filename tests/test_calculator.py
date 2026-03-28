"""Unit tests for dc_impact.calculator — core computation engine.

Covers:
- :func:`~dc_impact.calculator.compute_energy_kwh` with known reference values.
- :func:`~dc_impact.calculator.compute_co2e_kg` with known reference values.
- :func:`~dc_impact.calculator.compute_water_liters` with known reference values.
- :func:`~dc_impact.calculator.calculate_impact` end-to-end happy path.
- Input validation via :func:`~dc_impact.calculator._validate_inputs`
  (exercised through ``calculate_impact``).
- Lookup failures for unknown hardware / region keys.
- :class:`~dc_impact.calculator.ImpactResult` serialisation via ``to_dict``.
- :func:`~dc_impact.calculator.get_available_hardware` and
  :func:`~dc_impact.calculator.get_available_regions` helpers.
- PUE/WUE fallback behaviour for regions not in the lookup tables.
- Private lookup helper functions.
- Reference calculations against order-of-magnitude literature estimates.
"""

from __future__ import annotations

import json
import math

import pytest

from dc_impact.calculator import (
    CalculatorError,
    ImpactResult,
    calculate_impact,
    compute_co2e_kg,
    compute_energy_kwh,
    compute_water_liters,
    get_available_hardware,
    get_available_regions,
    _lookup_pue,
    _lookup_wue,
    _lookup_hardware_tdp,
    _lookup_carbon_intensity,
)
from dc_impact import data


# ---------------------------------------------------------------------------
# compute_energy_kwh
# ---------------------------------------------------------------------------


class TestComputeEnergyKwh:
    """Tests for the energy calculation helper."""

    def test_single_accelerator_one_hour_pue_one(self) -> None:
        """Single 400 W device at 100 % for 1 h with no PUE overhead => 0.4 kWh."""
        result = compute_energy_kwh(
            tdp_watts=400.0,
            num_accelerators=1,
            utilization=1.0,
            duration_hours=1.0,
            pue=1.0,
        )
        assert result == pytest.approx(0.4)

    def test_pue_scales_linearly(self) -> None:
        """PUE of 1.2 should give exactly 1.2x the energy vs PUE of 1.0."""
        base = compute_energy_kwh(
            tdp_watts=400.0,
            num_accelerators=1,
            utilization=1.0,
            duration_hours=1.0,
            pue=1.0,
        )
        scaled = compute_energy_kwh(
            tdp_watts=400.0,
            num_accelerators=1,
            utilization=1.0,
            duration_hours=1.0,
            pue=1.2,
        )
        assert scaled == pytest.approx(base * 1.2)

    def test_multiple_accelerators(self) -> None:
        """8x A100s (400 W each) at 100 % for 72 h with PUE 1.2."""
        result = compute_energy_kwh(
            tdp_watts=400.0,
            num_accelerators=8,
            utilization=1.0,
            duration_hours=72.0,
            pue=1.2,
        )
        # 8 * 400 / 1000 * 72 * 1.2 = 3.2 * 72 * 1.2 = 276.48
        assert result == pytest.approx(276.48)

    def test_utilization_scales_power(self) -> None:
        """50 % utilisation should halve the energy vs 100 %."""
        full = compute_energy_kwh(
            tdp_watts=700.0,
            num_accelerators=1,
            utilization=1.0,
            duration_hours=10.0,
            pue=1.1,
        )
        half = compute_energy_kwh(
            tdp_watts=700.0,
            num_accelerators=1,
            utilization=0.5,
            duration_hours=10.0,
            pue=1.1,
        )
        assert half == pytest.approx(full / 2)

    def test_small_inference_workload(self) -> None:
        """T4 GPU (70 W) at 80 % utilisation for 0.5 h with PUE 1.18."""
        result = compute_energy_kwh(
            tdp_watts=70.0,
            num_accelerators=1,
            utilization=0.8,
            duration_hours=0.5,
            pue=1.18,
        )
        # 70 * 0.8 / 1000 * 0.5 * 1.18 = 0.056 * 0.5 * 1.18 = 0.03304
        assert result == pytest.approx(0.03304)

    def test_result_is_positive_float(self) -> None:
        """Result is always a positive float."""
        result = compute_energy_kwh(
            tdp_watts=192.0,
            num_accelerators=4,
            utilization=0.9,
            duration_hours=24.0,
            pue=1.09,
        )
        assert isinstance(result, float)
        assert result > 0

    def test_very_small_duration(self) -> None:
        """Very small duration produces a very small but positive result."""
        result = compute_energy_kwh(
            tdp_watts=400.0,
            num_accelerators=1,
            utilization=1.0,
            duration_hours=0.001,
            pue=1.0,
        )
        assert result > 0
        assert result < 1.0

    def test_large_cluster(self) -> None:
        """1024 GPUs at full power for 100 h should give a large but finite result."""
        result = compute_energy_kwh(
            tdp_watts=400.0,
            num_accelerators=1024,
            utilization=1.0,
            duration_hours=100.0,
            pue=1.2,
        )
        # 1024 * 400 / 1000 * 100 * 1.2 = 409.6 * 100 * 1.2 = 49152 kWh
        assert result == pytest.approx(49152.0)

    def test_formula_correctness(self) -> None:
        """Verify formula: (tdp * n * util / 1000) * hours * pue."""
        tdp = 300.0
        n = 4
        util = 0.75
        hours = 12.0
        pue = 1.15
        expected = (tdp * n * util / 1000.0) * hours * pue
        result = compute_energy_kwh(
            tdp_watts=tdp,
            num_accelerators=n,
            utilization=util,
            duration_hours=hours,
            pue=pue,
        )
        assert result == pytest.approx(expected)

    # --- validation ---

    def test_raises_on_zero_tdp(self) -> None:
        with pytest.raises(CalculatorError, match="tdp_watts"):
            compute_energy_kwh(
                tdp_watts=0.0,
                num_accelerators=1,
                utilization=1.0,
                duration_hours=1.0,
                pue=1.2,
            )

    def test_raises_on_negative_tdp(self) -> None:
        with pytest.raises(CalculatorError, match="tdp_watts"):
            compute_energy_kwh(
                tdp_watts=-100.0,
                num_accelerators=1,
                utilization=1.0,
                duration_hours=1.0,
                pue=1.2,
            )

    def test_raises_on_zero_accelerators(self) -> None:
        with pytest.raises(CalculatorError, match="num_accelerators"):
            compute_energy_kwh(
                tdp_watts=400.0,
                num_accelerators=0,
                utilization=1.0,
                duration_hours=1.0,
                pue=1.2,
            )

    def test_raises_on_negative_accelerators(self) -> None:
        with pytest.raises(CalculatorError, match="num_accelerators"):
            compute_energy_kwh(
                tdp_watts=400.0,
                num_accelerators=-1,
                utilization=1.0,
                duration_hours=1.0,
                pue=1.2,
            )

    def test_raises_on_zero_utilization(self) -> None:
        with pytest.raises(CalculatorError, match="utilization"):
            compute_energy_kwh(
                tdp_watts=400.0,
                num_accelerators=1,
                utilization=0.0,
                duration_hours=1.0,
                pue=1.2,
            )

    def test_raises_on_utilization_above_one(self) -> None:
        with pytest.raises(CalculatorError, match="utilization"):
            compute_energy_kwh(
                tdp_watts=400.0,
                num_accelerators=1,
                utilization=1.01,
                duration_hours=1.0,
                pue=1.2,
            )

    def test_raises_on_negative_utilization(self) -> None:
        with pytest.raises(CalculatorError, match="utilization"):
            compute_energy_kwh(
                tdp_watts=400.0,
                num_accelerators=1,
                utilization=-0.5,
                duration_hours=1.0,
                pue=1.2,
            )

    def test_raises_on_zero_duration(self) -> None:
        with pytest.raises(CalculatorError, match="duration_hours"):
            compute_energy_kwh(
                tdp_watts=400.0,
                num_accelerators=1,
                utilization=1.0,
                duration_hours=0.0,
                pue=1.2,
            )

    def test_raises_on_negative_duration(self) -> None:
        with pytest.raises(CalculatorError, match="duration_hours"):
            compute_energy_kwh(
                tdp_watts=400.0,
                num_accelerators=1,
                utilization=1.0,
                duration_hours=-5.0,
                pue=1.2,
            )

    def test_raises_on_pue_below_one(self) -> None:
        with pytest.raises(CalculatorError, match="pue"):
            compute_energy_kwh(
                tdp_watts=400.0,
                num_accelerators=1,
                utilization=1.0,
                duration_hours=1.0,
                pue=0.99,
            )

    def test_raises_on_pue_zero(self) -> None:
        with pytest.raises(CalculatorError, match="pue"):
            compute_energy_kwh(
                tdp_watts=400.0,
                num_accelerators=1,
                utilization=1.0,
                duration_hours=1.0,
                pue=0.0,
            )

    def test_pue_exactly_one_accepted(self) -> None:
        """PUE of exactly 1.0 is physically meaningful and must be accepted."""
        result = compute_energy_kwh(
            tdp_watts=400.0,
            num_accelerators=1,
            utilization=1.0,
            duration_hours=1.0,
            pue=1.0,
        )
        assert result == pytest.approx(0.4)

    def test_utilization_exactly_one_accepted(self) -> None:
        """Utilisation of exactly 1.0 must be accepted."""
        result = compute_energy_kwh(
            tdp_watts=400.0,
            num_accelerators=1,
            utilization=1.0,
            duration_hours=1.0,
            pue=1.2,
        )
        assert result == pytest.approx(0.48)

    def test_num_accelerators_one_accepted(self) -> None:
        """num_accelerators=1 is the minimum valid value."""
        result = compute_energy_kwh(
            tdp_watts=100.0,
            num_accelerators=1,
            utilization=1.0,
            duration_hours=1.0,
            pue=1.0,
        )
        assert result == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# compute_co2e_kg
# ---------------------------------------------------------------------------


class TestComputeCo2eKg:
    """Tests for the CO2e conversion helper."""

    def test_reference_value(self) -> None:
        """100 kWh at 415 gCO2e/kWh => 41.5 kg."""
        result = compute_co2e_kg(
            energy_kwh=100.0,
            carbon_intensity_g_per_kwh=415.0,
        )
        assert result == pytest.approx(41.5)

    def test_zero_energy_gives_zero_emissions(self) -> None:
        result = compute_co2e_kg(energy_kwh=0.0, carbon_intensity_g_per_kwh=415.0)
        assert result == pytest.approx(0.0)

    def test_zero_intensity_gives_zero_emissions(self) -> None:
        """A zero-carbon grid should yield zero emissions regardless of energy."""
        result = compute_co2e_kg(energy_kwh=100.0, carbon_intensity_g_per_kwh=0.0)
        assert result == pytest.approx(0.0)

    def test_very_low_carbon_region(self) -> None:
        """Simulate EU-north-1 Stockholm (8 gCO2/kWh) at 50 kWh."""
        result = compute_co2e_kg(
            energy_kwh=50.0,
            carbon_intensity_g_per_kwh=8.0,
        )
        assert result == pytest.approx(0.4)

    def test_high_carbon_region(self) -> None:
        """Simulate South Africa (928 gCO2/kWh) at 100 kWh."""
        result = compute_co2e_kg(
            energy_kwh=100.0,
            carbon_intensity_g_per_kwh=928.0,
        )
        assert result == pytest.approx(92.8)

    def test_units_conversion_grams_to_kg(self) -> None:
        """Confirm the /1000 conversion from grams to kilograms."""
        result = compute_co2e_kg(
            energy_kwh=1.0,
            carbon_intensity_g_per_kwh=1000.0,
        )
        assert result == pytest.approx(1.0)

    def test_result_is_non_negative(self) -> None:
        result = compute_co2e_kg(
            energy_kwh=276.48,
            carbon_intensity_g_per_kwh=136.0,
        )
        assert result >= 0

    def test_formula_correctness(self) -> None:
        """co2e_kg = energy_kwh * intensity_g_per_kwh / 1000."""
        energy = 42.0
        intensity = 337.0
        expected = energy * intensity / 1000.0
        result = compute_co2e_kg(
            energy_kwh=energy,
            carbon_intensity_g_per_kwh=intensity,
        )
        assert result == pytest.approx(expected)

    def test_large_energy_value(self) -> None:
        """Very large energy values should work without overflow."""
        result = compute_co2e_kg(
            energy_kwh=1_000_000.0,
            carbon_intensity_g_per_kwh=500.0,
        )
        assert result == pytest.approx(500_000.0)

    # --- validation ---

    def test_raises_on_negative_energy(self) -> None:
        with pytest.raises(CalculatorError, match="energy_kwh"):
            compute_co2e_kg(energy_kwh=-1.0, carbon_intensity_g_per_kwh=415.0)

    def test_raises_on_negative_intensity(self) -> None:
        with pytest.raises(CalculatorError, match="carbon_intensity"):
            compute_co2e_kg(energy_kwh=100.0, carbon_intensity_g_per_kwh=-10.0)


# ---------------------------------------------------------------------------
# compute_water_liters
# ---------------------------------------------------------------------------


class TestComputeWaterLiters:
    """Tests for the water consumption helper."""

    def test_reference_value_oregon(self) -> None:
        """276.48 kWh at WUE 0.18 => ~49.77 L."""
        result = compute_water_liters(energy_kwh=276.48, wue=0.18)
        assert result == pytest.approx(49.7664)

    def test_zero_wue(self) -> None:
        """WUE of 0 (waterless cooling) should give 0 litres."""
        result = compute_water_liters(energy_kwh=100.0, wue=0.0)
        assert result == pytest.approx(0.0)

    def test_zero_energy(self) -> None:
        """Zero energy should give zero water consumption."""
        result = compute_water_liters(energy_kwh=0.0, wue=1.5)
        assert result == pytest.approx(0.0)

    def test_high_wue_hot_region(self) -> None:
        """UAE region WUE 2.5 at 100 kWh => 250 L."""
        result = compute_water_liters(energy_kwh=100.0, wue=2.5)
        assert result == pytest.approx(250.0)

    def test_result_is_non_negative(self) -> None:
        result = compute_water_liters(energy_kwh=50.0, wue=0.9)
        assert result >= 0

    def test_linearity_with_energy(self) -> None:
        """Doubling energy should double water consumption."""
        single = compute_water_liters(energy_kwh=100.0, wue=0.49)
        double = compute_water_liters(energy_kwh=200.0, wue=0.49)
        assert double == pytest.approx(single * 2)

    def test_linearity_with_wue(self) -> None:
        """Doubling WUE should double water consumption."""
        low = compute_water_liters(energy_kwh=100.0, wue=0.5)
        high = compute_water_liters(energy_kwh=100.0, wue=1.0)
        assert high == pytest.approx(low * 2)

    def test_formula_correctness(self) -> None:
        """water_liters = energy_kwh * wue."""
        energy = 55.5
        wue = 1.32
        expected = energy * wue
        result = compute_water_liters(energy_kwh=energy, wue=wue)
        assert result == pytest.approx(expected)

    def test_result_is_float(self) -> None:
        result = compute_water_liters(energy_kwh=10.0, wue=0.5)
        assert isinstance(result, float)

    # --- validation ---

    def test_raises_on_negative_energy(self) -> None:
        with pytest.raises(CalculatorError, match="energy_kwh"):
            compute_water_liters(energy_kwh=-0.1, wue=1.0)

    def test_raises_on_negative_wue(self) -> None:
        with pytest.raises(CalculatorError, match="wue"):
            compute_water_liters(energy_kwh=100.0, wue=-0.5)


# ---------------------------------------------------------------------------
# calculate_impact — end-to-end tests
# ---------------------------------------------------------------------------


class TestCalculateImpact:
    """End-to-end tests for the primary public API."""

    def test_basic_training_workload(self) -> None:
        """8x A100_80GB for 72 h at 100 % in us-west-2 (Oregon, low-carbon)."""
        result = calculate_impact(
            hardware="A100_80GB",
            region="us-west-2",
            duration_hours=72.0,
            num_accelerators=8,
            workload_type="training",
            utilization=1.0,
        )
        assert isinstance(result, ImpactResult)
        # energy: 8 * 400 / 1000 * 72 * 1.20 = 276.48 kWh
        assert result.energy_kwh == pytest.approx(276.48, rel=1e-4)
        # co2e: 276.48 * 136 / 1000 = 37.6013 kg
        assert result.co2e_kg == pytest.approx(37.6013, rel=1e-3)
        # water: 276.48 * 0.18 = 49.7664 L
        assert result.water_liters == pytest.approx(49.7664, rel=1e-4)

    def test_inference_workload_t4(self) -> None:
        """Single T4 at 80 % utilisation for 1 h in us-east-1."""
        result = calculate_impact(
            hardware="T4",
            region="us-east-1",
            duration_hours=1.0,
            num_accelerators=1,
            workload_type="inference",
            utilization=0.8,
        )
        # energy: 1 * 70 * 0.8 / 1000 * 1.0 * 1.20 = 0.0672 kWh
        assert result.energy_kwh == pytest.approx(0.0672, rel=1e-4)
        # co2e: 0.0672 * 415 / 1000 = 0.027888 kg
        assert result.co2e_kg == pytest.approx(0.027888, rel=1e-3)
        assert result.workload_type == "inference"

    def test_result_fields_match_inputs(self) -> None:
        """ImpactResult fields must reflect the inputs exactly."""
        result = calculate_impact(
            hardware="H100_SXM",
            region="europe-west4",
            duration_hours=48.0,
            num_accelerators=4,
            workload_type="training",
            utilization=0.9,
        )
        assert result.hardware == "H100_SXM"
        assert result.region == "europe-west4"
        assert result.duration_hours == 48.0
        assert result.num_accelerators == 4
        assert result.workload_type == "training"
        assert result.utilization == 0.9

    def test_result_lookup_factors_match_data_tables(self) -> None:
        """PUE, WUE, TDP, and carbon intensity must come from the data tables."""
        hw = "A100_80GB"
        region = "us-east-1"
        result = calculate_impact(
            hardware=hw,
            region=region,
            duration_hours=1.0,
            num_accelerators=1,
            workload_type="training",
            utilization=1.0,
        )
        assert result.tdp_watts == data.HARDWARE_TDP_WATTS[hw]
        assert result.pue == data.PUE_BY_REGION[region]
        assert result.wue == data.WUE_BY_REGION[region]
        assert result.carbon_intensity_g_per_kwh == data.CARBON_INTENSITY_G_PER_KWH[region]

    def test_result_is_frozen(self) -> None:
        """ImpactResult must be immutable."""
        result = calculate_impact(
            hardware="T4",
            region="us-east-1",
            duration_hours=1.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.energy_kwh = 999.0  # type: ignore[misc]

    def test_default_num_accelerators_is_one(self) -> None:
        single = calculate_impact(
            hardware="A100_80GB",
            region="us-east-1",
            duration_hours=1.0,
        )
        explicit = calculate_impact(
            hardware="A100_80GB",
            region="us-east-1",
            duration_hours=1.0,
            num_accelerators=1,
        )
        assert single.energy_kwh == pytest.approx(explicit.energy_kwh)

    def test_default_workload_type_is_training(self) -> None:
        result = calculate_impact(
            hardware="A100_80GB",
            region="us-east-1",
            duration_hours=1.0,
        )
        assert result.workload_type == "training"

    def test_default_utilization_is_one(self) -> None:
        result = calculate_impact(
            hardware="A100_80GB",
            region="us-east-1",
            duration_hours=1.0,
        )
        assert result.utilization == 1.0

    def test_multiple_accelerators_scale_linearly(self) -> None:
        """Energy should scale exactly linearly with num_accelerators."""
        single = calculate_impact(
            hardware="H100_SXM",
            region="europe-west1",
            duration_hours=10.0,
            num_accelerators=1,
        )
        eight = calculate_impact(
            hardware="H100_SXM",
            region="europe-west1",
            duration_hours=10.0,
            num_accelerators=8,
        )
        assert eight.energy_kwh == pytest.approx(single.energy_kwh * 8, rel=1e-9)
        assert eight.co2e_kg == pytest.approx(single.co2e_kg * 8, rel=1e-9)
        assert eight.water_liters == pytest.approx(single.water_liters * 8, rel=1e-9)

    def test_duration_scales_linearly(self) -> None:
        """Energy should scale exactly linearly with duration."""
        one_hour = calculate_impact(
            hardware="TPU_v4",
            region="us-central1",
            duration_hours=1.0,
        )
        ten_hours = calculate_impact(
            hardware="TPU_v4",
            region="us-central1",
            duration_hours=10.0,
        )
        assert ten_hours.energy_kwh == pytest.approx(one_hour.energy_kwh * 10, rel=1e-9)

    def test_low_carbon_region_less_co2_than_high_carbon(self) -> None:
        """Oregon (low-carbon) must produce less CO2e than Bahrain (high-carbon)."""
        oregon = calculate_impact(
            hardware="A100_80GB",
            region="us-west-2",
            duration_hours=24.0,
        )
        bahrain = calculate_impact(
            hardware="A100_80GB",
            region="me-south-1",
            duration_hours=24.0,
        )
        assert oregon.co2e_kg < bahrain.co2e_kg

    def test_higher_utilization_means_more_energy(self) -> None:
        low = calculate_impact(
            hardware="A100_80GB",
            region="us-east-1",
            duration_hours=10.0,
            utilization=0.5,
        )
        high = calculate_impact(
            hardware="A100_80GB",
            region="us-east-1",
            duration_hours=10.0,
            utilization=1.0,
        )
        assert high.energy_kwh > low.energy_kwh
        assert high.co2e_kg > low.co2e_kg
        assert high.water_liters > low.water_liters

    def test_energy_always_positive(self) -> None:
        result = calculate_impact(
            hardware="L4",
            region="eu-north-1",
            duration_hours=0.01,
            num_accelerators=1,
            utilization=0.1,
        )
        assert result.energy_kwh > 0

    def test_co2e_always_non_negative(self) -> None:
        result = calculate_impact(
            hardware="T4",
            region="us-west-2",
            duration_hours=1.0,
        )
        assert result.co2e_kg >= 0

    def test_water_always_non_negative(self) -> None:
        result = calculate_impact(
            hardware="T4",
            region="us-west-2",
            duration_hours=1.0,
        )
        assert result.water_liters >= 0

    def test_rounded_to_six_decimal_places(self) -> None:
        """All float metrics must be rounded to at most 6 decimal places."""
        result = calculate_impact(
            hardware="A100_80GB",
            region="us-east-1",
            duration_hours=1.0 / 3,  # produces many decimal places
        )
        for field_name in ("energy_kwh", "co2e_kg", "water_liters"):
            value = getattr(result, field_name)
            # Check that value matches its 6-dp rounded form
            assert value == round(value, 6), (
                f"{field_name}={value} is not rounded to 6 decimal places"
            )

    def test_inference_workload_type_stored(self) -> None:
        result = calculate_impact(
            hardware="T4",
            region="us-east-1",
            duration_hours=1.0,
            workload_type="inference",
        )
        assert result.workload_type == "inference"

    def test_tpu_hardware(self) -> None:
        """TPU v4 calculation should work correctly."""
        result = calculate_impact(
            hardware="TPU_v4",
            region="us-central1",
            duration_hours=24.0,
            num_accelerators=64,
            workload_type="training",
        )
        # 64 * 192 / 1000 * 24 * PUE(us-central1=1.11)
        expected_energy = (64 * 192 / 1000) * 24 * 1.11
        assert result.energy_kwh == pytest.approx(expected_energy, rel=1e-4)

    def test_amd_mi300x(self) -> None:
        """MI300X should use 750 W TDP."""
        result = calculate_impact(
            hardware="MI300X",
            region="us-east-1",
            duration_hours=1.0,
            num_accelerators=1,
        )
        assert result.tdp_watts == pytest.approx(750.0)

    def test_h100_pcie(self) -> None:
        """H100 PCIe should use 350 W TDP."""
        result = calculate_impact(
            hardware="H100_PCIe",
            region="europe-west4",
            duration_hours=10.0,
        )
        assert result.tdp_watts == pytest.approx(350.0)
        assert result.energy_kwh > 0

    def test_gaudi2_hardware(self) -> None:
        """Intel Gaudi 2 (600 W) should be correctly looked up."""
        result = calculate_impact(
            hardware="Gaudi2",
            region="eastus",
            duration_hours=5.0,
            num_accelerators=2,
        )
        assert result.tdp_watts == pytest.approx(600.0)
        assert result.num_accelerators == 2

    def test_nordic_region_low_carbon_intensity(self) -> None:
        """eu-north-1 (Stockholm) has near-zero carbon — very low CO2e."""
        result = calculate_impact(
            hardware="A100_80GB",
            region="eu-north-1",
            duration_hours=24.0,
        )
        assert result.carbon_intensity_g_per_kwh < 50.0
        assert result.co2e_kg < result.energy_kwh  # very low CI ratio

    def test_result_pue_is_float(self) -> None:
        result = calculate_impact(
            hardware="A100_80GB",
            region="us-east-1",
            duration_hours=1.0,
        )
        assert isinstance(result.pue, float)
        assert result.pue >= 1.0

    def test_result_wue_is_float(self) -> None:
        result = calculate_impact(
            hardware="A100_80GB",
            region="us-east-1",
            duration_hours=1.0,
        )
        assert isinstance(result.wue, float)
        assert result.wue >= 0.0

    def test_result_tdp_matches_data(self) -> None:
        """tdp_watts in result must match the data table exactly."""
        hw = "H100_SXM"
        result = calculate_impact(
            hardware=hw,
            region="us-east-1",
            duration_hours=1.0,
        )
        assert result.tdp_watts == pytest.approx(data.HARDWARE_TDP_WATTS[hw])

    def test_utilization_half_halves_energy_vs_full(self) -> None:
        """50% utilization gives exactly half the energy of 100%."""
        full = calculate_impact(
            hardware="A100_80GB",
            region="us-east-1",
            duration_hours=1.0,
            utilization=1.0,
        )
        half = calculate_impact(
            hardware="A100_80GB",
            region="us-east-1",
            duration_hours=1.0,
            utilization=0.5,
        )
        assert half.energy_kwh == pytest.approx(full.energy_kwh / 2, rel=1e-9)

    def test_co2e_and_water_derived_from_energy(self) -> None:
        """CO2e and water should be derivable from the energy value and factors."""
        result = calculate_impact(
            hardware="V100_32GB",
            region="eu-west-3",
            duration_hours=6.0,
            num_accelerators=2,
        )
        expected_co2e = result.energy_kwh * result.carbon_intensity_g_per_kwh / 1000.0
        expected_water = result.energy_kwh * result.wue
        assert result.co2e_kg == pytest.approx(expected_co2e, rel=1e-5)
        assert result.water_liters == pytest.approx(expected_water, rel=1e-5)


# ---------------------------------------------------------------------------
# calculate_impact — input validation
# ---------------------------------------------------------------------------


class TestCalculateImpactValidation:
    """Validation tests exercised through calculate_impact."""

    def test_unknown_hardware_raises(self) -> None:
        with pytest.raises(CalculatorError, match="Unknown hardware"):
            calculate_impact(
                hardware="NONEXISTENT_GPU",
                region="us-east-1",
                duration_hours=1.0,
            )

    def test_unknown_region_raises(self) -> None:
        with pytest.raises(CalculatorError, match="Unknown region"):
            calculate_impact(
                hardware="A100_80GB",
                region="mars-east-1",
                duration_hours=1.0,
            )

    def test_empty_hardware_raises(self) -> None:
        with pytest.raises(CalculatorError, match="hardware"):
            calculate_impact(
                hardware="",
                region="us-east-1",
                duration_hours=1.0,
            )

    def test_whitespace_only_hardware_raises(self) -> None:
        with pytest.raises(CalculatorError, match="hardware"):
            calculate_impact(
                hardware="   ",
                region="us-east-1",
                duration_hours=1.0,
            )

    def test_empty_region_raises(self) -> None:
        with pytest.raises(CalculatorError, match="region"):
            calculate_impact(
                hardware="A100_80GB",
                region="",
                duration_hours=1.0,
            )

    def test_whitespace_only_region_raises(self) -> None:
        with pytest.raises(CalculatorError, match="region"):
            calculate_impact(
                hardware="A100_80GB",
                region="  ",
                duration_hours=1.0,
            )

    def test_zero_duration_raises(self) -> None:
        with pytest.raises(CalculatorError, match="duration_hours"):
            calculate_impact(
                hardware="A100_80GB",
                region="us-east-1",
                duration_hours=0.0,
            )

    def test_negative_duration_raises(self) -> None:
        with pytest.raises(CalculatorError, match="duration_hours"):
            calculate_impact(
                hardware="A100_80GB",
                region="us-east-1",
                duration_hours=-1.0,
            )

    def test_zero_accelerators_raises(self) -> None:
        with pytest.raises(CalculatorError, match="num_accelerators"):
            calculate_impact(
                hardware="A100_80GB",
                region="us-east-1",
                duration_hours=1.0,
                num_accelerators=0,
            )

    def test_negative_accelerators_raises(self) -> None:
        with pytest.raises(CalculatorError, match="num_accelerators"):
            calculate_impact(
                hardware="A100_80GB",
                region="us-east-1",
                duration_hours=1.0,
                num_accelerators=-4,
            )

    def test_float_accelerators_raises(self) -> None:
        with pytest.raises(CalculatorError, match="num_accelerators"):
            calculate_impact(
                hardware="A100_80GB",
                region="us-east-1",
                duration_hours=1.0,
                num_accelerators=2.0,  # type: ignore[arg-type]
            )

    def test_invalid_workload_type_raises(self) -> None:
        with pytest.raises(CalculatorError, match="workload_type"):
            calculate_impact(
                hardware="A100_80GB",
                region="us-east-1",
                duration_hours=1.0,
                workload_type="fine-tuning",
            )

    def test_workload_type_empty_string_raises(self) -> None:
        with pytest.raises(CalculatorError, match="workload_type"):
            calculate_impact(
                hardware="A100_80GB",
                region="us-east-1",
                duration_hours=1.0,
                workload_type="",
            )

    def test_zero_utilization_raises(self) -> None:
        with pytest.raises(CalculatorError, match="utilization"):
            calculate_impact(
                hardware="A100_80GB",
                region="us-east-1",
                duration_hours=1.0,
                utilization=0.0,
            )

    def test_utilization_above_one_raises(self) -> None:
        with pytest.raises(CalculatorError, match="utilization"):
            calculate_impact(
                hardware="A100_80GB",
                region="us-east-1",
                duration_hours=1.0,
                utilization=1.1,
            )

    def test_negative_utilization_raises(self) -> None:
        with pytest.raises(CalculatorError, match="utilization"):
            calculate_impact(
                hardware="A100_80GB",
                region="us-east-1",
                duration_hours=1.0,
                utilization=-0.1,
            )

    def test_bool_duration_raises(self) -> None:
        """Boolean True (== 1 in Python) must be rejected for duration_hours."""
        with pytest.raises(CalculatorError, match="duration_hours"):
            calculate_impact(
                hardware="A100_80GB",
                region="us-east-1",
                duration_hours=True,  # type: ignore[arg-type]
            )

    def test_bool_utilization_raises(self) -> None:
        """Boolean True must be rejected for utilization."""
        with pytest.raises(CalculatorError, match="utilization"):
            calculate_impact(
                hardware="A100_80GB",
                region="us-east-1",
                duration_hours=1.0,
                utilization=True,  # type: ignore[arg-type]
            )

    def test_bool_num_accelerators_raises(self) -> None:
        """Boolean True must be rejected for num_accelerators."""
        with pytest.raises(CalculatorError, match="num_accelerators"):
            calculate_impact(
                hardware="A100_80GB",
                region="us-east-1",
                duration_hours=1.0,
                num_accelerators=True,  # type: ignore[arg-type]
            )

    def test_unknown_hardware_error_lists_options(self) -> None:
        """Error message for unknown hardware must mention available options."""
        with pytest.raises(CalculatorError, match="Available options"):
            calculate_impact(
                hardware="FAKE_GPU",
                region="us-east-1",
                duration_hours=1.0,
            )

    def test_unknown_region_error_lists_options(self) -> None:
        """Error message for unknown region must mention available options."""
        with pytest.raises(CalculatorError, match="Available options"):
            calculate_impact(
                hardware="A100_80GB",
                region="fake-region-99",
                duration_hours=1.0,
            )

    def test_calculator_error_is_value_error_subclass(self) -> None:
        """CalculatorError must be a subclass of ValueError."""
        assert issubclass(CalculatorError, ValueError)

    def test_calculator_error_can_be_caught_as_value_error(self) -> None:
        """CalculatorError must be catchable as ValueError."""
        caught = False
        try:
            calculate_impact(
                hardware="FAKE",
                region="us-east-1",
                duration_hours=1.0,
            )
        except ValueError:
            caught = True
        assert caught

    def test_none_hardware_raises(self) -> None:
        with pytest.raises(CalculatorError):
            calculate_impact(
                hardware=None,  # type: ignore[arg-type]
                region="us-east-1",
                duration_hours=1.0,
            )

    def test_none_region_raises(self) -> None:
        with pytest.raises(CalculatorError):
            calculate_impact(
                hardware="A100_80GB",
                region=None,  # type: ignore[arg-type]
                duration_hours=1.0,
            )


# ---------------------------------------------------------------------------
# ImpactResult.to_dict
# ---------------------------------------------------------------------------


class TestImpactResultToDict:
    """Tests for the ImpactResult serialisation helper."""

    def _make_result(self) -> ImpactResult:
        return calculate_impact(
            hardware="A100_80GB",
            region="us-east-1",
            duration_hours=1.0,
            num_accelerators=1,
            workload_type="training",
            utilization=1.0,
        )

    def test_returns_dict(self) -> None:
        result = self._make_result()
        assert isinstance(result.to_dict(), dict)

    def test_expected_keys_present(self) -> None:
        d = self._make_result().to_dict()
        expected_keys = {
            "energy_kwh",
            "co2e_kg",
            "water_liters",
            "hardware",
            "region",
            "duration_hours",
            "num_accelerators",
            "utilization",
            "workload_type",
            "pue",
            "wue",
            "carbon_intensity_g_per_kwh",
            "tdp_watts",
        }
        assert set(d.keys()) == expected_keys

    def test_values_match_result_fields(self) -> None:
        result = self._make_result()
        d = result.to_dict()
        assert d["energy_kwh"] == result.energy_kwh
        assert d["co2e_kg"] == result.co2e_kg
        assert d["water_liters"] == result.water_liters
        assert d["hardware"] == result.hardware
        assert d["region"] == result.region
        assert d["duration_hours"] == result.duration_hours
        assert d["num_accelerators"] == result.num_accelerators
        assert d["utilization"] == result.utilization
        assert d["workload_type"] == result.workload_type
        assert d["pue"] == result.pue
        assert d["wue"] == result.wue
        assert d["carbon_intensity_g_per_kwh"] == result.carbon_intensity_g_per_kwh
        assert d["tdp_watts"] == result.tdp_watts

    def test_all_values_json_serialisable(self) -> None:
        """to_dict() output must be JSON-serialisable without custom encoders."""
        d = self._make_result().to_dict()
        # This should not raise
        serialised = json.dumps(d)
        roundtripped = json.loads(serialised)
        assert roundtripped["hardware"] == "A100_80GB"

    def test_to_dict_called_twice_returns_equal_dicts(self) -> None:
        """to_dict() is deterministic — calling it twice gives identical results."""
        result = self._make_result()
        d1 = result.to_dict()
        d2 = result.to_dict()
        assert d1 == d2

    def test_to_dict_hardware_is_string(self) -> None:
        d = self._make_result().to_dict()
        assert isinstance(d["hardware"], str)

    def test_to_dict_region_is_string(self) -> None:
        d = self._make_result().to_dict()
        assert isinstance(d["region"], str)

    def test_to_dict_num_accelerators_is_int(self) -> None:
        d = self._make_result().to_dict()
        assert isinstance(d["num_accelerators"], int)

    def test_to_dict_energy_is_float(self) -> None:
        d = self._make_result().to_dict()
        assert isinstance(d["energy_kwh"], float)

    def test_to_dict_co2e_is_float(self) -> None:
        d = self._make_result().to_dict()
        assert isinstance(d["co2e_kg"], float)

    def test_to_dict_water_is_float(self) -> None:
        d = self._make_result().to_dict()
        assert isinstance(d["water_liters"], float)


# ---------------------------------------------------------------------------
# get_available_hardware / get_available_regions
# ---------------------------------------------------------------------------


class TestHelperListFunctions:
    """Tests for the list-of-options helper functions."""

    def test_get_available_hardware_returns_list(self) -> None:
        hw = get_available_hardware()
        assert isinstance(hw, list)

    def test_get_available_hardware_non_empty(self) -> None:
        assert len(get_available_hardware()) > 0

    def test_get_available_hardware_sorted(self) -> None:
        hw = get_available_hardware()
        assert hw == sorted(hw)

    def test_get_available_hardware_contains_a100(self) -> None:
        assert "A100_80GB" in get_available_hardware()

    def test_get_available_hardware_contains_h100(self) -> None:
        assert "H100_SXM" in get_available_hardware()

    def test_get_available_hardware_contains_t4(self) -> None:
        assert "T4" in get_available_hardware()

    def test_get_available_hardware_contains_tpu_v4(self) -> None:
        assert "TPU_v4" in get_available_hardware()

    def test_get_available_hardware_matches_data_keys(self) -> None:
        assert set(get_available_hardware()) == set(data.HARDWARE_TDP_WATTS.keys())

    def test_get_available_hardware_all_strings(self) -> None:
        for hw in get_available_hardware():
            assert isinstance(hw, str)
            assert len(hw) > 0

    def test_get_available_regions_returns_list(self) -> None:
        regions = get_available_regions()
        assert isinstance(regions, list)

    def test_get_available_regions_non_empty(self) -> None:
        assert len(get_available_regions()) > 0

    def test_get_available_regions_sorted(self) -> None:
        regions = get_available_regions()
        assert regions == sorted(regions)

    def test_get_available_regions_contains_us_east_1(self) -> None:
        assert "us-east-1" in get_available_regions()

    def test_get_available_regions_contains_europe_west4(self) -> None:
        assert "europe-west4" in get_available_regions()

    def test_get_available_regions_matches_data_keys(self) -> None:
        assert set(get_available_regions()) == set(
            data.CARBON_INTENSITY_G_PER_KWH.keys()
        )

    def test_get_available_regions_all_strings(self) -> None:
        for region in get_available_regions():
            assert isinstance(region, str)
            assert len(region) > 0

    def test_get_available_hardware_minimum_count(self) -> None:
        """We expect at least 20 hardware presets."""
        assert len(get_available_hardware()) >= 20

    def test_get_available_regions_minimum_count(self) -> None:
        """We expect at least 40 regions."""
        assert len(get_available_regions()) >= 40


# ---------------------------------------------------------------------------
# PUE/WUE fallback behaviour
# ---------------------------------------------------------------------------


class TestFallbackBehaviour:
    """Verify that regions without explicit PUE/WUE use the global defaults."""

    def test_fallback_pue_applied_for_known_region(self) -> None:
        """A region in both tables returns its explicit PUE."""
        pue = _lookup_pue("eu-west-3")
        expected = data.PUE_BY_REGION["eu-west-3"]
        assert pue == pytest.approx(expected)

    def test_default_pue_returned_for_unknown_key(self) -> None:
        """A key not in PUE_BY_REGION should return DEFAULT_PUE."""
        pue = _lookup_pue("__synthetic_test_region__")
        assert pue == pytest.approx(data.DEFAULT_PUE)

    def test_default_wue_returned_for_unknown_key(self) -> None:
        """A key not in WUE_BY_REGION should return DEFAULT_WUE."""
        wue = _lookup_wue("__synthetic_test_region__")
        assert wue == pytest.approx(data.DEFAULT_WUE)

    def test_default_pue_gte_one(self) -> None:
        """DEFAULT_PUE must be physically meaningful (>= 1.0)."""
        assert data.DEFAULT_PUE >= 1.0

    def test_default_wue_non_negative(self) -> None:
        """DEFAULT_WUE must be non-negative."""
        assert data.DEFAULT_WUE >= 0.0

    def test_lookup_pue_returns_float(self) -> None:
        pue = _lookup_pue("us-east-1")
        assert isinstance(pue, float)

    def test_lookup_wue_returns_float(self) -> None:
        wue = _lookup_wue("us-east-1")
        assert isinstance(wue, float)

    def test_all_known_regions_have_valid_pue(self) -> None:
        """Every region in the carbon intensity table should return a PUE >= 1.0."""
        for region in data.CARBON_INTENSITY_G_PER_KWH:
            pue = _lookup_pue(region)
            assert pue >= 1.0, f"PUE for {region!r} is {pue} (< 1.0)"

    def test_all_known_regions_have_valid_wue(self) -> None:
        """Every region in the carbon intensity table should return a WUE >= 0."""
        for region in data.CARBON_INTENSITY_G_PER_KWH:
            wue = _lookup_wue(region)
            assert wue >= 0.0, f"WUE for {region!r} is {wue} (< 0)"


# ---------------------------------------------------------------------------
# Private lookup helpers — direct tests
# ---------------------------------------------------------------------------


class TestPrivateLookupHelpers:
    """Direct tests for the private lookup functions."""

    def test_lookup_hardware_tdp_known_key(self) -> None:
        tdp = _lookup_hardware_tdp("A100_80GB")
        assert tdp == pytest.approx(400.0)

    def test_lookup_hardware_tdp_returns_float(self) -> None:
        tdp = _lookup_hardware_tdp("T4")
        assert isinstance(tdp, float)

    def test_lookup_hardware_tdp_unknown_raises(self) -> None:
        with pytest.raises(CalculatorError, match="Unknown hardware"):
            _lookup_hardware_tdp("NOT_A_REAL_GPU")

    def test_lookup_hardware_tdp_error_lists_available(self) -> None:
        with pytest.raises(CalculatorError, match="Available options"):
            _lookup_hardware_tdp("NOT_A_REAL_GPU")

    def test_lookup_carbon_intensity_known_key(self) -> None:
        ci = _lookup_carbon_intensity("us-east-1")
        assert ci == pytest.approx(415.0)

    def test_lookup_carbon_intensity_returns_float(self) -> None:
        ci = _lookup_carbon_intensity("eu-west-3")
        assert isinstance(ci, float)

    def test_lookup_carbon_intensity_unknown_raises(self) -> None:
        with pytest.raises(CalculatorError, match="Unknown region"):
            _lookup_carbon_intensity("not-a-real-region")

    def test_lookup_carbon_intensity_error_lists_available(self) -> None:
        with pytest.raises(CalculatorError, match="Available options"):
            _lookup_carbon_intensity("not-a-real-region")

    def test_lookup_pue_gcp_finland_very_low(self) -> None:
        """Finland GCP campus (europe-north1) has very low PUE."""
        pue = _lookup_pue("europe-north1")
        assert pue < 1.10

    def test_lookup_wue_oregon_low(self) -> None:
        """Oregon (us-west-2) has low WUE due to mild climate."""
        wue = _lookup_wue("us-west-2")
        assert wue < 0.5

    def test_lookup_wue_uae_high(self) -> None:
        """UAE (uaenorth) has high WUE due to extreme heat."""
        wue = _lookup_wue("uaenorth")
        assert wue >= 2.0


# ---------------------------------------------------------------------------
# Reference calculation: published ML training footprint estimates
# ---------------------------------------------------------------------------


class TestReferenceCalculations:
    """Sanity-check against order-of-magnitude estimates from literature.

    These are not meant to be exact reproductions of published figures but
    ensure our calculator is in the right ballpark.
    """

    def test_gpt3_scale_training_sanity(self) -> None:
        """GPT-3 training used ~1287 MWh (Patterson et al. 2021).

        We simulate a rough equivalent: 1024 A100s for ~34 days at 100 %
        utilisation. The result should be in the hundreds of MWh range.
        """
        result = calculate_impact(
            hardware="A100_80GB",
            region="us-east-1",
            duration_hours=34 * 24,  # 34 days
            num_accelerators=1024,
            workload_type="training",
            utilization=1.0,
        )
        # 1024 * 400 W * 816 h * 1.20 PUE / 1000 = 401,899 kWh ~= 402 MWh
        # We just check order of magnitude: > 100 MWh.
        assert result.energy_kwh > 100_000  # > 100 MWh
        assert result.energy_kwh < 10_000_000  # < 10 GWh (sanity cap)

    def test_small_inference_minimal_footprint(self) -> None:
        """Single T4 for 1 minute of inference should use < 1 Wh."""
        result = calculate_impact(
            hardware="T4",
            region="europe-west1",
            duration_hours=1 / 60,  # 1 minute
            num_accelerators=1,
            workload_type="inference",
            utilization=0.5,
        )
        # 70 * 0.5 / 1000 * (1/60) * 1.09 ~= 0.000636 kWh = 0.636 Wh
        assert result.energy_kwh < 0.01  # less than 10 Wh
        assert result.co2e_kg < 0.01

    def test_single_a100_single_hour_energy(self) -> None:
        """Single A100 (400 W) at 100% for 1 hour in a region with PUE 1.20."""
        result = calculate_impact(
            hardware="A100_80GB",
            region="us-east-1",  # PUE = 1.20
            duration_hours=1.0,
            num_accelerators=1,
            utilization=1.0,
        )
        # 400 W * 1 / 1000 * 1 h * 1.20 = 0.48 kWh
        assert result.energy_kwh == pytest.approx(0.48, rel=1e-4)

    def test_carbon_order_of_magnitude_high_carbon_region(self) -> None:
        """South Africa (928 gCO2/kWh) should produce >> Sweden (8 gCO2/kWh)."""
        za = calculate_impact(
            hardware="A100_80GB",
            region="af-south-1",
            duration_hours=1.0,
        )
        se = calculate_impact(
            hardware="A100_80GB",
            region="eu-north-1",
            duration_hours=1.0,
        )
        # South Africa intensity (928) / Sweden intensity (8) ~= 116x
        ratio = za.co2e_kg / se.co2e_kg
        assert ratio > 50  # at least 50x more CO2 in South Africa

    def test_water_hot_desert_vs_cold_nordic(self) -> None:
        """UAE should use far more water per kWh than Nordic regions."""
        uae = calculate_impact(
            hardware="T4",
            region="uaenorth",
            duration_hours=10.0,
        )
        nordic = calculate_impact(
            hardware="T4",
            region="swedencentral",
            duration_hours=10.0,
        )
        assert uae.water_liters > nordic.water_liters * 5
