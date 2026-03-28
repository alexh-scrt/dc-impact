"""Unit tests for dc_impact.data — static lookup tables.

Verifies consistency, completeness, and value sanity for all lookup tables
defined in the data module. These tests do not exercise business logic;
they guard against accidental corruption of reference values and ensure
that all tables remain internally consistent.
"""

from __future__ import annotations

import pytest

from dc_impact import data


# ---------------------------------------------------------------------------
# Structural / consistency tests
# ---------------------------------------------------------------------------


def test_hardware_tdp_all_positive() -> None:
    """Every TDP value in HARDWARE_TDP_WATTS must be strictly positive."""
    for key, tdp in data.HARDWARE_TDP_WATTS.items():
        assert tdp > 0, f"TDP for {key!r} is not positive: {tdp}"


def test_carbon_intensity_all_non_negative() -> None:
    """Every carbon intensity value must be >= 0."""
    for key, ci in data.CARBON_INTENSITY_G_PER_KWH.items():
        assert ci >= 0, f"Carbon intensity for {key!r} is negative: {ci}"


def test_pue_all_gte_one() -> None:
    """Every PUE value must be >= 1.0 (physically impossible to have PUE < 1)."""
    for key, pue in data.PUE_BY_REGION.items():
        assert pue >= 1.0, f"PUE for {key!r} is < 1.0: {pue}"


def test_wue_all_non_negative() -> None:
    """Every WUE value must be >= 0."""
    for key, wue in data.WUE_BY_REGION.items():
        assert wue >= 0, f"WUE for {key!r} is negative: {wue}"


def test_pue_regions_subset_of_carbon_intensity_regions() -> None:
    """Every region with a PUE value must also have a carbon intensity value.

    The carbon intensity table is the authoritative list of valid regions;
    PUE and WUE are supplementary — they may be a subset.
    """
    ci_keys = set(data.CARBON_INTENSITY_G_PER_KWH.keys())
    pue_keys = set(data.PUE_BY_REGION.keys())
    extra = pue_keys - ci_keys
    assert not extra, (
        f"PUE table contains regions not in carbon intensity table: {extra}"
    )


def test_wue_regions_subset_of_carbon_intensity_regions() -> None:
    """Every region with a WUE value must also have a carbon intensity value."""
    ci_keys = set(data.CARBON_INTENSITY_G_PER_KWH.keys())
    wue_keys = set(data.WUE_BY_REGION.keys())
    extra = wue_keys - ci_keys
    assert not extra, (
        f"WUE table contains regions not in carbon intensity table: {extra}"
    )


def test_pue_and_wue_regions_match() -> None:
    """PUE and WUE tables must cover exactly the same set of regions."""
    pue_keys = set(data.PUE_BY_REGION.keys())
    wue_keys = set(data.WUE_BY_REGION.keys())
    assert pue_keys == wue_keys, (
        f"PUE/WUE region mismatch.\n"
        f"Only in PUE: {pue_keys - wue_keys}\n"
        f"Only in WUE: {wue_keys - pue_keys}"
    )


def test_hardware_metadata_keys_match_tdp_keys() -> None:
    """HARDWARE_METADATA must contain exactly the same keys as HARDWARE_TDP_WATTS."""
    tdp_keys = set(data.HARDWARE_TDP_WATTS.keys())
    meta_keys = set(data.HARDWARE_METADATA.keys())
    assert tdp_keys == meta_keys, (
        f"Hardware metadata/TDP key mismatch.\n"
        f"Only in TDP: {tdp_keys - meta_keys}\n"
        f"Only in metadata: {meta_keys - tdp_keys}"
    )


def test_region_metadata_keys_subset_of_carbon_intensity() -> None:
    """REGION_METADATA keys must be a subset of CARBON_INTENSITY_G_PER_KWH."""
    ci_keys = set(data.CARBON_INTENSITY_G_PER_KWH.keys())
    meta_keys = set(data.REGION_METADATA.keys())
    extra = meta_keys - ci_keys
    assert not extra, (
        f"REGION_METADATA contains regions not in carbon intensity table: {extra}"
    )


# ---------------------------------------------------------------------------
# Default constant tests
# ---------------------------------------------------------------------------


def test_default_pue_value() -> None:
    """DEFAULT_PUE should equal 1.58 (Uptime Institute 2023 average)."""
    assert data.DEFAULT_PUE == pytest.approx(1.58)


def test_default_wue_value() -> None:
    """DEFAULT_WUE should equal 1.8."""
    assert data.DEFAULT_WUE == pytest.approx(1.8)


def test_default_pue_gte_one() -> None:
    """DEFAULT_PUE must be >= 1.0."""
    assert data.DEFAULT_PUE >= 1.0


def test_default_wue_non_negative() -> None:
    """DEFAULT_WUE must be non-negative."""
    assert data.DEFAULT_WUE >= 0.0


# ---------------------------------------------------------------------------
# Spot-check known reference values
# ---------------------------------------------------------------------------


class TestHardwareTDP:
    """Spot-check TDP values against published vendor datasheets."""

    def test_a100_80gb_tdp(self) -> None:
        assert data.HARDWARE_TDP_WATTS["A100_80GB"] == pytest.approx(400.0)

    def test_a100_40gb_tdp(self) -> None:
        assert data.HARDWARE_TDP_WATTS["A100_40GB"] == pytest.approx(400.0)

    def test_h100_sxm_tdp(self) -> None:
        assert data.HARDWARE_TDP_WATTS["H100_SXM"] == pytest.approx(700.0)

    def test_h100_pcie_tdp(self) -> None:
        assert data.HARDWARE_TDP_WATTS["H100_PCIe"] == pytest.approx(350.0)

    def test_v100_32gb_tdp(self) -> None:
        assert data.HARDWARE_TDP_WATTS["V100_32GB"] == pytest.approx(300.0)

    def test_t4_tdp(self) -> None:
        assert data.HARDWARE_TDP_WATTS["T4"] == pytest.approx(70.0)

    def test_l4_tdp(self) -> None:
        assert data.HARDWARE_TDP_WATTS["L4"] == pytest.approx(72.0)

    def test_tpu_v4_tdp(self) -> None:
        assert data.HARDWARE_TDP_WATTS["TPU_v4"] == pytest.approx(192.0)

    def test_tpu_v5e_tdp(self) -> None:
        assert data.HARDWARE_TDP_WATTS["TPU_v5e"] == pytest.approx(197.0)

    def test_mi300x_tdp(self) -> None:
        assert data.HARDWARE_TDP_WATTS["MI300X"] == pytest.approx(750.0)

    def test_mi250x_tdp(self) -> None:
        assert data.HARDWARE_TDP_WATTS["MI250X"] == pytest.approx(560.0)

    def test_gaudi2_tdp(self) -> None:
        assert data.HARDWARE_TDP_WATTS["Gaudi2"] == pytest.approx(600.0)


class TestCarbonIntensity:
    """Spot-check carbon intensity values against known grid data."""

    def test_us_east_1(self) -> None:
        # PJM grid ~415 gCO2/kWh
        assert data.CARBON_INTENSITY_G_PER_KWH["us-east-1"] == pytest.approx(415.0)

    def test_us_west_2_low_carbon(self) -> None:
        # Oregon is hydro-heavy — should be well below 200 gCO2/kWh
        ci = data.CARBON_INTENSITY_G_PER_KWH["us-west-2"]
        assert ci < 200.0, f"Expected Oregon to be low-carbon, got {ci}"

    def test_eu_north_1_near_zero(self) -> None:
        # Swedish/Nordic grid — near zero carbon
        ci = data.CARBON_INTENSITY_G_PER_KWH["eu-north-1"]
        assert ci < 50.0, f"Expected Stockholm to be near-zero, got {ci}"

    def test_af_south_1_coal_heavy(self) -> None:
        # South Africa is heavily coal-dependent
        ci = data.CARBON_INTENSITY_G_PER_KWH["af-south-1"]
        assert ci > 800.0, f"Expected Cape Town to be coal-heavy, got {ci}"

    def test_eu_west_3_paris_low_carbon(self) -> None:
        # France is nuclear-heavy
        ci = data.CARBON_INTENSITY_G_PER_KWH["eu-west-3"]
        assert ci < 100.0, f"Expected Paris to be low-carbon, got {ci}"

    def test_europe_north1_gcp(self) -> None:
        # GCP Finland — Nordic grid
        ci = data.CARBON_INTENSITY_G_PER_KWH["europe-north1"]
        assert ci < 50.0

    def test_swedencentral_azure(self) -> None:
        ci = data.CARBON_INTENSITY_G_PER_KWH["swedencentral"]
        assert ci < 50.0

    def test_gcp_europe_west1_belgium(self) -> None:
        # Belgium has nuclear + renewables
        ci = data.CARBON_INTENSITY_G_PER_KWH["europe-west1"]
        assert ci < 200.0

    def test_asia_east1_taiwan(self) -> None:
        ci = data.CARBON_INTENSITY_G_PER_KWH["asia-east1"]
        assert ci == pytest.approx(545.0)


class TestPUE:
    """Spot-check PUE values against published cloud provider reports."""

    def test_gcp_europe_north1_lowest(self) -> None:
        # Finland — sea water cooling gives very low PUE
        pue = data.PUE_BY_REGION["europe-north1"]
        assert pue < 1.10

    def test_aws_regions_around_1_20(self) -> None:
        # AWS publicly states ~1.20 fleet PUE
        for region in ("us-east-1", "eu-west-1", "ap-northeast-1"):
            pue = data.PUE_BY_REGION[region]
            assert 1.15 <= pue <= 1.30, f"AWS {region} PUE {pue} out of expected range"

    def test_hot_regions_higher_pue(self) -> None:
        # UAE and Bahrain require more cooling
        for region in ("uaenorth", "uaecentral"):
            pue = data.PUE_BY_REGION[region]
            assert pue >= 1.25, f"{region} PUE {pue} expected >= 1.25"

    def test_azure_fleet_pue_range(self) -> None:
        # Azure reports ~1.18 fleet PUE
        for region in ("eastus", "westus2", "westeurope", "northeurope"):
            pue = data.PUE_BY_REGION[region]
            assert 1.10 <= pue <= 1.30, f"Azure {region} PUE {pue} out of range"


class TestWUE:
    """Spot-check WUE values."""

    def test_us_west_2_low_wue(self) -> None:
        # Oregon — cool climate, low evaporative demand
        wue = data.WUE_BY_REGION["us-west-2"]
        assert wue < 0.5

    def test_hot_regions_high_wue(self) -> None:
        # UAE — extreme heat requires heavy evaporative cooling
        for region in ("uaenorth", "uaecentral"):
            wue = data.WUE_BY_REGION[region]
            assert wue >= 2.0, f"{region} WUE {wue} expected >= 2.0"

    def test_finland_very_low_wue(self) -> None:
        # GCP Finland uses sea water cooling — near-zero WUE
        wue = data.WUE_BY_REGION["europe-north1"]
        assert wue < 0.2

    def test_tropical_regions_high_wue(self) -> None:
        for region in ("ap-southeast-1", "asia-southeast1", "southeastasia"):
            wue = data.WUE_BY_REGION[region]
            assert wue >= 1.0, f"{region} WUE {wue} expected >= 1.0 (tropical)"


# ---------------------------------------------------------------------------
# Metadata structure tests
# ---------------------------------------------------------------------------


def test_hardware_metadata_has_label_and_vendor() -> None:
    """Every HARDWARE_METADATA entry must have 'label' and 'vendor' keys."""
    for key, meta in data.HARDWARE_METADATA.items():
        assert "label" in meta, f"Missing 'label' for {key!r}"
        assert "vendor" in meta, f"Missing 'vendor' for {key!r}"
        assert isinstance(meta["label"], str) and meta["label"], (
            f"Empty label for {key!r}"
        )
        assert meta["vendor"] in {"nvidia", "amd", "google", "intel"}, (
            f"Unknown vendor {meta['vendor']!r} for {key!r}"
        )


def test_region_metadata_has_label_and_provider() -> None:
    """Every REGION_METADATA entry must have 'label' and 'provider' keys."""
    for key, meta in data.REGION_METADATA.items():
        assert "label" in meta, f"Missing 'label' for {key!r}"
        assert "provider" in meta, f"Missing 'provider' for {key!r}"
        assert isinstance(meta["label"], str) and meta["label"], (
            f"Empty label for {key!r}"
        )
        assert meta["provider"] in {"aws", "gcp", "azure"}, (
            f"Unknown provider {meta['provider']!r} for {key!r}"
        )


def test_table_sizes_reasonable() -> None:
    """Lookup tables should have a minimum number of entries."""
    assert len(data.HARDWARE_TDP_WATTS) >= 20
    assert len(data.CARBON_INTENSITY_G_PER_KWH) >= 40
    assert len(data.PUE_BY_REGION) >= 40
    assert len(data.WUE_BY_REGION) >= 40
    assert len(data.REGION_METADATA) >= 40
    assert len(data.HARDWARE_METADATA) >= 20
