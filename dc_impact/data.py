"""Static lookup tables for dc_impact.

This module holds all bundled reference data used by the calculation engine:

* :data:`HARDWARE_TDP_WATTS` — Thermal design power for GPU/TPU presets.
* :data:`CARBON_INTENSITY_G_PER_KWH` — Regional grid carbon intensity.
* :data:`PUE_BY_REGION` — Power Usage Effectiveness by cloud region.
* :data:`WUE_BY_REGION` — Water Usage Effectiveness by cloud region.
* :data:`DEFAULT_PUE` / :data:`DEFAULT_WUE` — Global fallback values.

All values are sourced from publicly available data (see individual table
docstrings for references). The tables will be fully populated in phase 2;
this module provides the structure and a small seed set so that the rest of
the package can import and reference them immediately.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Fallback / default constants
# ---------------------------------------------------------------------------

#: Global average PUE used when a region-specific value is unavailable.
#: Source: Uptime Institute Global Data Center Survey 2023 (average ~1.58).
DEFAULT_PUE: Final[float] = 1.58

#: Global average WUE (litres per kWh) used as fallback.
#: Source: The Green Grid / industry estimates.
DEFAULT_WUE: Final[float] = 1.8

# ---------------------------------------------------------------------------
# Hardware TDP lookup table
# ---------------------------------------------------------------------------

#: Mapping of hardware preset key to Thermal Design Power in watts.
#:
#: These figures represent the *maximum* rated TDP for each accelerator.
#: Actual power draw is scaled by the utilisation factor in the calculator.
#: Phase 2 will expand this table significantly.
HARDWARE_TDP_WATTS: Final[dict[str, float]] = {
    # NVIDIA GPUs
    "A100_40GB": 400.0,
    "A100_80GB": 400.0,
    "H100_SXM": 700.0,
    "H100_PCIe": 350.0,
    "V100_16GB": 300.0,
    "V100_32GB": 300.0,
    "A10G": 150.0,
    "A10": 150.0,
    "T4": 70.0,
    "L4": 72.0,
    "L40": 300.0,
    "L40S": 350.0,
    "RTX_4090": 450.0,
    "RTX_3090": 350.0,
    # Google TPUs
    "TPU_v3": 450.0,
    "TPU_v4": 192.0,
    "TPU_v5e": 197.0,
    # AMD GPUs
    "MI250X": 560.0,
    "MI300X": 750.0,
}

# ---------------------------------------------------------------------------
# Regional carbon intensity lookup table
# ---------------------------------------------------------------------------

#: Mapping of cloud region key to grid carbon intensity in gCO2e per kWh.
#:
#: Values are annual averages or best available estimates. Phase 2 will
#: expand this table and add citations.
CARBON_INTENSITY_G_PER_KWH: Final[dict[str, float]] = {
    # AWS regions
    "us-east-1": 415.0,   # N. Virginia, US average
    "us-east-2": 476.0,   # Ohio
    "us-west-1": 210.0,   # N. California
    "us-west-2": 136.0,   # Oregon (high renewable)
    "eu-west-1": 316.0,   # Ireland
    "eu-central-1": 338.0, # Frankfurt
    "ap-southeast-1": 493.0, # Singapore
    "ap-northeast-1": 506.0, # Tokyo
    # GCP regions
    "us-central1": 479.0,  # Iowa
    "us-east4": 415.0,    # N. Virginia
    "europe-west4": 284.0, # Netherlands
    "europe-west1": 150.0, # Belgium (low carbon)
    "asia-east1": 545.0,  # Taiwan
    # Azure regions
    "eastus": 415.0,
    "westus2": 136.0,
    "westeurope": 284.0,
    "northeurope": 316.0,
}

# ---------------------------------------------------------------------------
# PUE lookup table
# ---------------------------------------------------------------------------

#: Mapping of cloud region key to Power Usage Effectiveness.
#:
#: PUE expresses how much extra power a data centre uses beyond what the
#: IT equipment itself consumes. A PUE of 1.0 is ideal (no overhead); the
#: industry average is approximately 1.58. Hyperscale cloud DCs typically
#: achieve 1.1–1.2.
PUE_BY_REGION: Final[dict[str, float]] = {
    # AWS
    "us-east-1": 1.20,
    "us-east-2": 1.20,
    "us-west-1": 1.20,
    "us-west-2": 1.20,
    "eu-west-1": 1.20,
    "eu-central-1": 1.20,
    "ap-southeast-1": 1.20,
    "ap-northeast-1": 1.20,
    # GCP — Google publishes trailing-12-month fleet PUE
    "us-central1": 1.10,
    "us-east4": 1.10,
    "europe-west4": 1.09,
    "europe-west1": 1.09,
    "asia-east1": 1.13,
    # Azure
    "eastus": 1.18,
    "westus2": 1.18,
    "westeurope": 1.18,
    "northeurope": 1.18,
}

# ---------------------------------------------------------------------------
# WUE lookup table
# ---------------------------------------------------------------------------

#: Mapping of cloud region key to Water Usage Effectiveness in litres per kWh.
#:
#: WUE captures onsite water consumption (evaporative cooling) per unit of IT
#: energy consumed. Values vary widely by climate and cooling technology.
WUE_BY_REGION: Final[dict[str, float]] = {
    # AWS
    "us-east-1": 0.49,
    "us-east-2": 0.93,
    "us-west-1": 1.32,
    "us-west-2": 0.18,
    "eu-west-1": 0.25,
    "eu-central-1": 0.80,
    "ap-southeast-1": 1.50,
    "ap-northeast-1": 0.90,
    # GCP
    "us-central1": 1.10,
    "us-east4": 0.80,
    "europe-west4": 0.60,
    "europe-west1": 0.40,
    "asia-east1": 1.20,
    # Azure
    "eastus": 0.49,
    "westus2": 0.18,
    "westeurope": 0.60,
    "northeurope": 0.25,
}
