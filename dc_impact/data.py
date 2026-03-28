"""Static lookup tables for dc_impact.

This module holds all bundled reference data used by the calculation engine:

* :data:`HARDWARE_TDP_WATTS` — Thermal design power for GPU/TPU presets.
* :data:`CARBON_INTENSITY_G_PER_KWH` — Regional grid carbon intensity.
* :data:`PUE_BY_REGION` — Power Usage Effectiveness by cloud region.
* :data:`WUE_BY_REGION` — Water Usage Effectiveness by cloud region.
* :data:`DEFAULT_PUE` / :data:`DEFAULT_WUE` — Global fallback values.
* :data:`REGION_METADATA` — Human-readable region labels and provider tags.
* :data:`HARDWARE_METADATA` — Human-readable hardware labels and vendor tags.

Data sources
------------
* Hardware TDP: NVIDIA, AMD, and Google official product datasheets.
* Carbon intensity: electricityMap / Our World in Data annual average grid
  carbon intensity figures (2022–2023); cloud provider sustainability
  reports where available.
* PUE: Google Environmental Report 2023; AWS Sustainability pages;
  Microsoft Azure sustainability disclosures; Uptime Institute 2023 Global
  Data Center Survey (industry average).
* WUE: Google Environmental Report 2023; AWS re:Invent sustainability
  sessions; The Green Grid WUE whitepaper.

All values are intentionally conservative estimates suitable for order-of-
magnitude impact reporting. They are NOT real-time and should not be used
for regulatory compliance purposes.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Fallback / default constants
# ---------------------------------------------------------------------------

#: Global average PUE used when a region-specific value is unavailable.
#: Source: Uptime Institute Global Data Center Survey 2023 (mean ~1.58).
DEFAULT_PUE: Final[float] = 1.58

#: Global average WUE (litres per kWh) used as fallback.
#: Source: The Green Grid / industry estimates (~1.8 L/kWh global average).
DEFAULT_WUE: Final[float] = 1.8

# ---------------------------------------------------------------------------
# Hardware TDP lookup table
# ---------------------------------------------------------------------------

#: Mapping of hardware preset key to Thermal Design Power in watts.
#:
#: These figures represent the *maximum* rated TDP for each accelerator as
#: published in vendor datasheets. Actual power draw during training workloads
#: is typically 80–100 % of TDP; inference may be lower. The utilisation
#: factor supplied by the user scales this value before the energy calculation.
#:
#: Sources
#: ~~~~~~~
#: * NVIDIA A100: https://www.nvidia.com/en-us/data-center/a100/ (400 W SXM)
#: * NVIDIA H100: https://www.nvidia.com/en-us/data-center/h100/
#:   (700 W SXM5, 350 W PCIe)
#: * NVIDIA V100: https://www.nvidia.com/en-us/data-center/v100/ (300 W SXM2)
#: * NVIDIA A10/A10G: https://www.nvidia.com/en-us/data-center/products/a10-gpu/ (150 W)
#: * NVIDIA T4: https://www.nvidia.com/en-us/data-center/tesla-t4/ (70 W)
#: * NVIDIA L4: https://www.nvidia.com/en-us/data-center/l4/ (72 W)
#: * NVIDIA L40/L40S: https://www.nvidia.com/en-us/data-center/l40s/ (300/350 W)
#: * NVIDIA RTX 4090: TDP 450 W (consumer, included for completeness)
#: * NVIDIA RTX 3090: TDP 350 W (consumer, included for completeness)
#: * NVIDIA A40: https://www.nvidia.com/en-us/data-center/a40/ (300 W)
#: * NVIDIA A30: https://www.nvidia.com/en-us/data-center/a30-gpu/ (165 W)
#: * NVIDIA A2: https://www.nvidia.com/en-us/data-center/products/a2/ (60 W)
#: * NVIDIA H200: announced TDP ~700 W (SXM)
#: * NVIDIA GH200: announced TDP ~1000 W (Grace-Hopper superchip)
#: * Google TPU v3: ~450 W per chip (Google internal estimates)
#: * Google TPU v4: 192 W per chip (Google Environmental Report 2021)
#: * Google TPU v5e: 197 W per chip (Google TPU v5e datasheet)
#: * Google TPU v5p: ~459 W per chip (Google TPU v5p datasheet)
#: * AMD MI250X: 560 W TDP (AMD product page)
#: * AMD MI300X: 750 W TDP (AMD product page)
#: * AMD MI210: 300 W TDP (AMD product page)
#: * Intel Gaudi 2: 600 W TDP (Intel Habana Labs)
#: * Intel Gaudi 3: 900 W TDP (Intel Habana Labs)
HARDWARE_TDP_WATTS: Final[dict[str, float]] = {
    # ------------------------------------------------------------------
    # NVIDIA Data-Centre GPUs
    # ------------------------------------------------------------------
    "A100_40GB": 400.0,       # A100 SXM4 40 GB
    "A100_80GB": 400.0,       # A100 SXM4 80 GB
    "H100_SXM": 700.0,        # H100 SXM5 (NVLink)
    "H100_PCIe": 350.0,       # H100 PCIe
    "H200_SXM": 700.0,        # H200 SXM5 (same board TDP as H100 SXM)
    "GH200": 1000.0,          # Grace-Hopper superchip (GPU die + Grace CPU)
    "V100_16GB": 300.0,       # V100 SXM2 16 GB
    "V100_32GB": 300.0,       # V100 SXM2 32 GB
    "A40": 300.0,             # A40 PCIe
    "A30": 165.0,             # A30
    "A10": 150.0,             # A10 PCIe
    "A10G": 150.0,            # A10G (AWS variant)
    "A2": 60.0,               # A2 (edge/inference)
    "T4": 70.0,               # Tesla T4 PCIe
    "L4": 72.0,               # L4 PCIe
    "L40": 300.0,             # L40 PCIe
    "L40S": 350.0,            # L40S PCIe
    # ------------------------------------------------------------------
    # NVIDIA Consumer / Workstation GPUs (often used in research clusters)
    # ------------------------------------------------------------------
    "RTX_4090": 450.0,        # GeForce RTX 4090
    "RTX_3090": 350.0,        # GeForce RTX 3090
    "RTX_3090Ti": 450.0,      # GeForce RTX 3090 Ti
    "RTX_A6000": 300.0,       # RTX A6000 (Ampere workstation)
    # ------------------------------------------------------------------
    # Google TPUs
    # ------------------------------------------------------------------
    "TPU_v3": 450.0,          # TPU v3 (per chip estimate)
    "TPU_v4": 192.0,          # TPU v4 (per chip, Google 2021 report)
    "TPU_v5e": 197.0,         # TPU v5e (per chip)
    "TPU_v5p": 459.0,         # TPU v5p (per chip)
    # ------------------------------------------------------------------
    # AMD GPUs
    # ------------------------------------------------------------------
    "MI250X": 560.0,          # Instinct MI250X
    "MI300X": 750.0,          # Instinct MI300X
    "MI210": 300.0,           # Instinct MI210
    # ------------------------------------------------------------------
    # Intel AI Accelerators
    # ------------------------------------------------------------------
    "Gaudi2": 600.0,          # Intel Gaudi 2 (Habana Labs)
    "Gaudi3": 900.0,          # Intel Gaudi 3 (Habana Labs)
}

# ---------------------------------------------------------------------------
# Regional carbon intensity lookup table
# ---------------------------------------------------------------------------

#: Mapping of cloud region key to grid carbon intensity in gCO2e per kWh.
#:
#: Values are annual-average figures for the relevant electricity grid zone.
#: Where cloud providers publish their own net figures (accounting for RECs
#: or PPAs), we use the *location-based* market average to remain conservative
#: and comparable across providers.
#:
#: Sources
#: ~~~~~~~
#: * US regions: U.S. EPA eGRID 2022 subregion averages.
#: * European regions: ENTSO-E / electricityMap 2022 annual averages.
#: * Asia-Pacific regions: IEA / electricityMap 2022 estimates.
#: * Cloud provider sustainability reports (cross-referenced).
CARBON_INTENSITY_G_PER_KWH: Final[dict[str, float]] = {
    # ------------------------------------------------------------------
    # AWS Regions
    # ------------------------------------------------------------------
    # United States
    "us-east-1": 415.0,        # N. Virginia — PJM grid average
    "us-east-2": 476.0,        # Ohio — RFC West subregion
    "us-west-1": 210.0,        # N. California — WECC CA (high solar/hydro)
    "us-west-2": 136.0,        # Oregon — WECC NW (heavy hydro + wind)
    "us-gov-east-1": 415.0,    # GovCloud East (co-located with us-east-1)
    "us-gov-west-1": 136.0,    # GovCloud West (co-located with us-west-2)
    # Canada
    "ca-central-1": 130.0,     # Canada Central (Quebec hydro-heavy)
    "ca-west-1": 160.0,        # Canada West (British Columbia — hydro)
    # South America
    "sa-east-1": 74.0,         # São Paulo — Brazilian grid (high hydro)
    # Europe
    "eu-west-1": 316.0,        # Ireland — SEM grid
    "eu-west-2": 233.0,        # London — UK national grid
    "eu-west-3": 56.0,         # Paris — French grid (nuclear-heavy)
    "eu-central-1": 338.0,     # Frankfurt — German grid
    "eu-central-2": 233.0,     # Zurich — Swiss grid (hydro + nuclear)
    "eu-north-1": 8.0,         # Stockholm — Nordic grid (almost zero-carbon)
    "eu-south-1": 233.0,       # Milan — Italian grid
    "eu-south-2": 165.0,       # Spain — REE grid (high renewables)
    # Middle East
    "me-south-1": 700.0,       # Bahrain — Gulf Cooperation Council grid
    "me-central-1": 500.0,     # UAE
    # Africa
    "af-south-1": 928.0,       # Cape Town — Eskom (coal-heavy)
    # Asia Pacific
    "ap-southeast-1": 493.0,   # Singapore — SING grid
    "ap-southeast-2": 610.0,   # Sydney — Australian NEM
    "ap-southeast-3": 763.0,   # Jakarta — Indonesian grid (coal-heavy)
    "ap-northeast-1": 506.0,   # Tokyo — JEPX (post-Fukushima, high fossil)
    "ap-northeast-2": 415.0,   # Seoul — Korean grid
    "ap-northeast-3": 506.0,   # Osaka — KEPCO (similar to Tokyo)
    "ap-south-1": 713.0,       # Mumbai — Indian western grid
    "ap-south-2": 713.0,       # Hyderabad — Indian southern grid
    "ap-east-1": 700.0,        # Hong Kong — CLP/HKE grid
    # ------------------------------------------------------------------
    # GCP Regions
    # ------------------------------------------------------------------
    # United States
    "us-central1": 479.0,      # Iowa — MISO Midwest
    "us-east1": 610.0,         # South Carolina — SERC Southeast (coal)
    "us-east4": 415.0,         # N. Virginia — PJM
    "us-east5": 415.0,         # Columbus, Ohio — PJM
    "us-south1": 396.0,        # Dallas, TX — ERCOT
    "us-west1": 136.0,         # The Dalles, Oregon — WECC NW
    "us-west2": 210.0,         # Los Angeles — WECC CA
    "us-west3": 396.0,         # Salt Lake City — WECC Rockies
    "us-west4": 396.0,         # Las Vegas — WECC Southwest
    # Canada
    "northamerica-northeast1": 27.0,   # Montréal — Hydro-Québec
    "northamerica-northeast2": 27.0,   # Toronto — Ontario grid
    # South America
    "southamerica-east1": 74.0,        # São Paulo
    "southamerica-west1": 160.0,       # Santiago — Chilean grid
    # Europe
    "europe-west1": 150.0,     # St. Ghislain, Belgium
    "europe-west2": 233.0,     # London
    "europe-west3": 338.0,     # Frankfurt
    "europe-west4": 284.0,     # Eemshaven, Netherlands
    "europe-west6": 56.0,      # Zurich
    "europe-west8": 233.0,     # Milan
    "europe-west9": 56.0,      # Paris
    "europe-west10": 338.0,    # Berlin
    "europe-west12": 233.0,    # Turin
    "europe-north1": 8.0,      # Hamina, Finland — Nordic grid
    "europe-central2": 681.0,  # Warsaw — Polish grid (coal-heavy)
    "europe-southwest1": 165.0, # Madrid
    # Middle East
    "me-west1": 500.0,         # Tel Aviv — Israeli grid
    "me-central1": 500.0,      # Doha — Qatar
    "me-central2": 500.0,      # Dammam — Saudi Arabia
    # Asia Pacific
    "asia-east1": 545.0,       # Changhua County, Taiwan
    "asia-east2": 700.0,       # Hong Kong
    "asia-northeast1": 506.0,  # Tokyo
    "asia-northeast2": 506.0,  # Osaka
    "asia-northeast3": 415.0,  # Seoul
    "asia-southeast1": 493.0,  # Singapore
    "asia-southeast2": 763.0,  # Jakarta
    "asia-south1": 713.0,      # Mumbai
    "asia-south2": 713.0,      # Delhi
    "australia-southeast1": 610.0,   # Sydney
    "australia-southeast2": 660.0,   # Melbourne
    # ------------------------------------------------------------------
    # Azure Regions
    # ------------------------------------------------------------------
    # United States
    "eastus": 415.0,            # East US (N. Virginia)
    "eastus2": 415.0,           # East US 2 (N. Virginia)
    "centralus": 479.0,         # Central US (Iowa)
    "northcentralus": 476.0,    # North Central US (Chicago)
    "southcentralus": 396.0,    # South Central US (Texas)
    "westcentralus": 396.0,     # West Central US (Wyoming)
    "westus": 210.0,            # West US (California)
    "westus2": 136.0,           # West US 2 (Washington/Oregon)
    "westus3": 396.0,           # West US 3 (Arizona)
    # Canada
    "canadacentral": 130.0,     # Canada Central (Toronto)
    "canadaeast": 27.0,         # Canada East (Québec)
    # South America
    "brazilsouth": 74.0,        # Brazil South (São Paulo)
    "brazilsoutheast": 74.0,    # Brazil Southeast
    # Europe
    "northeurope": 316.0,       # North Europe (Ireland)
    "westeurope": 284.0,        # West Europe (Netherlands)
    "uksouth": 233.0,           # UK South (London)
    "ukwest": 233.0,            # UK West (Cardiff)
    "francecentral": 56.0,      # France Central (Paris)
    "francesouth": 56.0,        # France South (Marseille)
    "germanywestcentral": 338.0, # Germany West Central (Frankfurt)
    "germanynorth": 338.0,      # Germany North
    "switzerlandnorth": 56.0,   # Switzerland North (Zurich)
    "switzerlandwest": 56.0,    # Switzerland West (Geneva)
    "swedencentral": 8.0,       # Sweden Central (Gävle) — near-zero carbon
    "norwayeast": 26.0,         # Norway East (Oslo) — hydro-dominated
    "norwaywest": 26.0,         # Norway West
    "finlandcentral": 8.0,      # Finland Central
    "italynorth": 233.0,        # Italy North (Milan)
    "polandcentral": 681.0,     # Poland Central (Warsaw) — very coal-heavy
    "spaincentral": 165.0,      # Spain Central (Madrid)
    # Middle East
    "uaenorth": 500.0,          # UAE North (Dubai)
    "uaecentral": 500.0,        # UAE Central (Abu Dhabi)
    "qatarcentral": 500.0,      # Qatar Central (Doha)
    "israelcentral": 500.0,     # Israel Central (Tel Aviv)
    # Africa
    "southafricanorth": 928.0,  # South Africa North (Johannesburg)
    "southafricawest": 928.0,   # South Africa West (Cape Town)
    # Asia Pacific
    "eastasia": 700.0,          # East Asia (Hong Kong)
    "southeastasia": 493.0,     # Southeast Asia (Singapore)
    "australiaeast": 610.0,     # Australia East (Sydney)
    "australiasoutheast": 660.0, # Australia Southeast (Melbourne)
    "japaneast": 506.0,         # Japan East (Tokyo)
    "japanwest": 506.0,         # Japan West (Osaka)
    "koreacentral": 415.0,      # Korea Central (Seoul)
    "koreasouth": 415.0,        # Korea South (Busan)
    "centralindia": 713.0,      # Central India (Pune)
    "southindia": 713.0,        # South India (Chennai)
    "westindia": 713.0,         # West India (Mumbai)
    "chinanorth": 537.0,        # China North (Beijing) — coal-heavy
    "chinaeast": 537.0,         # China East (Shanghai)
}

# ---------------------------------------------------------------------------
# PUE lookup table
# ---------------------------------------------------------------------------

#: Mapping of cloud region key to Power Usage Effectiveness (dimensionless).
#:
#: PUE = total facility power / IT equipment power. A value of 1.0 is ideal
#: (all power goes to IT). The global data-centre average is ~1.58 (Uptime
#: Institute 2023). Hyperscale cloud providers typically achieve 1.09–1.20.
#:
#: Sources
#: ~~~~~~~
#: * Google: Google Environmental Report 2023 (trailing 12-month fleet PUE).
#: * AWS: AWS Sustainability page and re:Invent 2022/2023 sessions.
#: * Azure: Microsoft Sustainability Report 2023.
#: * Industry average: Uptime Institute Global Data Center Survey 2023.
PUE_BY_REGION: Final[dict[str, float]] = {
    # ------------------------------------------------------------------
    # AWS Regions  (AWS reports ~1.20 fleet average)
    # ------------------------------------------------------------------
    "us-east-1": 1.20,
    "us-east-2": 1.20,
    "us-west-1": 1.20,
    "us-west-2": 1.20,
    "us-gov-east-1": 1.20,
    "us-gov-west-1": 1.20,
    "ca-central-1": 1.20,
    "ca-west-1": 1.20,
    "sa-east-1": 1.20,
    "eu-west-1": 1.20,
    "eu-west-2": 1.20,
    "eu-west-3": 1.20,
    "eu-central-1": 1.20,
    "eu-central-2": 1.20,
    "eu-north-1": 1.15,    # Stockholm benefits from free-air cooling
    "eu-south-1": 1.20,
    "eu-south-2": 1.20,
    "me-south-1": 1.25,    # Hot climate — more cooling overhead
    "me-central-1": 1.25,
    "af-south-1": 1.25,
    "ap-southeast-1": 1.25,  # Singapore — tropical climate
    "ap-southeast-2": 1.20,
    "ap-southeast-3": 1.25,
    "ap-northeast-1": 1.20,
    "ap-northeast-2": 1.20,
    "ap-northeast-3": 1.20,
    "ap-south-1": 1.25,    # Mumbai — hot and humid
    "ap-south-2": 1.25,
    "ap-east-1": 1.25,
    # ------------------------------------------------------------------
    # GCP Regions  (Google publishes per-campus trailing PUE)
    # ------------------------------------------------------------------
    "us-central1": 1.11,   # Council Bluffs, Iowa — Google campus PUE
    "us-east1": 1.11,      # Moncks Corner, SC
    "us-east4": 1.10,      # Ashburn, VA
    "us-east5": 1.10,      # Columbus, OH
    "us-south1": 1.12,
    "us-west1": 1.09,      # The Dalles, OR — mild climate, river cooling
    "us-west2": 1.11,      # Los Angeles
    "us-west3": 1.12,
    "us-west4": 1.12,
    "northamerica-northeast1": 1.08,  # Montréal — free-air cooling
    "northamerica-northeast2": 1.08,
    "southamerica-east1": 1.12,
    "southamerica-west1": 1.12,
    "europe-west1": 1.09,  # St. Ghislain, Belgium — very efficient
    "europe-west2": 1.10,  # London
    "europe-west3": 1.10,  # Frankfurt
    "europe-west4": 1.09,  # Eemshaven, Netherlands — sea air cooling
    "europe-west6": 1.10,  # Zurich
    "europe-west8": 1.10,
    "europe-west9": 1.10,
    "europe-west10": 1.10,
    "europe-west12": 1.10,
    "europe-north1": 1.07, # Hamina, Finland — sea water cooling
    "europe-central2": 1.12,
    "europe-southwest1": 1.10,
    "me-west1": 1.20,
    "me-central1": 1.20,
    "me-central2": 1.20,
    "asia-east1": 1.13,    # Changhua County, Taiwan
    "asia-east2": 1.18,    # Hong Kong
    "asia-northeast1": 1.11,  # Tokyo
    "asia-northeast2": 1.11,  # Osaka
    "asia-northeast3": 1.13,  # Seoul
    "asia-southeast1": 1.18,  # Singapore
    "asia-southeast2": 1.20,  # Jakarta
    "asia-south1": 1.20,   # Mumbai
    "asia-south2": 1.20,   # Delhi
    "australia-southeast1": 1.15,
    "australia-southeast2": 1.15,
    # ------------------------------------------------------------------
    # Azure Regions  (Microsoft targets 1.0 through renewable cooling)
    # Currently reporting fleet PUE of ~1.18
    # ------------------------------------------------------------------
    "eastus": 1.18,
    "eastus2": 1.18,
    "centralus": 1.18,
    "northcentralus": 1.18,
    "southcentralus": 1.18,
    "westcentralus": 1.18,
    "westus": 1.18,
    "westus2": 1.18,
    "westus3": 1.18,
    "canadacentral": 1.18,
    "canadaeast": 1.15,    # Québec — cooler climate
    "brazilsouth": 1.20,
    "brazilsoutheast": 1.20,
    "northeurope": 1.18,   # Dublin, Ireland
    "westeurope": 1.18,    # Amsterdam
    "uksouth": 1.18,
    "ukwest": 1.18,
    "francecentral": 1.18,
    "francesouth": 1.18,
    "germanywestcentral": 1.18,
    "germanynorth": 1.18,
    "switzerlandnorth": 1.18,
    "switzerlandwest": 1.18,
    "swedencentral": 1.12, # Gävle — sub-arctic, very efficient cooling
    "norwayeast": 1.12,    # Oslo — cold climate
    "norwaywest": 1.12,
    "finlandcentral": 1.12,
    "italynorth": 1.18,
    "polandcentral": 1.18,
    "spaincentral": 1.18,
    "uaenorth": 1.30,      # Dubai — extreme heat, high cooling overhead
    "uaecentral": 1.30,
    "qatarcentral": 1.30,
    "israelcentral": 1.22,
    "southafricanorth": 1.25,
    "southafricawest": 1.25,
    "eastasia": 1.22,
    "southeastasia": 1.25,
    "australiaeast": 1.18,
    "australiasoutheast": 1.18,
    "japaneast": 1.18,
    "japanwest": 1.18,
    "koreacentral": 1.18,
    "koreasouth": 1.18,
    "centralindia": 1.25,
    "southindia": 1.25,
    "westindia": 1.25,
    "chinanorth": 1.22,
    "chinaeast": 1.22,
}

# ---------------------------------------------------------------------------
# WUE lookup table
# ---------------------------------------------------------------------------

#: Mapping of cloud region key to Water Usage Effectiveness in litres per kWh.
#:
#: WUE captures *onsite* water consumption (primarily evaporative cooling)
#: per unit of IT energy consumed. Values vary significantly with climate,
#: cooling technology (evaporative vs. air-side vs. liquid cooling), and
#: local regulations.
#:
#: Note: Dry-bulb climate and wet-bulb temperature are the primary drivers.
#: Colder / coastal regions tend to have much lower WUE. Arid, hot climates
#: require more evaporative cooling and thus have higher WUE.
#:
#: Sources
#: ~~~~~~~
#: * Google Environmental Report 2023 (campus-level WUE figures).
#: * AWS re:Invent 2022 "Sustainability in the cloud" session.
#: * Microsoft Azure Sustainability Reports 2022–2023.
#: * The Green Grid "Water Usage Effectiveness" whitepaper.
WUE_BY_REGION: Final[dict[str, float]] = {
    # ------------------------------------------------------------------
    # AWS Regions
    # ------------------------------------------------------------------
    "us-east-1": 0.49,         # N. Virginia — moderate climate
    "us-east-2": 0.93,         # Ohio — more humid summers
    "us-west-1": 1.32,         # N. California — dry, hot summers
    "us-west-2": 0.18,         # Oregon — cool, wet climate (low WUE)
    "us-gov-east-1": 0.49,
    "us-gov-west-1": 0.18,
    "ca-central-1": 0.60,      # Toronto — moderate
    "ca-west-1": 0.30,         # British Columbia — mild and wet
    "sa-east-1": 1.20,         # São Paulo — tropical, high evaporation
    "eu-west-1": 0.25,         # Ireland — cool, rainy climate
    "eu-west-2": 0.35,         # London — mild oceanic
    "eu-west-3": 0.40,         # Paris — continental
    "eu-central-1": 0.80,      # Frankfurt — warm summers
    "eu-central-2": 0.50,      # Zurich — alpine climate
    "eu-north-1": 0.10,        # Stockholm — very cold, minimal cooling
    "eu-south-1": 0.90,        # Milan — warm Mediterranean
    "eu-south-2": 0.95,        # Madrid — hot and dry
    "me-south-1": 2.50,        # Bahrain — extreme heat and aridity
    "me-central-1": 2.20,      # UAE — hot desert
    "af-south-1": 1.80,        # Cape Town — warm, semi-arid
    "ap-southeast-1": 1.50,    # Singapore — tropical, very humid
    "ap-southeast-2": 1.10,    # Sydney — warm temperate
    "ap-southeast-3": 1.80,    # Jakarta — tropical, heavy cooling
    "ap-northeast-1": 0.90,    # Tokyo — humid subtropical
    "ap-northeast-2": 0.85,    # Seoul — continental
    "ap-northeast-3": 0.90,    # Osaka — similar to Tokyo
    "ap-south-1": 2.00,        # Mumbai — hot tropical monsoon
    "ap-south-2": 2.00,        # Hyderabad
    "ap-east-1": 1.40,         # Hong Kong — subtropical
    # ------------------------------------------------------------------
    # GCP Regions
    # ------------------------------------------------------------------
    "us-central1": 1.10,       # Iowa — Google reports ~1.10 WUE
    "us-east1": 0.80,          # South Carolina
    "us-east4": 0.80,          # N. Virginia
    "us-east5": 0.80,
    "us-south1": 1.30,         # Dallas — hot and dry
    "us-west1": 0.20,          # The Dalles, OR — river + cool climate
    "us-west2": 1.00,          # Los Angeles — warm, low humidity
    "us-west3": 1.40,          # Salt Lake City — arid
    "us-west4": 1.50,          # Las Vegas — extreme desert
    "northamerica-northeast1": 0.15,  # Montréal — very cold winters
    "northamerica-northeast2": 0.20,
    "southamerica-east1": 1.20,
    "southamerica-west1": 0.80,  # Santiago — Mediterranean climate
    "europe-west1": 0.25,      # Belgium — Google reports ~0.25
    "europe-west2": 0.35,      # London
    "europe-west3": 0.70,      # Frankfurt
    "europe-west4": 0.60,      # Netherlands — Google Eemshaven
    "europe-west6": 0.40,      # Zurich
    "europe-west8": 0.80,
    "europe-west9": 0.40,
    "europe-west10": 0.70,
    "europe-west12": 0.80,
    "europe-north1": 0.08,     # Finland — sea water cooling, near-zero WUE
    "europe-central2": 0.90,
    "europe-southwest1": 0.90,
    "me-west1": 2.00,
    "me-central1": 2.20,
    "me-central2": 2.20,
    "asia-east1": 1.20,        # Taiwan
    "asia-east2": 1.40,        # Hong Kong
    "asia-northeast1": 0.90,   # Tokyo
    "asia-northeast2": 0.90,   # Osaka
    "asia-northeast3": 0.85,   # Seoul
    "asia-southeast1": 1.50,   # Singapore
    "asia-southeast2": 1.80,   # Jakarta
    "asia-south1": 2.00,       # Mumbai
    "asia-south2": 2.00,       # Delhi
    "australia-southeast1": 1.10,
    "australia-southeast2": 1.20,
    # ------------------------------------------------------------------
    # Azure Regions
    # ------------------------------------------------------------------
    "eastus": 0.49,
    "eastus2": 0.49,
    "centralus": 1.10,
    "northcentralus": 0.80,
    "southcentralus": 1.30,
    "westcentralus": 1.20,
    "westus": 1.00,
    "westus2": 0.18,
    "westus3": 1.40,
    "canadacentral": 0.50,
    "canadaeast": 0.15,
    "brazilsouth": 1.20,
    "brazilsoutheast": 1.20,
    "northeurope": 0.25,       # Ireland
    "westeurope": 0.60,        # Netherlands
    "uksouth": 0.35,
    "ukwest": 0.30,
    "francecentral": 0.40,
    "francesouth": 0.50,
    "germanywestcentral": 0.70,
    "germanynorth": 0.60,
    "switzerlandnorth": 0.40,
    "switzerlandwest": 0.40,
    "swedencentral": 0.10,
    "norwayeast": 0.10,
    "norwaywest": 0.10,
    "finlandcentral": 0.10,
    "italynorth": 0.80,
    "polandcentral": 0.90,
    "spaincentral": 0.95,
    "uaenorth": 2.50,
    "uaecentral": 2.50,
    "qatarcentral": 2.50,
    "israelcentral": 1.80,
    "southafricanorth": 1.80,
    "southafricawest": 1.80,
    "eastasia": 1.40,
    "southeastasia": 1.50,
    "australiaeast": 1.10,
    "australiasoutheast": 1.20,
    "japaneast": 0.90,
    "japanwest": 0.90,
    "koreacentral": 0.85,
    "koreasouth": 0.85,
    "centralindia": 2.00,
    "southindia": 2.00,
    "westindia": 2.00,
    "chinanorth": 1.20,
    "chinaeast": 1.10,
}

# ---------------------------------------------------------------------------
# Region metadata (human-readable labels)
# ---------------------------------------------------------------------------

#: Mapping of region key to a dict with ``label`` (display name) and
#: ``provider`` (cloud provider tag: ``"aws"``, ``"gcp"``, or ``"azure"``).
#:
#: This is used by the UI template to build the region selector and by the
#: report module to add context to the generated report card.
REGION_METADATA: Final[dict[str, dict[str, str]]] = {
    # AWS
    "us-east-1": {"label": "AWS N. Virginia (us-east-1)", "provider": "aws"},
    "us-east-2": {"label": "AWS Ohio (us-east-2)", "provider": "aws"},
    "us-west-1": {"label": "AWS N. California (us-west-1)", "provider": "aws"},
    "us-west-2": {"label": "AWS Oregon (us-west-2)", "provider": "aws"},
    "us-gov-east-1": {"label": "AWS GovCloud East (us-gov-east-1)", "provider": "aws"},
    "us-gov-west-1": {"label": "AWS GovCloud West (us-gov-west-1)", "provider": "aws"},
    "ca-central-1": {"label": "AWS Canada Central (ca-central-1)", "provider": "aws"},
    "ca-west-1": {"label": "AWS Canada West (ca-west-1)", "provider": "aws"},
    "sa-east-1": {"label": "AWS São Paulo (sa-east-1)", "provider": "aws"},
    "eu-west-1": {"label": "AWS Ireland (eu-west-1)", "provider": "aws"},
    "eu-west-2": {"label": "AWS London (eu-west-2)", "provider": "aws"},
    "eu-west-3": {"label": "AWS Paris (eu-west-3)", "provider": "aws"},
    "eu-central-1": {"label": "AWS Frankfurt (eu-central-1)", "provider": "aws"},
    "eu-central-2": {"label": "AWS Zurich (eu-central-2)", "provider": "aws"},
    "eu-north-1": {"label": "AWS Stockholm (eu-north-1)", "provider": "aws"},
    "eu-south-1": {"label": "AWS Milan (eu-south-1)", "provider": "aws"},
    "eu-south-2": {"label": "AWS Spain (eu-south-2)", "provider": "aws"},
    "me-south-1": {"label": "AWS Bahrain (me-south-1)", "provider": "aws"},
    "me-central-1": {"label": "AWS UAE (me-central-1)", "provider": "aws"},
    "af-south-1": {"label": "AWS Cape Town (af-south-1)", "provider": "aws"},
    "ap-southeast-1": {"label": "AWS Singapore (ap-southeast-1)", "provider": "aws"},
    "ap-southeast-2": {"label": "AWS Sydney (ap-southeast-2)", "provider": "aws"},
    "ap-southeast-3": {"label": "AWS Jakarta (ap-southeast-3)", "provider": "aws"},
    "ap-northeast-1": {"label": "AWS Tokyo (ap-northeast-1)", "provider": "aws"},
    "ap-northeast-2": {"label": "AWS Seoul (ap-northeast-2)", "provider": "aws"},
    "ap-northeast-3": {"label": "AWS Osaka (ap-northeast-3)", "provider": "aws"},
    "ap-south-1": {"label": "AWS Mumbai (ap-south-1)", "provider": "aws"},
    "ap-south-2": {"label": "AWS Hyderabad (ap-south-2)", "provider": "aws"},
    "ap-east-1": {"label": "AWS Hong Kong (ap-east-1)", "provider": "aws"},
    # GCP
    "us-central1": {"label": "GCP Iowa (us-central1)", "provider": "gcp"},
    "us-east1": {"label": "GCP South Carolina (us-east1)", "provider": "gcp"},
    "us-east4": {"label": "GCP N. Virginia (us-east4)", "provider": "gcp"},
    "us-east5": {"label": "GCP Columbus, OH (us-east5)", "provider": "gcp"},
    "us-south1": {"label": "GCP Dallas (us-south1)", "provider": "gcp"},
    "us-west1": {"label": "GCP Oregon (us-west1)", "provider": "gcp"},
    "us-west2": {"label": "GCP Los Angeles (us-west2)", "provider": "gcp"},
    "us-west3": {"label": "GCP Salt Lake City (us-west3)", "provider": "gcp"},
    "us-west4": {"label": "GCP Las Vegas (us-west4)", "provider": "gcp"},
    "northamerica-northeast1": {"label": "GCP Montréal (northamerica-northeast1)", "provider": "gcp"},
    "northamerica-northeast2": {"label": "GCP Toronto (northamerica-northeast2)", "provider": "gcp"},
    "southamerica-east1": {"label": "GCP São Paulo (southamerica-east1)", "provider": "gcp"},
    "southamerica-west1": {"label": "GCP Santiago (southamerica-west1)", "provider": "gcp"},
    "europe-west1": {"label": "GCP Belgium (europe-west1)", "provider": "gcp"},
    "europe-west2": {"label": "GCP London (europe-west2)", "provider": "gcp"},
    "europe-west3": {"label": "GCP Frankfurt (europe-west3)", "provider": "gcp"},
    "europe-west4": {"label": "GCP Netherlands (europe-west4)", "provider": "gcp"},
    "europe-west6": {"label": "GCP Zurich (europe-west6)", "provider": "gcp"},
    "europe-west8": {"label": "GCP Milan (europe-west8)", "provider": "gcp"},
    "europe-west9": {"label": "GCP Paris (europe-west9)", "provider": "gcp"},
    "europe-west10": {"label": "GCP Berlin (europe-west10)", "provider": "gcp"},
    "europe-west12": {"label": "GCP Turin (europe-west12)", "provider": "gcp"},
    "europe-north1": {"label": "GCP Finland (europe-north1)", "provider": "gcp"},
    "europe-central2": {"label": "GCP Warsaw (europe-central2)", "provider": "gcp"},
    "europe-southwest1": {"label": "GCP Madrid (europe-southwest1)", "provider": "gcp"},
    "me-west1": {"label": "GCP Tel Aviv (me-west1)", "provider": "gcp"},
    "me-central1": {"label": "GCP Doha (me-central1)", "provider": "gcp"},
    "me-central2": {"label": "GCP Dammam (me-central2)", "provider": "gcp"},
    "asia-east1": {"label": "GCP Taiwan (asia-east1)", "provider": "gcp"},
    "asia-east2": {"label": "GCP Hong Kong (asia-east2)", "provider": "gcp"},
    "asia-northeast1": {"label": "GCP Tokyo (asia-northeast1)", "provider": "gcp"},
    "asia-northeast2": {"label": "GCP Osaka (asia-northeast2)", "provider": "gcp"},
    "asia-northeast3": {"label": "GCP Seoul (asia-northeast3)", "provider": "gcp"},
    "asia-southeast1": {"label": "GCP Singapore (asia-southeast1)", "provider": "gcp"},
    "asia-southeast2": {"label": "GCP Jakarta (asia-southeast2)", "provider": "gcp"},
    "asia-south1": {"label": "GCP Mumbai (asia-south1)", "provider": "gcp"},
    "asia-south2": {"label": "GCP Delhi (asia-south2)", "provider": "gcp"},
    "australia-southeast1": {"label": "GCP Sydney (australia-southeast1)", "provider": "gcp"},
    "australia-southeast2": {"label": "GCP Melbourne (australia-southeast2)", "provider": "gcp"},
    # Azure
    "eastus": {"label": "Azure East US (Virginia)", "provider": "azure"},
    "eastus2": {"label": "Azure East US 2 (Virginia)", "provider": "azure"},
    "centralus": {"label": "Azure Central US (Iowa)", "provider": "azure"},
    "northcentralus": {"label": "Azure North Central US (Chicago)", "provider": "azure"},
    "southcentralus": {"label": "Azure South Central US (Texas)", "provider": "azure"},
    "westcentralus": {"label": "Azure West Central US (Wyoming)", "provider": "azure"},
    "westus": {"label": "Azure West US (California)", "provider": "azure"},
    "westus2": {"label": "Azure West US 2 (Washington)", "provider": "azure"},
    "westus3": {"label": "Azure West US 3 (Arizona)", "provider": "azure"},
    "canadacentral": {"label": "Azure Canada Central (Toronto)", "provider": "azure"},
    "canadaeast": {"label": "Azure Canada East (Québec)", "provider": "azure"},
    "brazilsouth": {"label": "Azure Brazil South (São Paulo)", "provider": "azure"},
    "brazilsoutheast": {"label": "Azure Brazil Southeast", "provider": "azure"},
    "northeurope": {"label": "Azure North Europe (Ireland)", "provider": "azure"},
    "westeurope": {"label": "Azure West Europe (Netherlands)", "provider": "azure"},
    "uksouth": {"label": "Azure UK South (London)", "provider": "azure"},
    "ukwest": {"label": "Azure UK West (Cardiff)", "provider": "azure"},
    "francecentral": {"label": "Azure France Central (Paris)", "provider": "azure"},
    "francesouth": {"label": "Azure France South (Marseille)", "provider": "azure"},
    "germanywestcentral": {"label": "Azure Germany West Central (Frankfurt)", "provider": "azure"},
    "germanynorth": {"label": "Azure Germany North", "provider": "azure"},
    "switzerlandnorth": {"label": "Azure Switzerland North (Zurich)", "provider": "azure"},
    "switzerlandwest": {"label": "Azure Switzerland West (Geneva)", "provider": "azure"},
    "swedencentral": {"label": "Azure Sweden Central (Gävle)", "provider": "azure"},
    "norwayeast": {"label": "Azure Norway East (Oslo)", "provider": "azure"},
    "norwaywest": {"label": "Azure Norway West", "provider": "azure"},
    "finlandcentral": {"label": "Azure Finland Central", "provider": "azure"},
    "italynorth": {"label": "Azure Italy North (Milan)", "provider": "azure"},
    "polandcentral": {"label": "Azure Poland Central (Warsaw)", "provider": "azure"},
    "spaincentral": {"label": "Azure Spain Central (Madrid)", "provider": "azure"},
    "uaenorth": {"label": "Azure UAE North (Dubai)", "provider": "azure"},
    "uaecentral": {"label": "Azure UAE Central (Abu Dhabi)", "provider": "azure"},
    "qatarcentral": {"label": "Azure Qatar Central (Doha)", "provider": "azure"},
    "israelcentral": {"label": "Azure Israel Central (Tel Aviv)", "provider": "azure"},
    "southafricanorth": {"label": "Azure South Africa North (Johannesburg)", "provider": "azure"},
    "southafricawest": {"label": "Azure South Africa West (Cape Town)", "provider": "azure"},
    "eastasia": {"label": "Azure East Asia (Hong Kong)", "provider": "azure"},
    "southeastasia": {"label": "Azure Southeast Asia (Singapore)", "provider": "azure"},
    "australiaeast": {"label": "Azure Australia East (Sydney)", "provider": "azure"},
    "australiasoutheast": {"label": "Azure Australia Southeast (Melbourne)", "provider": "azure"},
    "japaneast": {"label": "Azure Japan East (Tokyo)", "provider": "azure"},
    "japanwest": {"label": "Azure Japan West (Osaka)", "provider": "azure"},
    "koreacentral": {"label": "Azure Korea Central (Seoul)", "provider": "azure"},
    "koreasouth": {"label": "Azure Korea South (Busan)", "provider": "azure"},
    "centralindia": {"label": "Azure Central India (Pune)", "provider": "azure"},
    "southindia": {"label": "Azure South India (Chennai)", "provider": "azure"},
    "westindia": {"label": "Azure West India (Mumbai)", "provider": "azure"},
    "chinanorth": {"label": "Azure China North (Beijing)", "provider": "azure"},
    "chinaeast": {"label": "Azure China East (Shanghai)", "provider": "azure"},
}

# ---------------------------------------------------------------------------
# Hardware metadata (human-readable labels)
# ---------------------------------------------------------------------------

#: Mapping of hardware preset key to a dict with ``label`` (display name)
#: and ``vendor`` tag (``"nvidia"``, ``"amd"``, ``"google"``, ``"intel"``).
#:
#: Used by the UI template to build the hardware selector.
HARDWARE_METADATA: Final[dict[str, dict[str, str]]] = {
    # NVIDIA
    "A100_40GB": {"label": "NVIDIA A100 40 GB SXM4", "vendor": "nvidia"},
    "A100_80GB": {"label": "NVIDIA A100 80 GB SXM4", "vendor": "nvidia"},
    "H100_SXM": {"label": "NVIDIA H100 SXM5 (NVLink)", "vendor": "nvidia"},
    "H100_PCIe": {"label": "NVIDIA H100 PCIe", "vendor": "nvidia"},
    "H200_SXM": {"label": "NVIDIA H200 SXM5", "vendor": "nvidia"},
    "GH200": {"label": "NVIDIA GH200 Grace-Hopper", "vendor": "nvidia"},
    "V100_16GB": {"label": "NVIDIA V100 16 GB SXM2", "vendor": "nvidia"},
    "V100_32GB": {"label": "NVIDIA V100 32 GB SXM2", "vendor": "nvidia"},
    "A40": {"label": "NVIDIA A40 PCIe", "vendor": "nvidia"},
    "A30": {"label": "NVIDIA A30", "vendor": "nvidia"},
    "A10": {"label": "NVIDIA A10 PCIe", "vendor": "nvidia"},
    "A10G": {"label": "NVIDIA A10G (AWS)", "vendor": "nvidia"},
    "A2": {"label": "NVIDIA A2 (edge/inference)", "vendor": "nvidia"},
    "T4": {"label": "NVIDIA Tesla T4", "vendor": "nvidia"},
    "L4": {"label": "NVIDIA L4", "vendor": "nvidia"},
    "L40": {"label": "NVIDIA L40 PCIe", "vendor": "nvidia"},
    "L40S": {"label": "NVIDIA L40S PCIe", "vendor": "nvidia"},
    "RTX_4090": {"label": "NVIDIA GeForce RTX 4090", "vendor": "nvidia"},
    "RTX_3090": {"label": "NVIDIA GeForce RTX 3090", "vendor": "nvidia"},
    "RTX_3090Ti": {"label": "NVIDIA GeForce RTX 3090 Ti", "vendor": "nvidia"},
    "RTX_A6000": {"label": "NVIDIA RTX A6000", "vendor": "nvidia"},
    # Google
    "TPU_v3": {"label": "Google TPU v3", "vendor": "google"},
    "TPU_v4": {"label": "Google TPU v4", "vendor": "google"},
    "TPU_v5e": {"label": "Google TPU v5e", "vendor": "google"},
    "TPU_v5p": {"label": "Google TPU v5p", "vendor": "google"},
    # AMD
    "MI250X": {"label": "AMD Instinct MI250X", "vendor": "amd"},
    "MI300X": {"label": "AMD Instinct MI300X", "vendor": "amd"},
    "MI210": {"label": "AMD Instinct MI210", "vendor": "amd"},
    # Intel
    "Gaudi2": {"label": "Intel Gaudi 2 (Habana Labs)", "vendor": "intel"},
    "Gaudi3": {"label": "Intel Gaudi 3 (Habana Labs)", "vendor": "intel"},
}
