# 🌍 DC Impact

**DC Impact** is a self-hosted web calculator that estimates the
environmental footprint of AI model training and inference workloads.
Given hardware type, cloud region, duration, and workload parameters it
computes:

| Metric | Unit |
|--------|------|
| Energy consumption | kWh |
| Carbon emissions | kg CO₂e |
| Water consumption | litres |

Results are rendered as a shareable **impact report card** — a styled
HTML badge and a structured JSON summary — ready for inclusion in
Hugging Face model cards, GitHub READMEs, or research papers.

> **Disclaimer** — all values are order-of-magnitude estimates based on
> publicly available hardware TDP and grid data. They are _not_ suitable
> for regulatory compliance reporting.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Self-Hosting](#self-hosting)
3. [REST API](#rest-api)
4. [Embed Usage](#embed-usage)
5. [Supported Hardware](#supported-hardware)
6. [Supported Regions](#supported-regions)
7. [Calculation Methodology](#calculation-methodology)
8. [Development](#development)
9. [Running Tests](#running-tests)
10. [License](#license)

---

## Quick Start

### Prerequisites

- Python 3.10 or later
- `pip` (or your preferred package manager)

### Install and run in 60 seconds

```bash
# 1. Clone the repository
git clone https://github.com/your-org/dc_impact.git
cd dc_impact

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install the package and its dependencies
pip install -e .

# 4. Start the development server
dc_impact
# → Listening on http://127.0.0.1:5000
```

Open <http://127.0.0.1:5000> in your browser, fill in the form, and click
**Calculate Environmental Impact**.

---

## Self-Hosting

### Development server

The built-in Flask development server is fine for local use:

```bash
dc_impact
# or equivalently:
flask --app dc_impact run --debug
```

### Production deployment with Gunicorn

For a production deployment install Gunicorn and run:

```bash
pip install gunicorn
gunicorn 'dc_impact:create_app()' \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --access-logfile -
```

### Docker

```dockerfile
# Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir gunicorn -e .
EXPOSE 8000
CMD ["gunicorn", "dc_impact:create_app()", "--bind", "0.0.0.0:8000", "--workers", "4"]
```

```bash
docker build -t dc_impact .
docker run -p 8000:8000 dc_impact
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `dev-secret-change-in-production` | Flask secret key — **always override in production** |
| `FLASK_DEBUG` | `0` | Set to `1` to enable debug mode |
| `PORT` | `8000` | Listening port (when using the Docker CMD above) |

Set via a `.env` file or your container orchestrator's secret management.

---

## REST API

DC Impact exposes a simple JSON API for programmatic use in CI/CD
pipelines, training scripts, or model reporting workflows.

### `POST /api/calculate`

Compute the environmental impact for a given workload.

**Request body** (JSON):

```json
{
  "hardware": "A100_80GB",
  "region": "us-east-1",
  "duration_hours": 72,
  "num_accelerators": 8,
  "workload_type": "training",
  "utilization": 1.0
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `hardware` | string | ✅ | — | Hardware preset key (e.g. `"A100_80GB"`) |
| `region` | string | ✅ | — | Cloud region key (e.g. `"us-east-1"`) |
| `duration_hours` | number | ✅ | — | Wall-clock duration in hours (> 0) |
| `num_accelerators` | integer | ❌ | `1` | Number of accelerator devices |
| `workload_type` | string | ❌ | `"training"` | `"training"` or `"inference"` |
| `utilization` | number | ❌ | `1.0` | Fractional GPU utilisation `(0, 1]` **or** percentage `(1, 100]` |

**Example — cURL**:

```bash
curl -s -X POST http://localhost:5000/api/calculate \
  -H 'Content-Type: application/json' \
  -d '{
    "hardware": "A100_80GB",
    "region": "us-east-1",
    "duration_hours": 72,
    "num_accelerators": 8,
    "workload_type": "training",
    "utilization": 1.0
  }' | python -m json.tool
```

**Example — Python**:

```python
import requests

response = requests.post(
    "http://localhost:5000/api/calculate",
    json={
        "hardware": "A100_80GB",
        "region": "us-east-1",
        "duration_hours": 72,
        "num_accelerators": 8,
        "workload_type": "training",
        "utilization": 1.0,
    },
)
report = response.json()
print(f"Energy:  {report['metrics']['energy_kwh']:.2f} kWh")
print(f"CO₂e:    {report['metrics']['co2e_kg']:.3f} kg")
print(f"Water:   {report['metrics']['water_liters']:.2f} L")
```

**Response** (200 OK, truncated):

```json
{
  "schema_version": "1.0",
  "generated_at": "2024-06-01T12:34:56.789012Z",
  "inputs": {
    "hardware": "A100_80GB",
    "region": "us-east-1",
    "duration_hours": 72.0,
    "num_accelerators": 8,
    "workload_type": "training",
    "utilization": 1.0,
    "utilization_pct": 100.0
  },
  "metrics": {
    "energy_kwh": 3317.76,
    "energy_mwh": 3.31776,
    "co2e_kg": 1376.87,
    "co2e_g": 1376870.0,
    "co2e_tonnes": 1.37687,
    "water_liters": 1625.51,
    "water_ml": 1625506.0
  },
  "factors": {
    "tdp_watts": 400.0,
    "pue": 1.2,
    "wue_l_per_kwh": 0.49,
    "carbon_intensity_g_per_kwh": 415.0,
    "effective_power_per_accelerator_w": 480.0
  },
  "comparisons": {
    "co2e": {
      "car_km_equivalent": 8099.24,
      "car_km_equivalent_description": "Equivalent to driving a petrol car approx. 8,099.2 km",
      "tree_sequestration_months": 758.63,
      "tree_sequestration_months_description": "Would take one tree approx. 758.6 months to sequester"
    },
    "energy": {
      "us_household_days_equivalent": 115.32,
      "us_household_days_equivalent_description": "Equivalent to approx. 115.3 days of average US household electricity consumption"
    },
    "water": {
      "showers_equivalent": 25.4,
      "showers_equivalent_description": "Equivalent to approx. 25.4 eight-minute showers"
    },
    "disclaimer": "Comparisons are approximate and intended for illustrative purposes only."
  }
}
```

**Error responses**:

| Status | Condition |
|--------|-----------|
| `400` | Missing required fields, unknown hardware/region key, invalid parameter values |
| `500` | Unexpected server error |

Error body: `{"error": "<human-readable description>"}`

---

### `GET /api/hardware`

List all supported hardware presets.

```bash
curl http://localhost:5000/api/hardware | python -m json.tool
```

```json
{
  "count": 32,
  "hardware": [
    {"key": "A100_40GB", "label": "NVIDIA A100 40 GB SXM4", "vendor": "nvidia", "tdp_watts": 400.0},
    {"key": "A100_80GB", "label": "NVIDIA A100 80 GB SXM4", "vendor": "nvidia", "tdp_watts": 400.0},
    "..."
  ]
}
```

### `GET /api/regions`

List all supported cloud regions.

```bash
curl http://localhost:5000/api/regions | python -m json.tool
```

```json
{
  "count": 120,
  "regions": [
    {
      "key": "eu-north-1",
      "label": "AWS Stockholm (eu-north-1)",
      "provider": "aws",
      "carbon_intensity_g_per_kwh": 8.0,
      "pue": 1.15,
      "wue": 0.1
    },
    "..."
  ]
}
```

---

## Embed Usage

### Hugging Face model card

Add the JSON report to your `README.md` under a `## Environmental Impact`
heading:

````markdown
## Environmental Impact

Generated with [DC Impact](https://github.com/your-org/dc_impact).

```json
{
  "schema_version": "1.0",
  "metrics": {
    "energy_kwh": 276.48,
    "co2e_kg": 37.60,
    "water_liters": 49.77
  },
  "inputs": {
    "hardware": "A100_80GB",
    "region": "us-west-2",
    "duration_hours": 72,
    "num_accelerators": 8,
    "workload_type": "training"
  }
}
```
````

### GitHub README badge / HTML snippet

The report card page provides a pre-rendered HTML snippet you can paste
directly into a `README.md` (GitHub renders raw HTML in READMEs):

1. Submit the form at your DC Impact instance.
2. On the results page click the **HTML Snippet** tab.
3. Copy and paste the `<div class="report-card">…</div>` block into your
   repository's `README.md`.

### CI/CD integration

Automatically report training footprint from your training script:

```python
# At the end of your training run:
import requests
import os

DC_IMPACT_URL = os.getenv("DC_IMPACT_URL", "http://localhost:5000")

report = requests.post(
    f"{DC_IMPACT_URL}/api/calculate",
    json={
        "hardware": "A100_80GB",
        "region": os.getenv("CLOUD_REGION", "us-east-1"),
        "duration_hours": training_wall_time_hours,
        "num_accelerators": num_gpus,
        "workload_type": "training",
        "utilization": avg_gpu_utilization,   # float 0–1
    },
    timeout=10,
).json()

print(
    f"Training footprint: "
    f"{report['metrics']['energy_kwh']:.1f} kWh, "
    f"{report['metrics']['co2e_kg']:.2f} kg CO2e, "
    f"{report['metrics']['water_liters']:.1f} L water"
)

# Optionally save the report as an artifact:
with open("impact_report.json", "w") as f:
    import json
    json.dump(report, f, indent=2)
```

---

## Supported Hardware

| Key | Label | Vendor | TDP (W) |
|-----|-------|--------|--------|
| `A100_40GB` | NVIDIA A100 40 GB SXM4 | NVIDIA | 400 |
| `A100_80GB` | NVIDIA A100 80 GB SXM4 | NVIDIA | 400 |
| `H100_SXM` | NVIDIA H100 SXM5 | NVIDIA | 700 |
| `H100_PCIe` | NVIDIA H100 PCIe | NVIDIA | 350 |
| `H200_SXM` | NVIDIA H200 SXM5 | NVIDIA | 700 |
| `GH200` | NVIDIA GH200 Grace-Hopper | NVIDIA | 1000 |
| `V100_16GB` | NVIDIA V100 16 GB | NVIDIA | 300 |
| `V100_32GB` | NVIDIA V100 32 GB | NVIDIA | 300 |
| `A40` | NVIDIA A40 | NVIDIA | 300 |
| `A30` | NVIDIA A30 | NVIDIA | 165 |
| `A10` | NVIDIA A10 | NVIDIA | 150 |
| `A10G` | NVIDIA A10G (AWS) | NVIDIA | 150 |
| `A2` | NVIDIA A2 | NVIDIA | 60 |
| `T4` | NVIDIA Tesla T4 | NVIDIA | 70 |
| `L4` | NVIDIA L4 | NVIDIA | 72 |
| `L40` | NVIDIA L40 | NVIDIA | 300 |
| `L40S` | NVIDIA L40S | NVIDIA | 350 |
| `RTX_4090` | NVIDIA RTX 4090 | NVIDIA | 450 |
| `RTX_3090` | NVIDIA RTX 3090 | NVIDIA | 350 |
| `RTX_3090Ti` | NVIDIA RTX 3090 Ti | NVIDIA | 450 |
| `RTX_A6000` | NVIDIA RTX A6000 | NVIDIA | 300 |
| `TPU_v3` | Google TPU v3 | Google | 450 |
| `TPU_v4` | Google TPU v4 | Google | 192 |
| `TPU_v5e` | Google TPU v5e | Google | 197 |
| `TPU_v5p` | Google TPU v5p | Google | 459 |
| `MI250X` | AMD Instinct MI250X | AMD | 560 |
| `MI300X` | AMD Instinct MI300X | AMD | 750 |
| `MI210` | AMD Instinct MI210 | AMD | 300 |
| `Gaudi2` | Intel Gaudi 2 | Intel | 600 |
| `Gaudi3` | Intel Gaudi 3 | Intel | 900 |

Query the live list: `GET /api/hardware`

---

## Supported Regions

Over **120 cloud regions** across AWS, GCP, and Azure are supported.

Highlighted examples:

| Key | Provider | Carbon Intensity (gCO₂/kWh) | Notes |
|-----|----------|----------------------------|-------|
| `eu-north-1` | AWS | 8 | Stockholm — near-zero carbon |
| `us-west-2` | AWS | 136 | Oregon — hydro-heavy |
| `europe-west1` | GCP | 150 | Belgium — nuclear + renewables |
| `us-east-1` | AWS | 415 | N. Virginia — PJM grid |
| `me-south-1` | AWS | 700 | Bahrain — Gulf grid |
| `af-south-1` | AWS | 928 | Cape Town — coal-heavy |

Query the live list: `GET /api/regions`

---

## Calculation Methodology

DC Impact uses a simple, transparent, three-step pipeline:

### 1. Power draw (W)

```
raw_power_kw = (TDP_watts × num_accelerators × utilization) / 1000
```

where `TDP_watts` is the hardware's Thermal Design Power from vendor
datasheets and `utilization ∈ (0, 1]` scales it for partial load.

### 2. Facility energy (kWh)

```
energy_kwh = raw_power_kw × duration_hours × PUE
```

where `PUE` (Power Usage Effectiveness) accounts for data-centre cooling
and other overhead. Values are sourced from cloud provider sustainability
reports and the Uptime Institute Global Data Center Survey 2023.

### 3. Environmental metrics

```
co2e_kg      = energy_kwh × carbon_intensity_g_per_kwh / 1000
water_liters = energy_kwh × WUE
```

- **`carbon_intensity_g_per_kwh`** — location-based grid average in
  gCO₂e/kWh from electricityMap / US EPA eGRID 2022 / IEA 2022.
- **`WUE`** (Water Usage Effectiveness, L/kWh) — onsite cooling water per
  unit of IT energy, from Google/AWS/Azure sustainability reports.

### Data sources

| Data | Sources |
|------|---------|
| Hardware TDP | Vendor datasheets (NVIDIA, AMD, Google, Intel) |
| Carbon intensity | US EPA eGRID 2022, electricityMap, IEA, ENTSO-E |
| PUE | Google Environmental Report 2023, AWS Sustainability, Microsoft Sustainability Report 2023, Uptime Institute 2023 |
| WUE | Google Environmental Report 2023, AWS re:Invent 2022/2023, The Green Grid WUE whitepaper |

---

## Development

### Project layout

```
dc_impact/
├── __init__.py          # Package init, exposes create_app
├── app.py               # Flask application factory & routes
├── calculator.py        # Core computation engine
├── data.py              # Static lookup tables (TDP, CI, PUE, WUE)
├── report.py            # JSON & HTML report generation
├── templates/
│   ├── index.html       # Calculator form UI
│   ├── report_card.html # Embeddable report card fragment
│   └── report_page.html # Full report page wrapper
└── static/
    └── style.css        # Stylesheet
tests/
├── test_calculator.py   # Unit tests for computation engine
├── test_report.py       # Unit tests for report generation
├── test_app.py          # Integration tests for Flask routes
└── test_data.py         # Sanity checks for lookup tables
pyproject.toml
README.md
```

### Install development dependencies

```bash
pip install -e ".[dev]"
# or simply:
pip install -e . pytest pytest-flask
```

### Adding a new hardware preset

1. Add an entry to `HARDWARE_TDP_WATTS` in `dc_impact/data.py`.
2. Add a corresponding entry to `HARDWARE_METADATA`.
3. Add a `<option>` to `dc_impact/templates/index.html` (optional — the
   template auto-generates options from metadata if `hardware_by_vendor`
   is non-empty).
4. Run the test suite to verify consistency.

### Adding a new cloud region

1. Add entries to `CARBON_INTENSITY_G_PER_KWH`, `PUE_BY_REGION`,
   `WUE_BY_REGION`, and `REGION_METADATA` in `dc_impact/data.py`.
2. Run the test suite — `tests/test_data.py` will catch any missing or
   inconsistent entries.

---

## Running Tests

```bash
# Run the full test suite
pytest

# With verbose output
pytest -v

# Run only calculator unit tests
pytest tests/test_calculator.py -v

# Run only integration tests
pytest tests/test_app.py -v

# Run with coverage (requires pytest-cov)
pip install pytest-cov
pytest --cov=dc_impact --cov-report=term-missing
```

The test suite covers:

- **Calculator** — reference values, edge cases, input validation, PUE/WUE
  fallback behaviour.
- **Report** — JSON schema structure, HTML fragment correctness, Jinja2
  custom filters.
- **Flask routes** — form submission, API endpoint, error handlers, known
  numeric reference values.
- **Data tables** — structural consistency (every PUE region has a
  corresponding CI entry, etc.) and spot-check values against published
  sources.

---

## License

MIT License — see `LICENSE` for details.

---

*Built with [Flask](https://flask.palletsprojects.com/) and
[Jinja2](https://jinja.palletsprojects.com/). Zero external API
dependencies — all data is bundled.*
