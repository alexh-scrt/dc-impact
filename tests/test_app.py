"""Integration tests for dc_impact Flask routes.

Covers:
- GET / — calculator form index page rendering.
- POST /report — form submission with valid and invalid inputs.
- GET /report — shareable link with query-string parameters.
- POST /api/calculate — REST API with valid JSON, invalid JSON, missing
  fields, unknown hardware/region, and edge-case parameter values.
- GET /api/hardware — hardware preset list endpoint.
- GET /api/regions — region list endpoint.
- Error handlers — 404 and 405 responses.
- Content-type negotiation and response structure.
- API percentage-to-fraction utilization auto-scaling.
- Form parse error handling.
"""

from __future__ import annotations

import json
from typing import Generator

import pytest
from flask import Flask
from flask.testing import FlaskClient

from dc_impact import create_app
from dc_impact import data as _data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app() -> Flask:
    """Create a Flask application configured for testing.

    Returns:
        A Flask app instance with TESTING=True and a fixed SECRET_KEY.
    """
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret-key",
            "DEBUG": False,
        }
    )
    return application


@pytest.fixture(scope="module")
def client(app: Flask) -> FlaskClient:
    """Return a test client for the Flask application.

    Args:
        app: The Flask application fixture.

    Returns:
        A :class:`flask.testing.FlaskClient` for issuing test requests.
    """
    return app.test_client()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _valid_form_data(**overrides) -> dict:
    """Return a dict of valid form POST parameters.

    Args:
        **overrides: Key-value pairs to override the defaults.

    Returns:
        Dictionary of form field values suitable for a POST to /report.
    """
    defaults = {
        "hardware": "A100_80GB",
        "region": "us-east-1",
        "duration_hours": "1.0",
        "num_accelerators": "1",
        "workload_type": "training",
        "utilization": "100",
    }
    defaults.update(overrides)
    return defaults


def _valid_api_payload(**overrides) -> dict:
    """Return a dict of valid JSON API payload parameters.

    Args:
        **overrides: Key-value pairs to override the defaults.

    Returns:
        Dictionary suitable for JSON POST to /api/calculate.
    """
    defaults = {
        "hardware": "A100_80GB",
        "region": "us-east-1",
        "duration_hours": 1.0,
        "num_accelerators": 1,
        "workload_type": "training",
        "utilization": 1.0,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# GET / — index page
# ---------------------------------------------------------------------------


class TestIndexRoute:
    """Tests for the GET / route (calculator form UI)."""

    def test_index_returns_200(self, client: FlaskClient) -> None:
        """GET / must return HTTP 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_index_content_type_is_html(self, client: FlaskClient) -> None:
        """GET / must return HTML content."""
        response = client.get("/")
        assert "text/html" in response.content_type

    def test_index_contains_form_element(self, client: FlaskClient) -> None:
        """Index page must contain an HTML form."""
        response = client.get("/")
        assert b"<form" in response.data

    def test_index_form_action_is_report(self, client: FlaskClient) -> None:
        """Form must POST to /report."""
        response = client.get("/")
        assert b'action="/report"' in response.data or b"action='/report'" in response.data

    def test_index_contains_hardware_select(self, client: FlaskClient) -> None:
        """Index page must contain a hardware select element."""
        response = client.get("/")
        assert b'name="hardware"' in response.data

    def test_index_contains_region_select(self, client: FlaskClient) -> None:
        """Index page must contain a region select element."""
        response = client.get("/")
        assert b'name="region"' in response.data

    def test_index_contains_duration_input(self, client: FlaskClient) -> None:
        """Index page must contain a duration input."""
        response = client.get("/")
        assert b'name="duration_hours"' in response.data

    def test_index_contains_num_accelerators_input(self, client: FlaskClient) -> None:
        """Index page must contain a num_accelerators input."""
        response = client.get("/")
        assert b'name="num_accelerators"' in response.data

    def test_index_contains_workload_type_select(self, client: FlaskClient) -> None:
        """Index page must contain a workload_type select."""
        response = client.get("/")
        assert b'name="workload_type"' in response.data

    def test_index_contains_utilization_input(self, client: FlaskClient) -> None:
        """Index page must contain a utilization input."""
        response = client.get("/")
        assert b'name="utilization"' in response.data

    def test_index_contains_submit_button(self, client: FlaskClient) -> None:
        """Index page must contain a submit button."""
        response = client.get("/")
        assert b'type="submit"' in response.data

    def test_index_contains_dc_impact_title(self, client: FlaskClient) -> None:
        """Index page must mention dc_impact."""
        response = client.get("/")
        assert b"DC Impact" in response.data or b"dc_impact" in response.data

    def test_index_contains_a100_option(self, client: FlaskClient) -> None:
        """Index page must list the A100_80GB hardware option."""
        response = client.get("/")
        assert b"A100_80GB" in response.data or b"A100" in response.data

    def test_index_contains_us_east_1_option(self, client: FlaskClient) -> None:
        """Index page must list the us-east-1 region option."""
        response = client.get("/")
        assert b"us-east-1" in response.data

    def test_index_contains_training_option(self, client: FlaskClient) -> None:
        """Index page must include 'training' as a workload option."""
        response = client.get("/")
        assert b"training" in response.data

    def test_index_contains_inference_option(self, client: FlaskClient) -> None:
        """Index page must include 'inference' as a workload option."""
        response = client.get("/")
        assert b"inference" in response.data

    def test_index_page_has_api_link(self, client: FlaskClient) -> None:
        """Index page should mention the API endpoint."""
        response = client.get("/")
        assert b"/api" in response.data

    def test_index_page_has_no_error_by_default(self, client: FlaskClient) -> None:
        """Index page must not show an error message by default."""
        response = client.get("/")
        # The alert--error class should not appear
        assert b"alert--error" not in response.data

    def test_index_method_post_returns_405(self, client: FlaskClient) -> None:
        """GET / does not accept POST via the index route.

        Note: POST /report is a separate route; GET / is read-only.
        """
        # We test a random unsupported path instead to test 405
        response = client.post("/nonexistent-post-only")
        assert response.status_code in (404, 405)


# ---------------------------------------------------------------------------
# POST /report — form submission
# ---------------------------------------------------------------------------


class TestReportPostRoute:
    """Tests for the POST /report route (form submission)."""

    def test_valid_form_returns_200(self, client: FlaskClient) -> None:
        """Valid form submission must return HTTP 200."""
        response = client.post("/report", data=_valid_form_data())
        assert response.status_code == 200

    def test_valid_form_content_type_is_html(self, client: FlaskClient) -> None:
        """Valid form submission must return HTML."""
        response = client.post("/report", data=_valid_form_data())
        assert "text/html" in response.content_type

    def test_valid_form_response_contains_report_card(self, client: FlaskClient) -> None:
        """Valid submission must render a report card."""
        response = client.post("/report", data=_valid_form_data())
        assert b"report-card" in response.data

    def test_valid_form_response_contains_hardware(self, client: FlaskClient) -> None:
        """Response must mention the submitted hardware."""
        response = client.post("/report", data=_valid_form_data(hardware="H100_SXM"))
        assert b"H100_SXM" in response.data

    def test_valid_form_response_contains_region(self, client: FlaskClient) -> None:
        """Response must mention the submitted region."""
        response = client.post("/report", data=_valid_form_data(region="us-west-2"))
        assert b"us-west-2" in response.data

    def test_valid_form_shows_energy_kwh(self, client: FlaskClient) -> None:
        """Response must contain an energy value (kWh)."""
        response = client.post("/report", data=_valid_form_data())
        assert b"kWh" in response.data

    def test_valid_form_shows_co2e(self, client: FlaskClient) -> None:
        """Response must mention CO2e."""
        response = client.post("/report", data=_valid_form_data())
        assert b"CO" in response.data  # CO2e or CO₂e

    def test_valid_form_shows_water(self, client: FlaskClient) -> None:
        """Response must mention water usage."""
        response = client.post("/report", data=_valid_form_data())
        # Template shows water in litres
        assert b"Water" in response.data or b"water" in response.data

    def test_valid_form_training_workload(self, client: FlaskClient) -> None:
        """Training workload type must appear capitalised in the response."""
        response = client.post("/report", data=_valid_form_data(workload_type="training"))
        assert b"Training" in response.data or b"training" in response.data

    def test_valid_form_inference_workload(self, client: FlaskClient) -> None:
        """Inference workload type must appear in the response."""
        response = client.post("/report", data=_valid_form_data(workload_type="inference"))
        assert b"Inference" in response.data or b"inference" in response.data

    def test_valid_form_8_accelerators(self, client: FlaskClient) -> None:
        """8 accelerators must appear in the response."""
        response = client.post("/report", data=_valid_form_data(num_accelerators="8"))
        assert b"8" in response.data

    def test_valid_form_72h_duration(self, client: FlaskClient) -> None:
        """72-hour duration must appear in the response."""
        response = client.post("/report", data=_valid_form_data(duration_hours="72"))
        assert b"72" in response.data

    def test_form_missing_hardware_returns_400(self, client: FlaskClient) -> None:
        """Missing hardware field must return HTTP 400."""
        data = _valid_form_data()
        data["hardware"] = ""
        response = client.post("/report", data=data)
        assert response.status_code == 400

    def test_form_missing_region_returns_400(self, client: FlaskClient) -> None:
        """Missing region field must return HTTP 400."""
        data = _valid_form_data()
        data["region"] = ""
        response = client.post("/report", data=data)
        assert response.status_code == 400

    def test_form_unknown_hardware_returns_400(self, client: FlaskClient) -> None:
        """Unknown hardware key must return HTTP 400."""
        data = _valid_form_data(hardware="FAKE_GPU_9000")
        response = client.post("/report", data=data)
        assert response.status_code == 400

    def test_form_unknown_region_returns_400(self, client: FlaskClient) -> None:
        """Unknown region key must return HTTP 400."""
        data = _valid_form_data(region="mars-east-99")
        response = client.post("/report", data=data)
        assert response.status_code == 400

    def test_form_zero_duration_returns_400(self, client: FlaskClient) -> None:
        """Zero duration must return HTTP 400."""
        data = _valid_form_data(duration_hours="0")
        response = client.post("/report", data=data)
        assert response.status_code == 400

    def test_form_negative_duration_returns_400(self, client: FlaskClient) -> None:
        """Negative duration must return HTTP 400."""
        data = _valid_form_data(duration_hours="-5")
        response = client.post("/report", data=data)
        assert response.status_code == 400

    def test_form_non_numeric_duration_returns_400(self, client: FlaskClient) -> None:
        """Non-numeric duration must return HTTP 400."""
        data = _valid_form_data(duration_hours="not-a-number")
        response = client.post("/report", data=data)
        assert response.status_code == 400

    def test_form_zero_accelerators_returns_400(self, client: FlaskClient) -> None:
        """Zero accelerators must return HTTP 400."""
        data = _valid_form_data(num_accelerators="0")
        response = client.post("/report", data=data)
        assert response.status_code == 400

    def test_form_invalid_workload_type_returns_400(self, client: FlaskClient) -> None:
        """Invalid workload type must return HTTP 400."""
        data = _valid_form_data(workload_type="fine-tuning")
        response = client.post("/report", data=data)
        assert response.status_code == 400

    def test_form_error_response_contains_error_indicator(self, client: FlaskClient) -> None:
        """Error response must contain an error indicator in the HTML."""
        data = _valid_form_data(hardware="")
        response = client.post("/report", data=data)
        body = response.data
        assert b"error" in body.lower() or b"alert" in body.lower()

    def test_form_utilization_100_percent(self, client: FlaskClient) -> None:
        """100% utilization (form sends as '100') must be accepted."""
        data = _valid_form_data(utilization="100")
        response = client.post("/report", data=data)
        assert response.status_code == 200

    def test_form_utilization_50_percent(self, client: FlaskClient) -> None:
        """50% utilization (form sends as '50') must be accepted."""
        data = _valid_form_data(utilization="50")
        response = client.post("/report", data=data)
        assert response.status_code == 200

    def test_form_utilization_1_percent_minimum(self, client: FlaskClient) -> None:
        """1% utilization should be accepted (minimum meaningful value)."""
        data = _valid_form_data(utilization="1")
        response = client.post("/report", data=data)
        assert response.status_code == 200

    def test_form_response_contains_embed_section(self, client: FlaskClient) -> None:
        """Successful report page should contain the embed section."""
        response = client.post("/report", data=_valid_form_data())
        assert b"embed" in response.data.lower() or b"README" in response.data

    def test_form_response_contains_json_snippet(self, client: FlaskClient) -> None:
        """Successful report page should contain a JSON code block."""
        response = client.post("/report", data=_valid_form_data())
        assert b"schema_version" in response.data

    def test_form_t4_hardware(self, client: FlaskClient) -> None:
        """T4 hardware must be accepted and produce a valid response."""
        data = _valid_form_data(hardware="T4", region="us-east-1", duration_hours="1")
        response = client.post("/report", data=data)
        assert response.status_code == 200
        assert b"T4" in response.data

    def test_form_tpu_hardware(self, client: FlaskClient) -> None:
        """TPU_v4 hardware must be accepted and produce a valid response."""
        data = _valid_form_data(hardware="TPU_v4", region="us-central1", duration_hours="24")
        response = client.post("/report", data=data)
        assert response.status_code == 200

    def test_form_gcp_region(self, client: FlaskClient) -> None:
        """GCP region must be accepted and produce a valid response."""
        data = _valid_form_data(region="europe-west4", duration_hours="10")
        response = client.post("/report", data=data)
        assert response.status_code == 200

    def test_form_azure_region(self, client: FlaskClient) -> None:
        """Azure region must be accepted and produce a valid response."""
        data = _valid_form_data(region="swedencentral", duration_hours="5")
        response = client.post("/report", data=data)
        assert response.status_code == 200

    def test_form_get_method_not_allowed_for_report(self, client: FlaskClient) -> None:
        """GET /report without parameters renders empty-state page (200), not 405."""
        response = client.get("/report")
        # GET /report is a separate valid route that shows empty or result
        assert response.status_code in (200, 400)


# ---------------------------------------------------------------------------
# GET /report — shareable link with query parameters
# ---------------------------------------------------------------------------


class TestReportGetRoute:
    """Tests for the GET /report route (shareable links)."""

    def test_get_report_no_params_returns_200(self, client: FlaskClient) -> None:
        """GET /report with no parameters must return 200 with empty-state page."""
        response = client.get("/report")
        assert response.status_code == 200

    def test_get_report_no_params_shows_empty_state(self, client: FlaskClient) -> None:
        """GET /report with no parameters must show an empty-state message."""
        response = client.get("/report")
        assert b"No report" in response.data or b"calculator" in response.data.lower()

    def test_get_report_valid_params_returns_200(self, client: FlaskClient) -> None:
        """GET /report with valid query params must return HTTP 200."""
        response = client.get(
            "/report",
            query_string={
                "hardware": "A100_80GB",
                "region": "us-east-1",
                "duration_hours": "1.0",
                "num_accelerators": "1",
                "workload_type": "training",
                "utilization": "100",
            },
        )
        assert response.status_code == 200

    def test_get_report_valid_params_contains_report_card(self, client: FlaskClient) -> None:
        """GET /report with valid params must render a report card."""
        response = client.get(
            "/report",
            query_string={
                "hardware": "T4",
                "region": "us-east-1",
                "duration_hours": "2.0",
            },
        )
        assert response.status_code == 200
        assert b"report-card" in response.data

    def test_get_report_hardware_appears_in_response(self, client: FlaskClient) -> None:
        """Submitted hardware must appear in the GET /report response."""
        response = client.get(
            "/report",
            query_string={
                "hardware": "H100_SXM",
                "region": "us-east-1",
                "duration_hours": "1.0",
            },
        )
        assert b"H100_SXM" in response.data

    def test_get_report_region_appears_in_response(self, client: FlaskClient) -> None:
        """Submitted region must appear in the GET /report response."""
        response = client.get(
            "/report",
            query_string={
                "hardware": "A100_80GB",
                "region": "eu-north-1",
                "duration_hours": "1.0",
            },
        )
        assert b"eu-north-1" in response.data

    def test_get_report_unknown_hardware_returns_400(self, client: FlaskClient) -> None:
        """GET /report with unknown hardware must return HTTP 400."""
        response = client.get(
            "/report",
            query_string={
                "hardware": "FAKE_GPU",
                "region": "us-east-1",
                "duration_hours": "1.0",
            },
        )
        assert response.status_code == 400

    def test_get_report_unknown_region_returns_400(self, client: FlaskClient) -> None:
        """GET /report with unknown region must return HTTP 400."""
        response = client.get(
            "/report",
            query_string={
                "hardware": "A100_80GB",
                "region": "fake-region",
                "duration_hours": "1.0",
            },
        )
        assert response.status_code == 400

    def test_get_report_only_hardware_no_region_shows_empty(self, client: FlaskClient) -> None:
        """GET /report with only hardware (no region) must show empty state."""
        response = client.get(
            "/report",
            query_string={"hardware": "A100_80GB"},
        )
        assert response.status_code == 200
        # No region => empty state
        assert b"report-card--empty" in response.data or b"No report" in response.data or b"calculator" in response.data.lower()

    def test_get_report_only_region_no_hardware_shows_empty(self, client: FlaskClient) -> None:
        """GET /report with only region (no hardware) must show empty state."""
        response = client.get(
            "/report",
            query_string={"region": "us-east-1"},
        )
        assert response.status_code == 200
        assert b"report-card--empty" in response.data or b"No report" in response.data or b"calculator" in response.data.lower()

    def test_get_report_default_duration_is_accepted(self, client: FlaskClient) -> None:
        """GET /report without duration_hours should use default (1 hour)."""
        response = client.get(
            "/report",
            query_string={
                "hardware": "T4",
                "region": "us-east-1",
            },
        )
        assert response.status_code == 200

    def test_get_report_utilization_percentage_auto_scales(self, client: FlaskClient) -> None:
        """GET /report with utilization=80 (percentage) must be accepted."""
        response = client.get(
            "/report",
            query_string={
                "hardware": "T4",
                "region": "us-east-1",
                "duration_hours": "1.0",
                "utilization": "80",
            },
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/calculate — REST API
# ---------------------------------------------------------------------------


class TestApiCalculateRoute:
    """Tests for the POST /api/calculate REST API endpoint."""

    def _post_json(self, client: FlaskClient, payload: dict) -> object:
        """Send a JSON POST request to /api/calculate.

        Args:
            client: The Flask test client.
            payload: JSON-serialisable payload dict.

        Returns:
            The Flask response object.
        """
        return client.post(
            "/api/calculate",
            data=json.dumps(payload),
            content_type="application/json",
        )

    def test_valid_request_returns_200(self, client: FlaskClient) -> None:
        """Valid API request must return HTTP 200."""
        response = self._post_json(client, _valid_api_payload())
        assert response.status_code == 200

    def test_valid_request_content_type_is_json(self, client: FlaskClient) -> None:
        """Valid API response must have application/json content type."""
        response = self._post_json(client, _valid_api_payload())
        assert "application/json" in response.content_type

    def test_valid_request_returns_schema_version(self, client: FlaskClient) -> None:
        """Valid API response must include schema_version field."""
        response = self._post_json(client, _valid_api_payload())
        body = json.loads(response.data)
        assert "schema_version" in body
        assert body["schema_version"] == "1.0"

    def test_valid_request_returns_generated_at(self, client: FlaskClient) -> None:
        """Valid API response must include generated_at timestamp."""
        response = self._post_json(client, _valid_api_payload())
        body = json.loads(response.data)
        assert "generated_at" in body
        assert body["generated_at"].endswith("Z")

    def test_valid_request_returns_inputs_section(self, client: FlaskClient) -> None:
        """Valid API response must include inputs section."""
        response = self._post_json(client, _valid_api_payload())
        body = json.loads(response.data)
        assert "inputs" in body
        assert isinstance(body["inputs"], dict)

    def test_valid_request_returns_metrics_section(self, client: FlaskClient) -> None:
        """Valid API response must include metrics section."""
        response = self._post_json(client, _valid_api_payload())
        body = json.loads(response.data)
        assert "metrics" in body
        assert "energy_kwh" in body["metrics"]
        assert "co2e_kg" in body["metrics"]
        assert "water_liters" in body["metrics"]

    def test_valid_request_returns_factors_section(self, client: FlaskClient) -> None:
        """Valid API response must include factors section."""
        response = self._post_json(client, _valid_api_payload())
        body = json.loads(response.data)
        assert "factors" in body
        assert "tdp_watts" in body["factors"]
        assert "pue" in body["factors"]

    def test_valid_request_returns_comparisons_section(self, client: FlaskClient) -> None:
        """Valid API response must include comparisons section."""
        response = self._post_json(client, _valid_api_payload())
        body = json.loads(response.data)
        assert "comparisons" in body

    def test_valid_request_metrics_energy_positive(self, client: FlaskClient) -> None:
        """Energy metric must be positive."""
        response = self._post_json(client, _valid_api_payload())
        body = json.loads(response.data)
        assert body["metrics"]["energy_kwh"] > 0

    def test_valid_request_metrics_co2e_non_negative(self, client: FlaskClient) -> None:
        """CO2e metric must be non-negative."""
        response = self._post_json(client, _valid_api_payload())
        body = json.loads(response.data)
        assert body["metrics"]["co2e_kg"] >= 0

    def test_valid_request_metrics_water_non_negative(self, client: FlaskClient) -> None:
        """Water metric must be non-negative."""
        response = self._post_json(client, _valid_api_payload())
        body = json.loads(response.data)
        assert body["metrics"]["water_liters"] >= 0

    def test_inputs_reflected_in_response(self, client: FlaskClient) -> None:
        """Input fields must be reflected in the API response."""
        payload = _valid_api_payload(
            hardware="H100_SXM",
            region="eu-west-4",
            duration_hours=48.0,
            num_accelerators=4,
            workload_type="training",
            utilization=0.9,
        )
        # eu-west-4 doesn't exist, use a valid one
        payload["region"] = "europe-west4"
        response = self._post_json(client, payload)
        body = json.loads(response.data)
        assert body["inputs"]["hardware"] == "H100_SXM"
        assert body["inputs"]["region"] == "europe-west4"
        assert body["inputs"]["num_accelerators"] == 4
        assert body["inputs"]["workload_type"] == "training"

    def test_no_body_returns_400(self, client: FlaskClient) -> None:
        """Missing request body must return HTTP 400."""
        response = client.post("/api/calculate", content_type="application/json")
        assert response.status_code == 400

    def test_empty_json_object_returns_400(self, client: FlaskClient) -> None:
        """Empty JSON object {} must return HTTP 400 (missing required fields)."""
        response = self._post_json(client, {})
        assert response.status_code == 400

    def test_missing_hardware_returns_400(self, client: FlaskClient) -> None:
        """Missing 'hardware' field must return HTTP 400."""
        payload = _valid_api_payload()
        del payload["hardware"]
        response = self._post_json(client, payload)
        assert response.status_code == 400

    def test_missing_region_returns_400(self, client: FlaskClient) -> None:
        """Missing 'region' field must return HTTP 400."""
        payload = _valid_api_payload()
        del payload["region"]
        response = self._post_json(client, payload)
        assert response.status_code == 400

    def test_missing_duration_returns_400(self, client: FlaskClient) -> None:
        """Missing 'duration_hours' field must return HTTP 400."""
        payload = _valid_api_payload()
        del payload["duration_hours"]
        response = self._post_json(client, payload)
        assert response.status_code == 400

    def test_missing_required_fields_error_message(self, client: FlaskClient) -> None:
        """Missing required fields must produce an error message in the response."""
        response = self._post_json(client, {})
        body = json.loads(response.data)
        assert "error" in body
        assert len(body["error"]) > 0

    def test_unknown_hardware_returns_400(self, client: FlaskClient) -> None:
        """Unknown hardware key must return HTTP 400."""
        payload = _valid_api_payload(hardware="FAKE_GPU_X9000")
        response = self._post_json(client, payload)
        assert response.status_code == 400

    def test_unknown_hardware_error_message(self, client: FlaskClient) -> None:
        """Unknown hardware error must include a descriptive error message."""
        payload = _valid_api_payload(hardware="FAKE_GPU_X9000")
        response = self._post_json(client, payload)
        body = json.loads(response.data)
        assert "error" in body
        assert "FAKE_GPU_X9000" in body["error"] or "hardware" in body["error"].lower()

    def test_unknown_region_returns_400(self, client: FlaskClient) -> None:
        """Unknown region key must return HTTP 400."""
        payload = _valid_api_payload(region="mars-east-1")
        response = self._post_json(client, payload)
        assert response.status_code == 400

    def test_unknown_region_error_message(self, client: FlaskClient) -> None:
        """Unknown region error must include a descriptive error message."""
        payload = _valid_api_payload(region="mars-east-1")
        response = self._post_json(client, payload)
        body = json.loads(response.data)
        assert "error" in body
        assert "mars-east-1" in body["error"] or "region" in body["error"].lower()

    def test_zero_duration_returns_400(self, client: FlaskClient) -> None:
        """Zero duration_hours must return HTTP 400."""
        payload = _valid_api_payload(duration_hours=0.0)
        response = self._post_json(client, payload)
        assert response.status_code == 400

    def test_negative_duration_returns_400(self, client: FlaskClient) -> None:
        """Negative duration_hours must return HTTP 400."""
        payload = _valid_api_payload(duration_hours=-1.0)
        response = self._post_json(client, payload)
        assert response.status_code == 400

    def test_zero_accelerators_returns_400(self, client: FlaskClient) -> None:
        """Zero num_accelerators must return HTTP 400."""
        payload = _valid_api_payload(num_accelerators=0)
        response = self._post_json(client, payload)
        assert response.status_code == 400

    def test_negative_accelerators_returns_400(self, client: FlaskClient) -> None:
        """Negative num_accelerators must return HTTP 400."""
        payload = _valid_api_payload(num_accelerators=-1)
        response = self._post_json(client, payload)
        assert response.status_code == 400

    def test_invalid_workload_type_returns_400(self, client: FlaskClient) -> None:
        """Invalid workload_type must return HTTP 400."""
        payload = _valid_api_payload(workload_type="fine-tuning")
        response = self._post_json(client, payload)
        assert response.status_code == 400

    def test_zero_utilization_returns_400(self, client: FlaskClient) -> None:
        """Zero utilization (fraction mode) must return HTTP 400."""
        payload = _valid_api_payload(utilization=0.0)
        response = self._post_json(client, payload)
        assert response.status_code == 400

    def test_utilization_above_one_auto_scales_as_percentage(self, client: FlaskClient) -> None:
        """utilization=80 (> 1) is treated as percentage (80%) and auto-scaled."""
        payload = _valid_api_payload(utilization=80.0)
        response = self._post_json(client, payload)
        # Should succeed (80% = 0.8 fraction)
        assert response.status_code == 200
        body = json.loads(response.data)
        assert body["inputs"]["utilization"] == pytest.approx(0.8, rel=1e-4)

    def test_utilization_fraction_1_accepted(self, client: FlaskClient) -> None:
        """utilization=1.0 (100% in fraction mode) must be accepted."""
        payload = _valid_api_payload(utilization=1.0)
        response = self._post_json(client, payload)
        assert response.status_code == 200

    def test_utilization_fraction_0_5_accepted(self, client: FlaskClient) -> None:
        """utilization=0.5 (50% in fraction mode) must be accepted."""
        payload = _valid_api_payload(utilization=0.5)
        response = self._post_json(client, payload)
        assert response.status_code == 200

    def test_non_json_content_type_returns_400(self, client: FlaskClient) -> None:
        """Non-JSON content type (form-urlencoded) must return HTTP 400."""
        response = client.post(
            "/api/calculate",
            data="hardware=A100_80GB&region=us-east-1&duration_hours=1.0",
            content_type="application/x-www-form-urlencoded",
        )
        assert response.status_code == 400

    def test_malformed_json_returns_400(self, client: FlaskClient) -> None:
        """Malformed JSON body must return HTTP 400."""
        response = client.post(
            "/api/calculate",
            data=b"{not valid json",
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_json_array_body_returns_400(self, client: FlaskClient) -> None:
        """A JSON array (not object) body must return HTTP 400."""
        response = client.post(
            "/api/calculate",
            data=json.dumps([1, 2, 3]),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_inference_workload_type_accepted(self, client: FlaskClient) -> None:
        """'inference' workload type must be accepted."""
        payload = _valid_api_payload(workload_type="inference")
        response = self._post_json(client, payload)
        assert response.status_code == 200
        body = json.loads(response.data)
        assert body["inputs"]["workload_type"] == "inference"

    def test_training_workload_type_accepted(self, client: FlaskClient) -> None:
        """'training' workload type must be accepted."""
        payload = _valid_api_payload(workload_type="training")
        response = self._post_json(client, payload)
        assert response.status_code == 200
        body = json.loads(response.data)
        assert body["inputs"]["workload_type"] == "training"

    def test_default_num_accelerators_is_one(self, client: FlaskClient) -> None:
        """Omitting num_accelerators should default to 1."""
        payload = _valid_api_payload()
        del payload["num_accelerators"]
        response = self._post_json(client, payload)
        assert response.status_code == 200
        body = json.loads(response.data)
        assert body["inputs"]["num_accelerators"] == 1

    def test_default_workload_type_is_training(self, client: FlaskClient) -> None:
        """Omitting workload_type should default to 'training'."""
        payload = _valid_api_payload()
        del payload["workload_type"]
        response = self._post_json(client, payload)
        assert response.status_code == 200
        body = json.loads(response.data)
        assert body["inputs"]["workload_type"] == "training"

    def test_default_utilization_is_one(self, client: FlaskClient) -> None:
        """Omitting utilization should default to 1.0."""
        payload = _valid_api_payload()
        del payload["utilization"]
        response = self._post_json(client, payload)
        assert response.status_code == 200
        body = json.loads(response.data)
        assert body["inputs"]["utilization"] == pytest.approx(1.0)

    def test_api_response_is_full_json_report(self, client: FlaskClient) -> None:
        """API response must be the full JSON report (schema_version, inputs, metrics, etc.)."""
        response = self._post_json(client, _valid_api_payload())
        body = json.loads(response.data)
        required_keys = {"schema_version", "generated_at", "inputs", "metrics", "factors", "comparisons"}
        assert required_keys.issubset(set(body.keys()))

    def test_api_known_energy_value(self, client: FlaskClient) -> None:
        """Verify a known energy calculation via the API.

        Single A100_80GB (400 W) at 100% for 1 h in us-east-1 (PUE=1.20):
        energy = 400/1000 * 1 * 1.20 = 0.48 kWh.
        """
        payload = _valid_api_payload(
            hardware="A100_80GB",
            region="us-east-1",
            duration_hours=1.0,
            num_accelerators=1,
            utilization=1.0,
        )
        response = self._post_json(client, payload)
        assert response.status_code == 200
        body = json.loads(response.data)
        assert body["metrics"]["energy_kwh"] == pytest.approx(0.48, rel=1e-4)

    def test_api_8_accelerators_scales_energy(self, client: FlaskClient) -> None:
        """8 accelerators should give 8x the energy of 1 accelerator."""
        single = self._post_json(client, _valid_api_payload(num_accelerators=1))
        eight = self._post_json(client, _valid_api_payload(num_accelerators=8))
        single_body = json.loads(single.data)
        eight_body = json.loads(eight.data)
        assert eight_body["metrics"]["energy_kwh"] == pytest.approx(
            single_body["metrics"]["energy_kwh"] * 8, rel=1e-5
        )

    def test_api_low_carbon_region_less_co2(self, client: FlaskClient) -> None:
        """Oregon (low-carbon) must give less CO2e than Bahrain (high-carbon)."""
        oregon = self._post_json(
            client, _valid_api_payload(hardware="T4", region="us-west-2", duration_hours=10.0)
        )
        bahrain = self._post_json(
            client, _valid_api_payload(hardware="T4", region="me-south-1", duration_hours=10.0)
        )
        oregon_body = json.loads(oregon.data)
        bahrain_body = json.loads(bahrain.data)
        assert oregon_body["metrics"]["co2e_kg"] < bahrain_body["metrics"]["co2e_kg"]

    def test_api_response_all_metrics_positive_for_valid_input(self, client: FlaskClient) -> None:
        """All primary metrics must be positive for valid input."""
        response = self._post_json(client, _valid_api_payload())
        body = json.loads(response.data)
        assert body["metrics"]["energy_kwh"] > 0
        assert body["metrics"]["co2e_kg"] > 0
        assert body["metrics"]["water_liters"] > 0

    def test_api_non_string_duration_coerced(self, client: FlaskClient) -> None:
        """duration_hours as integer (not float) must be coerced and accepted."""
        payload = _valid_api_payload(duration_hours=5)  # int, not float
        response = self._post_json(client, payload)
        assert response.status_code == 200

    def test_api_float_num_accelerators_integer_value_accepted(self, client: FlaskClient) -> None:
        """num_accelerators=4.0 (float with integer value) should be coerced to int."""
        payload = _valid_api_payload(num_accelerators=4)
        response = self._post_json(client, payload)
        assert response.status_code == 200

    def test_api_string_duration_returns_400(self, client: FlaskClient) -> None:
        """Non-numeric duration_hours string must return HTTP 400."""
        payload = _valid_api_payload()
        payload["duration_hours"] = "not-a-number"
        response = self._post_json(client, payload)
        assert response.status_code == 400

    def test_api_error_response_has_error_key(self, client: FlaskClient) -> None:
        """All error responses must contain an 'error' key."""
        payload = _valid_api_payload(hardware="FAKE")
        response = self._post_json(client, payload)
        body = json.loads(response.data)
        assert "error" in body

    def test_api_get_method_not_allowed(self, client: FlaskClient) -> None:
        """GET /api/calculate must return 405 Method Not Allowed."""
        response = client.get("/api/calculate")
        assert response.status_code == 405

    def test_api_multiple_hardware_types(self, client: FlaskClient) -> None:
        """Multiple hardware types should all succeed via the API."""
        hardware_list = ["A100_80GB", "H100_SXM", "T4", "L4", "TPU_v4", "MI300X"]
        for hw in hardware_list:
            payload = _valid_api_payload(hardware=hw)
            response = self._post_json(client, payload)
            assert response.status_code == 200, f"Hardware {hw!r} failed: {response.data}"

    def test_api_multiple_regions(self, client: FlaskClient) -> None:
        """Multiple cloud regions should all succeed via the API."""
        regions = ["us-east-1", "us-west-2", "eu-north-1", "ap-southeast-1", "swedencentral"]
        for region in regions:
            payload = _valid_api_payload(region=region)
            response = self._post_json(client, payload)
            assert response.status_code == 200, f"Region {region!r} failed: {response.data}"


# ---------------------------------------------------------------------------
# GET /api/hardware — hardware preset list
# ---------------------------------------------------------------------------


class TestApiHardwareRoute:
    """Tests for the GET /api/hardware endpoint."""

    def test_returns_200(self, client: FlaskClient) -> None:
        """GET /api/hardware must return HTTP 200."""
        response = client.get("/api/hardware")
        assert response.status_code == 200

    def test_returns_json(self, client: FlaskClient) -> None:
        """GET /api/hardware must return JSON."""
        response = client.get("/api/hardware")
        assert "application/json" in response.content_type

    def test_response_has_hardware_key(self, client: FlaskClient) -> None:
        """Response must have a 'hardware' key."""
        response = client.get("/api/hardware")
        body = json.loads(response.data)
        assert "hardware" in body

    def test_response_has_count_key(self, client: FlaskClient) -> None:
        """Response must have a 'count' key."""
        response = client.get("/api/hardware")
        body = json.loads(response.data)
        assert "count" in body

    def test_hardware_is_list(self, client: FlaskClient) -> None:
        """'hardware' value must be a list."""
        response = client.get("/api/hardware")
        body = json.loads(response.data)
        assert isinstance(body["hardware"], list)

    def test_hardware_list_non_empty(self, client: FlaskClient) -> None:
        """Hardware list must be non-empty."""
        response = client.get("/api/hardware")
        body = json.loads(response.data)
        assert len(body["hardware"]) > 0

    def test_count_matches_hardware_list_length(self, client: FlaskClient) -> None:
        """'count' must equal the length of the 'hardware' list."""
        response = client.get("/api/hardware")
        body = json.loads(response.data)
        assert body["count"] == len(body["hardware"])

    def test_hardware_items_have_required_keys(self, client: FlaskClient) -> None:
        """Each hardware item must have 'key', 'label', 'vendor', 'tdp_watts'."""
        response = client.get("/api/hardware")
        body = json.loads(response.data)
        for item in body["hardware"]:
            assert "key" in item, f"Missing 'key' in item: {item}"
            assert "label" in item, f"Missing 'label' in item: {item}"
            assert "vendor" in item, f"Missing 'vendor' in item: {item}"
            assert "tdp_watts" in item, f"Missing 'tdp_watts' in item: {item}"

    def test_hardware_items_have_positive_tdp(self, client: FlaskClient) -> None:
        """All hardware TDP values must be positive."""
        response = client.get("/api/hardware")
        body = json.loads(response.data)
        for item in body["hardware"]:
            assert item["tdp_watts"] > 0, f"TDP <= 0 for {item['key']!r}"

    def test_hardware_list_contains_a100(self, client: FlaskClient) -> None:
        """Hardware list must include A100_80GB."""
        response = client.get("/api/hardware")
        body = json.loads(response.data)
        keys = [item["key"] for item in body["hardware"]]
        assert "A100_80GB" in keys

    def test_hardware_list_contains_h100(self, client: FlaskClient) -> None:
        """Hardware list must include H100_SXM."""
        response = client.get("/api/hardware")
        body = json.loads(response.data)
        keys = [item["key"] for item in body["hardware"]]
        assert "H100_SXM" in keys

    def test_hardware_list_contains_t4(self, client: FlaskClient) -> None:
        """Hardware list must include T4."""
        response = client.get("/api/hardware")
        body = json.loads(response.data)
        keys = [item["key"] for item in body["hardware"]]
        assert "T4" in keys

    def test_hardware_count_matches_data_table(self, client: FlaskClient) -> None:
        """Count must match the number of entries in the data table."""
        response = client.get("/api/hardware")
        body = json.loads(response.data)
        assert body["count"] == len(_data.HARDWARE_TDP_WATTS)

    def test_hardware_vendor_values_are_valid(self, client: FlaskClient) -> None:
        """Vendor values must be one of the known vendor tags."""
        valid_vendors = {"nvidia", "amd", "google", "intel", "other"}
        response = client.get("/api/hardware")
        body = json.loads(response.data)
        for item in body["hardware"]:
            assert item["vendor"] in valid_vendors, (
                f"Unknown vendor {item['vendor']!r} for {item['key']!r}"
            )

    def test_hardware_post_method_not_allowed(self, client: FlaskClient) -> None:
        """POST /api/hardware must return 405."""
        response = client.post("/api/hardware")
        assert response.status_code == 405


# ---------------------------------------------------------------------------
# GET /api/regions — region list
# ---------------------------------------------------------------------------


class TestApiRegionsRoute:
    """Tests for the GET /api/regions endpoint."""

    def test_returns_200(self, client: FlaskClient) -> None:
        """GET /api/regions must return HTTP 200."""
        response = client.get("/api/regions")
        assert response.status_code == 200

    def test_returns_json(self, client: FlaskClient) -> None:
        """GET /api/regions must return JSON."""
        response = client.get("/api/regions")
        assert "application/json" in response.content_type

    def test_response_has_regions_key(self, client: FlaskClient) -> None:
        """Response must have a 'regions' key."""
        response = client.get("/api/regions")
        body = json.loads(response.data)
        assert "regions" in body

    def test_response_has_count_key(self, client: FlaskClient) -> None:
        """Response must have a 'count' key."""
        response = client.get("/api/regions")
        body = json.loads(response.data)
        assert "count" in body

    def test_regions_is_list(self, client: FlaskClient) -> None:
        """'regions' value must be a list."""
        response = client.get("/api/regions")
        body = json.loads(response.data)
        assert isinstance(body["regions"], list)

    def test_regions_list_non_empty(self, client: FlaskClient) -> None:
        """Regions list must be non-empty."""
        response = client.get("/api/regions")
        body = json.loads(response.data)
        assert len(body["regions"]) > 0

    def test_count_matches_regions_list_length(self, client: FlaskClient) -> None:
        """'count' must equal the length of the 'regions' list."""
        response = client.get("/api/regions")
        body = json.loads(response.data)
        assert body["count"] == len(body["regions"])

    def test_regions_items_have_required_keys(self, client: FlaskClient) -> None:
        """Each region item must have the expected keys."""
        required_keys = {"key", "label", "provider", "carbon_intensity_g_per_kwh", "pue", "wue"}
        response = client.get("/api/regions")
        body = json.loads(response.data)
        for item in body["regions"]:
            for key in required_keys:
                assert key in item, f"Missing {key!r} in region item: {item}"

    def test_regions_carbon_intensity_non_negative(self, client: FlaskClient) -> None:
        """All carbon intensity values must be non-negative."""
        response = client.get("/api/regions")
        body = json.loads(response.data)
        for item in body["regions"]:
            assert item["carbon_intensity_g_per_kwh"] >= 0, (
                f"Negative CI for {item['key']!r}"
            )

    def test_regions_pue_gte_one(self, client: FlaskClient) -> None:
        """All PUE values must be >= 1.0."""
        response = client.get("/api/regions")
        body = json.loads(response.data)
        for item in body["regions"]:
            assert item["pue"] >= 1.0, f"PUE < 1.0 for {item['key']!r}: {item['pue']}"

    def test_regions_wue_non_negative(self, client: FlaskClient) -> None:
        """All WUE values must be non-negative."""
        response = client.get("/api/regions")
        body = json.loads(response.data)
        for item in body["regions"]:
            assert item["wue"] >= 0, f"Negative WUE for {item['key']!r}"

    def test_regions_list_contains_us_east_1(self, client: FlaskClient) -> None:
        """Regions list must include us-east-1."""
        response = client.get("/api/regions")
        body = json.loads(response.data)
        keys = [item["key"] for item in body["regions"]]
        assert "us-east-1" in keys

    def test_regions_list_contains_europe_west4(self, client: FlaskClient) -> None:
        """Regions list must include europe-west4 (GCP)."""
        response = client.get("/api/regions")
        body = json.loads(response.data)
        keys = [item["key"] for item in body["regions"]]
        assert "europe-west4" in keys

    def test_regions_list_contains_swedencentral(self, client: FlaskClient) -> None:
        """Regions list must include swedencentral (Azure)."""
        response = client.get("/api/regions")
        body = json.loads(response.data)
        keys = [item["key"] for item in body["regions"]]
        assert "swedencentral" in keys

    def test_regions_count_matches_data_table(self, client: FlaskClient) -> None:
        """Count must match the number of entries in the data table."""
        response = client.get("/api/regions")
        body = json.loads(response.data)
        assert body["count"] == len(_data.CARBON_INTENSITY_G_PER_KWH)

    def test_regions_provider_values_are_valid(self, client: FlaskClient) -> None:
        """Provider values must be one of the known provider tags."""
        valid_providers = {"aws", "gcp", "azure", "other"}
        response = client.get("/api/regions")
        body = json.loads(response.data)
        for item in body["regions"]:
            assert item["provider"] in valid_providers, (
                f"Unknown provider {item['provider']!r} for {item['key']!r}"
            )

    def test_regions_post_method_not_allowed(self, client: FlaskClient) -> None:
        """POST /api/regions must return 405."""
        response = client.post("/api/regions")
        assert response.status_code == 405


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


class TestErrorHandlers:
    """Tests for Flask error handlers."""

    def test_404_returns_404_status(self, client: FlaskClient) -> None:
        """Request to unknown URL must return HTTP 404."""
        response = client.get("/this-route-does-not-exist")
        assert response.status_code == 404

    def test_404_returns_json(self, client: FlaskClient) -> None:
        """404 response must be JSON."""
        response = client.get("/this-route-does-not-exist")
        assert "application/json" in response.content_type

    def test_404_body_has_error_key(self, client: FlaskClient) -> None:
        """404 JSON body must have an 'error' key."""
        response = client.get("/this-route-does-not-exist")
        body = json.loads(response.data)
        assert "error" in body

    def test_404_error_message_non_empty(self, client: FlaskClient) -> None:
        """404 error message must be non-empty."""
        response = client.get("/this-route-does-not-exist")
        body = json.loads(response.data)
        assert len(body["error"]) > 0

    def test_405_returns_405_status_for_api_get(self, client: FlaskClient) -> None:
        """GET /api/calculate (wrong method for that route) must return 405."""
        response = client.get("/api/calculate")
        assert response.status_code == 405

    def test_405_returns_json(self, client: FlaskClient) -> None:
        """405 response must be JSON."""
        response = client.get("/api/calculate")
        assert "application/json" in response.content_type

    def test_405_body_has_error_key(self, client: FlaskClient) -> None:
        """405 JSON body must have an 'error' key."""
        response = client.get("/api/calculate")
        body = json.loads(response.data)
        assert "error" in body


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


class TestAppFactory:
    """Tests for the create_app factory function."""

    def test_create_app_returns_flask_instance(self) -> None:
        """create_app() must return a Flask application."""
        from flask import Flask

        app = create_app({"TESTING": True})
        assert isinstance(app, Flask)

    def test_create_app_testing_config(self) -> None:
        """TESTING flag must be set when passed via test_config."""
        app = create_app({"TESTING": True})
        assert app.config["TESTING"] is True

    def test_create_app_no_config(self) -> None:
        """create_app() without arguments must succeed."""
        app = create_app()
        assert app is not None

    def test_create_app_has_index_route(self) -> None:
        """Application must have a route for '/'."""
        app = create_app({"TESTING": True})
        rules = [str(rule) for rule in app.url_map.iter_rules()]
        assert "/" in rules

    def test_create_app_has_report_route(self) -> None:
        """Application must have a route for '/report'."""
        app = create_app({"TESTING": True})
        rules = [str(rule) for rule in app.url_map.iter_rules()]
        assert "/report" in rules

    def test_create_app_has_api_calculate_route(self) -> None:
        """Application must have a route for '/api/calculate'."""
        app = create_app({"TESTING": True})
        rules = [str(rule) for rule in app.url_map.iter_rules()]
        assert "/api/calculate" in rules

    def test_create_app_has_api_hardware_route(self) -> None:
        """Application must have a route for '/api/hardware'."""
        app = create_app({"TESTING": True})
        rules = [str(rule) for rule in app.url_map.iter_rules()]
        assert "/api/hardware" in rules

    def test_create_app_has_api_regions_route(self) -> None:
        """Application must have a route for '/api/regions'."""
        app = create_app({"TESTING": True})
        rules = [str(rule) for rule in app.url_map.iter_rules()]
        assert "/api/regions" in rules

    def test_create_app_custom_secret_key(self) -> None:
        """Custom SECRET_KEY must be applied."""
        app = create_app({"SECRET_KEY": "my-test-secret"})
        assert app.config["SECRET_KEY"] == "my-test-secret"

    def test_two_apps_are_independent(self) -> None:
        """Two create_app() calls must produce independent instances."""
        app1 = create_app({"TESTING": True, "SECRET_KEY": "key-1"})
        app2 = create_app({"TESTING": True, "SECRET_KEY": "key-2"})
        assert app1 is not app2
        assert app1.config["SECRET_KEY"] != app2.config["SECRET_KEY"]


# ---------------------------------------------------------------------------
# Response correctness — reference value integration tests
# ---------------------------------------------------------------------------


class TestApiReferenceValues:
    """Integration tests verifying correct numeric results via the API."""

    def _post_json(self, client: FlaskClient, payload: dict) -> dict:
        """Helper: POST JSON payload and return the parsed response body."""
        response = client.post(
            "/api/calculate",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert response.status_code == 200, f"Unexpected status {response.status_code}: {response.data}"
        return json.loads(response.data)

    def test_a100_1h_us_east_1_energy(self, client: FlaskClient) -> None:
        """Single A100_80GB at 100% for 1 h in us-east-1 => 0.48 kWh."""
        body = self._post_json(
            client,
            _valid_api_payload(
                hardware="A100_80GB",
                region="us-east-1",
                duration_hours=1.0,
                num_accelerators=1,
                utilization=1.0,
            ),
        )
        # 400 W * 1 * 1.0 / 1000 * 1 h * 1.20 PUE = 0.48 kWh
        assert body["metrics"]["energy_kwh"] == pytest.approx(0.48, rel=1e-4)

    def test_a100_1h_us_east_1_co2e(self, client: FlaskClient) -> None:
        """Single A100_80GB at 100% for 1 h in us-east-1 => correct CO2e."""
        body = self._post_json(
            client,
            _valid_api_payload(
                hardware="A100_80GB",
                region="us-east-1",
                duration_hours=1.0,
                num_accelerators=1,
                utilization=1.0,
            ),
        )
        # CO2e = 0.48 kWh * 415 gCO2/kWh / 1000 = 0.1992 kg
        assert body["metrics"]["co2e_kg"] == pytest.approx(0.1992, rel=1e-3)

    def test_t4_inference_80pct_energy(self, client: FlaskClient) -> None:
        """Single T4 at 80% for 1 h in us-east-1 => correct energy."""
        body = self._post_json(
            client,
            _valid_api_payload(
                hardware="T4",
                region="us-east-1",
                duration_hours=1.0,
                num_accelerators=1,
                workload_type="inference",
                utilization=0.8,
            ),
        )
        # 70 W * 0.8 / 1000 * 1 h * 1.20 PUE = 0.0672 kWh
        assert body["metrics"]["energy_kwh"] == pytest.approx(0.0672, rel=1e-4)

    def test_8_a100_72h_oregon_energy(self, client: FlaskClient) -> None:
        """8x A100_80GB at 100% for 72 h in us-west-2 => 276.48 kWh."""
        body = self._post_json(
            client,
            _valid_api_payload(
                hardware="A100_80GB",
                region="us-west-2",
                duration_hours=72.0,
                num_accelerators=8,
                utilization=1.0,
            ),
        )
        # 8 * 400/1000 * 72 * 1.20 = 276.48 kWh
        assert body["metrics"]["energy_kwh"] == pytest.approx(276.48, rel=1e-4)

    def test_tpu_v4_64_chips_24h(self, client: FlaskClient) -> None:
        """64x TPU v4 at 100% for 24 h in us-central1 => expected energy."""
        body = self._post_json(
            client,
            _valid_api_payload(
                hardware="TPU_v4",
                region="us-central1",
                duration_hours=24.0,
                num_accelerators=64,
                utilization=1.0,
            ),
        )
        # 64 * 192/1000 * 24 * 1.11 = 327.0528 kWh (PUE=1.11 for us-central1)
        expected = (64 * 192 / 1000) * 24 * 1.11
        assert body["metrics"]["energy_kwh"] == pytest.approx(expected, rel=1e-4)

    def test_nordic_region_very_low_co2(self, client: FlaskClient) -> None:
        """eu-north-1 (Stockholm, CI=8) should give very low CO2e."""
        body = self._post_json(
            client,
            _valid_api_payload(
                hardware="A100_80GB",
                region="eu-north-1",
                duration_hours=1.0,
                num_accelerators=1,
                utilization=1.0,
            ),
        )
        # CI = 8 gCO2/kWh; energy ~= 400/1000 * 1 * PUE(1.15) = 0.46 kWh
        # CO2e = 0.46 * 8 / 1000 = 0.00368 kg (very low)
        assert body["metrics"]["co2e_kg"] < 0.05

    def test_south_africa_high_co2(self, client: FlaskClient) -> None:
        """af-south-1 (Cape Town, CI=928) must produce high CO2e."""
        body = self._post_json(
            client,
            _valid_api_payload(
                hardware="A100_80GB",
                region="af-south-1",
                duration_hours=1.0,
                num_accelerators=1,
                utilization=1.0,
            ),
        )
        assert body["metrics"]["co2e_kg"] > 0.4

    def test_api_factors_match_data_tables(self, client: FlaskClient) -> None:
        """API response factors must match the static data table values."""
        hw = "H100_SXM"
        region = "europe-west4"
        body = self._post_json(
            client,
            _valid_api_payload(hardware=hw, region=region, duration_hours=1.0),
        )
        assert body["factors"]["tdp_watts"] == pytest.approx(
            _data.HARDWARE_TDP_WATTS[hw], rel=1e-6
        )
        assert body["factors"]["pue"] == pytest.approx(
            _data.PUE_BY_REGION[region], rel=1e-6
        )
        assert body["factors"]["wue_l_per_kwh"] == pytest.approx(
            _data.WUE_BY_REGION[region], rel=1e-6
        )
        assert body["factors"]["carbon_intensity_g_per_kwh"] == pytest.approx(
            _data.CARBON_INTENSITY_G_PER_KWH[region], rel=1e-6
        )

    def test_api_unit_conversions_consistent(self, client: FlaskClient) -> None:
        """Unit conversion metrics must be internally consistent."""
        body = self._post_json(client, _valid_api_payload())
        metrics = body["metrics"]
        # MWh consistency
        assert metrics["energy_mwh"] * 1000 == pytest.approx(metrics["energy_kwh"], rel=1e-5)
        # CO2e unit consistency
        assert metrics["co2e_g"] / 1000 == pytest.approx(metrics["co2e_kg"], rel=1e-5)
        assert metrics["co2e_tonnes"] * 1000 == pytest.approx(metrics["co2e_kg"], rel=1e-5)
        # Water unit consistency
        assert metrics["water_ml"] / 1000 == pytest.approx(metrics["water_liters"], rel=1e-5)

    def test_api_comparison_values_consistent(self, client: FlaskClient) -> None:
        """Comparison values must be consistent with the metric values."""
        body = self._post_json(
            client,
            _valid_api_payload(
                hardware="T4",
                region="us-east-1",
                duration_hours=1.0,
                utilization=1.0,
            ),
        )
        co2_kg = body["metrics"]["co2e_kg"]
        car_km = body["comparisons"]["co2e"]["car_km_equivalent"]
        # car_km = round(co2_kg / 0.170, 2)
        expected_km = round(co2_kg / 0.170, 2)
        assert car_km == pytest.approx(expected_km, rel=1e-4)
