"""Flask application factory and route definitions for dc_impact.

This module contains the ``create_app`` factory function and all route
definitions for the calculator UI, the REST API endpoint (/api/calculate),
and the report card rendering route.

Routes
------
* ``GET /``               — Calculator form UI.
* ``POST /report``        — Form submission; renders the HTML report card.
* ``GET /report``         — Renders report card from query-string params.
* ``POST /api/calculate`` — REST API; accepts JSON, returns JSON report.
* ``GET /api/hardware``   — Lists available hardware presets.
* ``GET /api/regions``    — Lists available cloud regions.
"""

from __future__ import annotations

import logging
from typing import Any

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
)

from dc_impact.calculator import (
    CalculatorError,
    calculate_impact,
    get_available_hardware,
    get_available_regions,
)
from dc_impact.data import HARDWARE_METADATA, REGION_METADATA
from dc_impact.report import generate_html_report, generate_json_report

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        test_config: Optional mapping of configuration overrides used during
            testing (e.g. ``{"TESTING": True, "SECRET_KEY": "test"}``).

    Returns:
        A fully configured :class:`flask.Flask` application instance with
        all routes registered.
    """
    app = Flask(__name__, instance_relative_config=True)

    # Default configuration
    app.config.from_mapping(
        SECRET_KEY="dev-secret-change-in-production",
        DEBUG=False,
        JSON_SORT_KEYS=False,
    )

    if test_config is not None:
        app.config.from_mapping(test_config)

    _register_routes(app)

    return app


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def _register_routes(app: Flask) -> None:  # noqa: C901  (complexity is acceptable here)
    """Register all URL routes on the given Flask application.

    Args:
        app: The Flask application instance to attach routes to.
    """

    # ------------------------------------------------------------------
    # Calculator UI — index page
    # ------------------------------------------------------------------

    @app.get("/")
    def index() -> str:
        """Render the main calculator form UI.

        Passes the hardware and region metadata to the template so that
        the ``<select>`` dropdowns can be populated dynamically.

        Returns:
            Rendered HTML string for the calculator index page.
        """
        # Group hardware by vendor for the optgroup layout
        hardware_by_vendor: dict[str, list[dict[str, str]]] = {}
        for key in get_available_hardware():
            meta = HARDWARE_METADATA.get(key, {"label": key, "vendor": "other"})
            vendor = meta["vendor"]
            hardware_by_vendor.setdefault(vendor, [])
            hardware_by_vendor[vendor].append({"key": key, "label": meta["label"]})

        # Group regions by provider for the optgroup layout
        regions_by_provider: dict[str, list[dict[str, str]]] = {}
        for key in get_available_regions():
            meta = REGION_METADATA.get(key, {"label": key, "provider": "other"})
            provider = meta["provider"]
            regions_by_provider.setdefault(provider, [])
            regions_by_provider[provider].append({"key": key, "label": meta["label"]})

        return render_template(
            "index.html",
            hardware_by_vendor=hardware_by_vendor,
            regions_by_provider=regions_by_provider,
        )

    # ------------------------------------------------------------------
    # Report card — form POST submission
    # ------------------------------------------------------------------

    @app.post("/report")
    def report_post() -> tuple[str, int] | str:
        """Handle calculator form submission and render the report card.

        Reads form fields from the POST body, runs the calculation, and
        renders the ``report_card.html`` template inside a full HTML page.
        On validation/calculation errors, re-renders the index form with
        an error message.

        Returns:
            Rendered HTML for the report card page, or a redirect to the
            index with an error message on failure.
        """
        form = request.form

        # --- Parse form fields ---
        hardware = form.get("hardware", "").strip()
        region = form.get("region", "").strip()
        workload_type = form.get("workload_type", "training").strip()

        parse_errors: list[str] = []

        try:
            duration_hours = float(form.get("duration_hours", "0"))
        except (ValueError, TypeError):
            duration_hours = 0.0
            parse_errors.append("Duration must be a number.")

        try:
            num_accelerators = int(form.get("num_accelerators", "1"))
        except (ValueError, TypeError):
            num_accelerators = 1
            parse_errors.append("Number of accelerators must be an integer.")

        try:
            utilization_pct = float(form.get("utilization", "100"))
            utilization = utilization_pct / 100.0
        except (ValueError, TypeError):
            utilization = 1.0
            parse_errors.append("Utilization must be a number between 1 and 100.")

        if parse_errors:
            return _render_index_with_error(
                "; ".join(parse_errors),
                form_values=dict(form),
            ), 400

        try:
            result = calculate_impact(
                hardware=hardware,
                region=region,
                duration_hours=duration_hours,
                num_accelerators=num_accelerators,
                workload_type=workload_type,
                utilization=utilization,
            )
        except CalculatorError as exc:
            logger.debug("Calculation error from form: %s", exc)
            return _render_index_with_error(str(exc), form_values=dict(form)), 400
        except Exception as exc:
            logger.exception("Unexpected error during form calculation: %s", exc)
            return _render_index_with_error(
                "An unexpected error occurred. Please try again.",
                form_values=dict(form),
            ), 500

        report_data = generate_json_report(result)
        html_card = generate_html_report(result)

        return render_template(
            "report_page.html",
            result=result,
            report=report_data,
            html_card=html_card,
        )

    # ------------------------------------------------------------------
    # Report card — GET with query-string parameters
    # ------------------------------------------------------------------

    @app.get("/report")
    def report_get() -> tuple[str, int] | str:
        """Render a shareable report card from query-string parameters.

        Accepts the same parameters as the POST form but as URL query
        parameters. Returns the empty report card template when no
        parameters are provided.

        Query parameters:
            hardware, region, duration_hours, num_accelerators,
            workload_type, utilization (percentage, 1–100).

        Returns:
            Rendered HTML for the report card page.
        """
        args = request.args

        hardware = args.get("hardware", "").strip()
        region = args.get("region", "").strip()

        # If no parameters provided, render the empty state
        if not hardware or not region:
            return render_template(
                "report_page.html",
                result=None,
                report=None,
                html_card=None,
            )

        workload_type = args.get("workload_type", "training").strip()

        try:
            duration_hours = float(args.get("duration_hours", "1"))
            num_accelerators = int(args.get("num_accelerators", "1"))
            utilization_pct = float(args.get("utilization", "100"))
            utilization = max(0.01, min(1.0, utilization_pct / 100.0))
        except (ValueError, TypeError) as exc:
            return jsonify({"error": f"Invalid query parameter: {exc}"}), 400

        try:
            result = calculate_impact(
                hardware=hardware,
                region=region,
                duration_hours=duration_hours,
                num_accelerators=num_accelerators,
                workload_type=workload_type,
                utilization=utilization,
            )
        except CalculatorError as exc:
            logger.debug("Calculation error from GET params: %s", exc)
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            logger.exception("Unexpected error during GET report: %s", exc)
            return jsonify({"error": "Internal server error."}), 500

        report_data = generate_json_report(result)
        html_card = generate_html_report(result)

        return render_template(
            "report_page.html",
            result=result,
            report=report_data,
            html_card=html_card,
        )

    # ------------------------------------------------------------------
    # REST API — /api/calculate
    # ------------------------------------------------------------------

    @app.post("/api/calculate")
    def api_calculate() -> tuple[Response, int]:
        """REST API endpoint that accepts JSON workload parameters and returns
        a structured impact report.

        Expected JSON body fields:
            - ``hardware`` (str): Hardware preset key, e.g. ``"A100_80GB"``.
            - ``region`` (str): Cloud region key, e.g. ``"us-east-1"``.
            - ``duration_hours`` (float): Wall-clock duration in hours.
            - ``num_accelerators`` (int, optional): Number of accelerator
              units (default: ``1``).
            - ``workload_type`` (str, optional): ``"training"`` or
              ``"inference"`` (default: ``"training"``).  
            - ``utilization`` (float, optional): Fractional GPU utilization
              between 0 and 1 (default: ``1.0``). May also be supplied as
              a percentage (>1 and <=100) and will be auto-scaled.

        Returns:
            ``200`` with the full JSON report on success.
            ``400`` with ``{"error": "..."}`` on invalid input.
            ``500`` with ``{"error": "..."}`` on unexpected errors.
        """
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        # --- Required fields ---
        hardware = payload.get("hardware")
        region = payload.get("region")
        duration_hours_raw = payload.get("duration_hours")

        missing = []
        if not hardware:
            missing.append("hardware")
        if not region:
            missing.append("region")
        if duration_hours_raw is None:
            missing.append("duration_hours")
        if missing:
            return (
                jsonify(
                    {
                        "error": (
                            f"Missing required fields: {', '.join(missing)}."
                        )
                    }
                ),
                400,
            )

        # --- Parse and coerce types ---
        try:
            duration_hours = float(duration_hours_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return (
                jsonify({"error": "'duration_hours' must be a positive number."}),
                400,
            )

        num_accelerators_raw = payload.get("num_accelerators", 1)
        try:
            num_accelerators = int(num_accelerators_raw)
            if isinstance(num_accelerators_raw, float) and not num_accelerators_raw.is_integer():
                raise ValueError("num_accelerators must be an integer")
        except (TypeError, ValueError):
            return (
                jsonify({"error": "'num_accelerators' must be a positive integer."}),
                400,
            )

        workload_type = payload.get("workload_type", "training")

        utilization_raw = payload.get("utilization", 1.0)
        try:
            utilization = float(utilization_raw)  # type: ignore[arg-type]
            # Auto-detect percentage vs fraction
            if utilization > 1.0:
                # Treat as percentage (e.g. 80 -> 0.80)
                utilization = utilization / 100.0
        except (TypeError, ValueError):
            return (
                jsonify({"error": "'utilization' must be a number in (0, 1] or a percentage (0–100)."}),
                400,
            )

        # --- Calculate ---
        try:
            result = calculate_impact(
                hardware=str(hardware),
                region=str(region),
                duration_hours=duration_hours,
                num_accelerators=num_accelerators,
                workload_type=str(workload_type),
                utilization=utilization,
            )
        except CalculatorError as exc:
            logger.debug("CalculatorError in API: %s", exc)
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            logger.exception("Unexpected error in API calculate: %s", exc)
            return jsonify({"error": "Internal server error."}), 500

        report = generate_json_report(result)
        return jsonify(report), 200

    # ------------------------------------------------------------------
    # REST API — convenience list endpoints
    # ------------------------------------------------------------------

    @app.get("/api/hardware")
    def api_hardware() -> tuple[Response, int]:
        """Return a list of all supported hardware presets with metadata.

        Returns:
            JSON array of hardware preset objects with keys:
            ``key``, ``label``, ``vendor``, ``tdp_watts``.
        """
        from dc_impact.data import HARDWARE_TDP_WATTS

        items = []
        for key in get_available_hardware():
            meta = HARDWARE_METADATA.get(key, {"label": key, "vendor": "other"})
            items.append(
                {
                    "key": key,
                    "label": meta.get("label", key),
                    "vendor": meta.get("vendor", "other"),
                    "tdp_watts": HARDWARE_TDP_WATTS[key],
                }
            )
        return jsonify({"hardware": items, "count": len(items)}), 200

    @app.get("/api/regions")
    def api_regions() -> tuple[Response, int]:
        """Return a list of all supported cloud regions with metadata.

        Returns:
            JSON array of region objects with keys:
            ``key``, ``label``, ``provider``, ``carbon_intensity_g_per_kwh``,
            ``pue``, ``wue``.
        """
        from dc_impact.data import (
            CARBON_INTENSITY_G_PER_KWH,
            DEFAULT_PUE,
            DEFAULT_WUE,
            PUE_BY_REGION,
            WUE_BY_REGION,
        )

        items = []
        for key in get_available_regions():
            meta = REGION_METADATA.get(key, {"label": key, "provider": "other"})
            items.append(
                {
                    "key": key,
                    "label": meta.get("label", key),
                    "provider": meta.get("provider", "other"),
                    "carbon_intensity_g_per_kwh": CARBON_INTENSITY_G_PER_KWH[key],
                    "pue": PUE_BY_REGION.get(key, DEFAULT_PUE),
                    "wue": WUE_BY_REGION.get(key, DEFAULT_WUE),
                }
            )
        return jsonify({"regions": items, "count": len(items)}), 200

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------

    @app.errorhandler(404)
    def not_found(exc: Exception) -> tuple[Response, int]:
        """Handle 404 Not Found errors with a JSON response.

        Args:
            exc: The exception that triggered the error handler.

        Returns:
            A tuple of (JSON response, 404 status code).
        """
        return jsonify({"error": "Resource not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(exc: Exception) -> tuple[Response, int]:
        """Handle 405 Method Not Allowed errors with a JSON response.

        Args:
            exc: The exception that triggered the error handler.

        Returns:
            A tuple of (JSON response, 405 status code).
        """
        return jsonify({"error": "Method not allowed."}), 405

    @app.errorhandler(500)
    def internal_error(exc: Exception) -> tuple[Response, int]:
        """Handle 500 Internal Server Error with a JSON response.

        Args:
            exc: The exception that triggered the error handler.

        Returns:
            A tuple of (JSON response, 500 status code).
        """
        logger.exception("Unhandled internal error: %s", exc)
        return jsonify({"error": "Internal server error."}), 500


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _render_index_with_error(
    error_message: str,
    form_values: dict[str, Any] | None = None,
) -> str:
    """Render the index template with an inline error message.

    Args:
        error_message: Human-readable error text to display to the user.
        form_values: Optional dict of previously submitted form values so
            that the form can be pre-populated after an error.

    Returns:
        Rendered HTML string for the index page with error state.
    """
    # Rebuild the grouped hardware / region lists the same way as index()
    hardware_by_vendor: dict[str, list[dict[str, str]]] = {}
    for key in get_available_hardware():
        meta = HARDWARE_METADATA.get(key, {"label": key, "vendor": "other"})
        vendor = meta["vendor"]
        hardware_by_vendor.setdefault(vendor, [])
        hardware_by_vendor[vendor].append({"key": key, "label": meta["label"]})

    regions_by_provider: dict[str, list[dict[str, str]]] = {}
    for key in get_available_regions():
        meta = REGION_METADATA.get(key, {"label": key, "provider": "other"})
        provider = meta["provider"]
        regions_by_provider.setdefault(provider, [])
        regions_by_provider[provider].append({"key": key, "label": meta["label"]})

    return render_template(
        "index.html",
        error=error_message,
        form_values=form_values or {},
        hardware_by_vendor=hardware_by_vendor,
        regions_by_provider=regions_by_provider,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the ``dc_impact`` command-line script.

    Starts the Flask development server on ``0.0.0.0:5000``.
    """
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
