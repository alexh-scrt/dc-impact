"""Flask application factory and route definitions for dc_impact.

This module contains the ``create_app`` factory function and all route
definitions for the calculator UI, the REST API endpoint (/api/calculate),
and the report card rendering route.

Routes
------
* ``GET /``               — Calculator form UI.
* ``POST /report``        — Form submission; renders the HTML report card page.
* ``GET /report``         — Render report card from query-string parameters
                            (shareable link).
* ``POST /api/calculate`` — REST API; accepts JSON, returns a full JSON report.
* ``GET /api/hardware``   — List all supported hardware presets with metadata.
* ``GET /api/regions``    — List all supported cloud regions with metadata.

Error handling
--------------
All routes catch :class:`~dc_impact.calculator.CalculatorError` and return
HTTP 400 with a descriptive message.  Unexpected exceptions are logged and
return HTTP 500.  JSON-only API routes always return JSON error bodies;
UI routes re-render the form with an inline error banner.
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
from dc_impact.data import (
    CARBON_INTENSITY_G_PER_KWH,
    DEFAULT_PUE,
    DEFAULT_WUE,
    HARDWARE_METADATA,
    HARDWARE_TDP_WATTS,
    PUE_BY_REGION,
    REGION_METADATA,
    WUE_BY_REGION,
)
from dc_impact.report import generate_html_report, generate_json_report

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    """Create and configure the Flask application.

    The factory pattern allows multiple independent instances to be created
    (e.g. for testing) without shared global state.

    Args:
        test_config: Optional mapping of configuration overrides applied
            after the defaults.  Typically used in tests::

                app = create_app({"TESTING": True, "SECRET_KEY": "test"})

    Returns:
        A fully configured :class:`flask.Flask` application instance with
        all routes and error handlers registered.
    """
    app = Flask(__name__, instance_relative_config=True)

    # Default configuration
    app.config.from_mapping(
        SECRET_KEY="dev-secret-change-in-production",
        DEBUG=False,
        JSON_SORT_KEYS=False,
        # Disable strict trailing-slash matching for a nicer UX
        STRICT_SLASHES=False,
    )

    if test_config is not None:
        app.config.from_mapping(test_config)

    _register_routes(app)
    _register_error_handlers(app)

    return app


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def _register_routes(app: Flask) -> None:
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

        Passes hardware and region metadata to the template so that
        ``<optgroup>`` dropdowns can be populated dynamically from the
        bundled lookup tables.

        Returns:
            Rendered HTML string for the calculator index page.
        """
        hardware_by_vendor = _group_hardware_by_vendor()
        regions_by_provider = _group_regions_by_provider()

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

        Reads form fields from the POST body, validates and runs the
        calculation, then renders ``report_page.html`` with the result.
        On parse or calculation errors the index form is re-rendered with
        an inline error banner and HTTP 400.

        Form fields
        -----------
        * ``hardware``        (str)   — hardware preset key
        * ``region``          (str)   — cloud region key
        * ``duration_hours``  (float) — workload wall-clock duration in hours
        * ``num_accelerators``(int)   — number of accelerator devices (>=1)
        * ``workload_type``   (str)   — ``"training"`` or ``"inference"``
        * ``utilization``     (float) — GPU utilisation percentage (1–100)

        Returns:
            Rendered HTML for the report card page on success, or a
            re-rendered index page with an error banner on failure.
        """
        form = request.form

        hardware = form.get("hardware", "").strip()
        region = form.get("region", "").strip()
        workload_type = form.get("workload_type", "training").strip()

        parse_errors: list[str] = []

        # --- duration_hours ---
        try:
            duration_hours_raw = form.get("duration_hours", "0")
            duration_hours = float(duration_hours_raw)  # type: ignore[arg-type]
            if duration_hours <= 0:
                parse_errors.append("Duration must be greater than zero.")
        except (ValueError, TypeError):
            duration_hours = 0.0
            parse_errors.append("Duration must be a valid positive number.")

        # --- num_accelerators ---
        try:
            num_accelerators = int(form.get("num_accelerators", "1"))
            if num_accelerators < 1:
                parse_errors.append("Number of accelerators must be at least 1.")
        except (ValueError, TypeError):
            num_accelerators = 1
            parse_errors.append("Number of accelerators must be a positive integer.")

        # --- utilization (form sends as percentage 1–100) ---
        try:
            utilization_pct = float(form.get("utilization", "100"))
            if not (1.0 <= utilization_pct <= 100.0):
                parse_errors.append("Utilization must be between 1 % and 100 %.")
            utilization = max(0.01, min(1.0, utilization_pct / 100.0))
        except (ValueError, TypeError):
            utilization = 1.0
            parse_errors.append("Utilization must be a number between 1 and 100.")

        if parse_errors:
            return (
                _render_index_with_error(
                    "; ".join(parse_errors),
                    form_values=dict(form),
                ),
                400,
            )

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
            return (
                _render_index_with_error(str(exc), form_values=dict(form)),
                400,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error during form calculation: %s", exc)
            return (
                _render_index_with_error(
                    "An unexpected server error occurred. Please try again.",
                    form_values=dict(form),
                ),
                500,
            )

        report_data = generate_json_report(result)
        html_card = generate_html_report(result)

        return render_template(
            "report_page.html",
            result=result,
            report=report_data,
            html_card=html_card,
        )

    # ------------------------------------------------------------------
    # Report card — GET with query-string parameters (shareable link)
    # ------------------------------------------------------------------

    @app.get("/report")
    def report_get() -> tuple[str | Response, int] | str:
        """Render a shareable report card from URL query-string parameters.

        Accepts the same parameters as the POST form submission but as URL
        query parameters.  Returns the empty-state report page when neither
        ``hardware`` nor ``region`` are provided.

        Query parameters
        ----------------
        Same as the POST form fields above.  ``utilization`` is interpreted
        as a percentage (1–100); it is clamped to ``[0.01, 1.0]`` before
        being passed to the calculator.

        Returns:
            Rendered HTML for the report card page (200), an empty-state
            page (200 when no params), or a JSON error (400/500).
        """
        args = request.args

        hardware = args.get("hardware", "").strip()
        region = args.get("region", "").strip()

        # If neither key parameter is provided, render empty state
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
            # Accept both percentage (>1) and fraction (<=1) for flexibility
            if utilization_pct > 1.0:
                utilization = max(0.01, min(1.0, utilization_pct / 100.0))
            else:
                utilization = max(0.01, min(1.0, utilization_pct))
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
        except Exception as exc:  # noqa: BLE001
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
    # REST API — POST /api/calculate
    # ------------------------------------------------------------------

    @app.post("/api/calculate")
    def api_calculate() -> tuple[Response, int]:
        """REST API endpoint: accepts a JSON workload description and returns
        a structured environmental impact report.

        Request body (JSON object)
        --------------------------
        Required fields:

        * ``hardware``       (str)   — hardware preset key, e.g. ``"A100_80GB"``.
        * ``region``         (str)   — cloud region key, e.g. ``"us-east-1"``.
        * ``duration_hours`` (float) — wall-clock duration in hours (> 0).

        Optional fields:

        * ``num_accelerators`` (int,   default 1)         — number of devices.
        * ``workload_type``    (str,   default "training")— ``"training"`` or
          ``"inference"``.
        * ``utilization``      (float, default 1.0)       — fractional GPU
          utilisation in ``(0, 1]``, **or** a percentage in ``(1, 100]``
          which is auto-scaled to a fraction.

        Response
        --------
        * **200** — full JSON impact report (schema_version, generated_at,
          inputs, metrics, factors, comparisons).
        * **400** — ``{"error": "..."}`` for invalid or missing parameters.
        * **500** — ``{"error": "..."}`` for unexpected server errors.

        Returns:
            Tuple of (JSON Response, HTTP status code).
        """
        # --- Parse body ---
        payload = request.get_json(silent=True, force=False)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        # --- Required fields ---
        hardware = payload.get("hardware")
        region = payload.get("region")
        duration_hours_raw = payload.get("duration_hours")

        missing: list[str] = []
        if not hardware:
            missing.append("hardware")
        if not region:
            missing.append("region")
        if duration_hours_raw is None:
            missing.append("duration_hours")

        if missing:
            return (
                jsonify(
                    {"error": f"Missing required field(s): {', '.join(missing)}."}
                ),
                400,
            )

        # --- duration_hours coercion ---
        try:
            duration_hours = float(duration_hours_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return (
                jsonify({"error": "'duration_hours' must be a positive number."}),
                400,
            )

        # --- num_accelerators coercion ---
        num_accelerators_raw = payload.get("num_accelerators", 1)
        try:
            if isinstance(num_accelerators_raw, float):
                if not num_accelerators_raw.is_integer():
                    return (
                        jsonify(
                            {"error": "'num_accelerators' must be a positive integer."}
                        ),
                        400,
                    )
                num_accelerators = int(num_accelerators_raw)
            else:
                num_accelerators = int(num_accelerators_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return (
                jsonify({"error": "'num_accelerators' must be a positive integer."}),
                400,
            )

        # --- workload_type ---
        workload_type = str(payload.get("workload_type", "training"))

        # --- utilization coercion with auto-detect percentage vs fraction ---
        utilization_raw = payload.get("utilization", 1.0)
        try:
            utilization = float(utilization_raw)  # type: ignore[arg-type]
            # Auto-scale: treat values > 1 as a percentage (e.g. 80 -> 0.80)
            if utilization > 1.0:
                utilization = utilization / 100.0
        except (TypeError, ValueError):
            return (
                jsonify(
                    {
                        "error": (
                            "'utilization' must be a number in (0, 1] "
                            "or a percentage in (0, 100]."
                        )
                    }
                ),
                400,
            )

        # --- Run calculation ---
        try:
            result = calculate_impact(
                hardware=str(hardware),
                region=str(region),
                duration_hours=duration_hours,
                num_accelerators=num_accelerators,
                workload_type=workload_type,
                utilization=utilization,
            )
        except CalculatorError as exc:
            logger.debug("CalculatorError in /api/calculate: %s", exc)
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in /api/calculate: %s", exc)
            return jsonify({"error": "Internal server error."}), 500

        report = generate_json_report(result)
        return jsonify(report), 200

    # ------------------------------------------------------------------
    # REST API — GET /api/hardware
    # ------------------------------------------------------------------

    @app.get("/api/hardware")
    def api_hardware() -> tuple[Response, int]:
        """Return a list of all supported hardware presets with metadata.

        Returns:
            JSON object with:

            * ``hardware`` — list of objects with keys ``key``, ``label``,
              ``vendor``, ``tdp_watts``.
            * ``count`` — total number of presets.
        """
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

    # ------------------------------------------------------------------
    # REST API — GET /api/regions
    # ------------------------------------------------------------------

    @app.get("/api/regions")
    def api_regions() -> tuple[Response, int]:
        """Return a list of all supported cloud regions with metadata.

        Returns:
            JSON object with:

            * ``regions`` — list of objects with keys ``key``, ``label``,
              ``provider``, ``carbon_intensity_g_per_kwh``, ``pue``, ``wue``.
            * ``count`` — total number of regions.
        """
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


# ---------------------------------------------------------------------------
# Error handler registration
# ---------------------------------------------------------------------------


def _register_error_handlers(app: Flask) -> None:
    """Register HTTP error handlers on the Flask application.

    All handlers return JSON-formatted error responses to ensure API clients
    always receive machine-readable error information.

    Args:
        app: The Flask application to register handlers on.
    """

    @app.errorhandler(400)
    def bad_request(exc: Exception) -> tuple[Response, int]:
        """Handle 400 Bad Request.

        Args:
            exc: The exception that triggered the handler.

        Returns:
            JSON error response with status 400.
        """
        # Prefer the exception message if it's a Werkzeug HTTPException
        description = getattr(exc, "description", str(exc)) or "Bad request."
        return jsonify({"error": description}), 400

    @app.errorhandler(404)
    def not_found(exc: Exception) -> tuple[Response, int]:
        """Handle 404 Not Found.

        Args:
            exc: The exception that triggered the handler.

        Returns:
            JSON error response with status 404.
        """
        return jsonify({"error": "Resource not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(exc: Exception) -> tuple[Response, int]:
        """Handle 405 Method Not Allowed.

        Args:
            exc: The exception that triggered the handler.

        Returns:
            JSON error response with status 405.
        """
        return jsonify({"error": "Method not allowed."}), 405

    @app.errorhandler(500)
    def internal_error(exc: Exception) -> tuple[Response, int]:
        """Handle 500 Internal Server Error.

        Logs the full traceback at ERROR level before returning a generic
        message so that internal details are not exposed to clients.

        Args:
            exc: The exception that triggered the handler.

        Returns:
            JSON error response with status 500.
        """
        logger.exception("Unhandled internal error: %s", exc)
        return jsonify({"error": "Internal server error."}), 500


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _group_hardware_by_vendor() -> dict[str, list[dict[str, str]]]:
    """Build a vendor-grouped mapping of hardware options for template rendering.

    Returns:
        Dict mapping vendor name (str) to a list of
        ``{"key": ..., "label": ...}`` dicts, ordered as returned by
        :func:`~dc_impact.calculator.get_available_hardware`.
    """
    hardware_by_vendor: dict[str, list[dict[str, str]]] = {}
    for key in get_available_hardware():
        meta = HARDWARE_METADATA.get(key, {"label": key, "vendor": "other"})
        vendor = meta.get("vendor", "other")
        hardware_by_vendor.setdefault(vendor, [])
        hardware_by_vendor[vendor].append({"key": key, "label": meta.get("label", key)})
    return hardware_by_vendor


def _group_regions_by_provider() -> dict[str, list[dict[str, str]]]:
    """Build a provider-grouped mapping of region options for template rendering.

    Returns:
        Dict mapping provider name (str) to a list of
        ``{"key": ..., "label": ...}`` dicts, ordered as returned by
        :func:`~dc_impact.calculator.get_available_regions`.
    """
    regions_by_provider: dict[str, list[dict[str, str]]] = {}
    for key in get_available_regions():
        meta = REGION_METADATA.get(key, {"label": key, "provider": "other"})
        provider = meta.get("provider", "other")
        regions_by_provider.setdefault(provider, [])
        regions_by_provider[provider].append(
            {"key": key, "label": meta.get("label", key)}
        )
    return regions_by_provider


def _render_index_with_error(
    error_message: str,
    form_values: dict[str, Any] | None = None,
) -> str:
    """Render the index template with an inline error banner.

    Re-builds the hardware/region option groups the same way as the
    :func:`index` route so the dropdowns remain populated after an error.

    Args:
        error_message: Human-readable error text to display.
        form_values: Previously submitted form values used to pre-populate
            the form fields after the error, so the user does not have to
            re-enter everything.

    Returns:
        Rendered HTML string for the index page in the error state.
    """
    return render_template(
        "index.html",
        error=error_message,
        form_values=form_values or {},
        hardware_by_vendor=_group_hardware_by_vendor(),
        regions_by_provider=_group_regions_by_provider(),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the ``dc_impact`` console script.

    Starts the Flask development server on ``0.0.0.0:5000`` with debug
    mode enabled.  For production deployments use a WSGI server such as
    Gunicorn or uWSGI instead::

        gunicorn 'dc_impact:create_app()' --bind 0.0.0.0:8000
    """
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
