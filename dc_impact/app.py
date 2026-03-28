"""Flask application factory and route definitions for dc_impact.

This module contains the ``create_app`` factory function and all route
definitions for the calculator UI, the REST API endpoint, and report card
rendering. Routes are implemented as lightweight stubs in phase 1 and will
be fully wired up in phase 5.
"""

from __future__ import annotations

import logging
from typing import Any

from flask import Flask, jsonify, render_template, request

logger = logging.getLogger(__name__)


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        test_config: Optional mapping of configuration overrides used during
            testing (e.g. ``{"TESTING": True}``).

    Returns:
        A fully configured :class:`flask.Flask` application instance.
    """
    app = Flask(__name__, instance_relative_config=True)

    # Default configuration
    app.config.from_mapping(
        SECRET_KEY="dev-secret-change-in-production",
        DEBUG=False,
    )

    if test_config is not None:
        app.config.from_mapping(test_config)

    _register_routes(app)

    return app


def _register_routes(app: Flask) -> None:
    """Register all URL routes on the given Flask application.

    Args:
        app: The Flask application instance to attach routes to.
    """

    @app.get("/")
    def index() -> str:
        """Render the main calculator form UI.

        Returns:
            Rendered HTML string for the calculator index page.
        """
        return render_template("index.html")

    @app.post("/api/calculate")
    def api_calculate() -> tuple[Any, int]:
        """REST API endpoint that accepts JSON workload parameters and returns
        a structured impact report.

        Expected JSON body fields (all required):
            - ``hardware`` (str): Hardware preset key, e.g. ``"A100"``.
            - ``region`` (str): Cloud region key, e.g. ``"us-east-1"``.
            - ``duration_hours`` (float): Wall-clock duration of the workload.
            - ``num_accelerators`` (int): Number of accelerator units used.
            - ``workload_type`` (str): One of ``"training"`` or
              ``"inference"``.
            - ``utilization`` (float, optional): Fractional GPU utilization
              between 0 and 1 (default ``1.0``).

        Returns:
            A tuple of (JSON response, HTTP status code).
        """
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        # Full implementation deferred to phase 5; return placeholder.
        return jsonify({"status": "not_implemented", "received": payload}), 501

    @app.get("/report")
    def report() -> str:
        """Render a shareable HTML report card from query-string parameters.

        Returns:
            Rendered HTML string for the impact report card.
        """
        # Full implementation deferred to phase 5.
        return render_template("report_card.html", result=None)

    @app.errorhandler(404)
    def not_found(exc: Exception) -> tuple[Any, int]:
        """Handle 404 Not Found errors with a JSON response.

        Args:
            exc: The exception that triggered the error handler.

        Returns:
            A tuple of (JSON response, 404 status code).
        """
        return jsonify({"error": "Resource not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(exc: Exception) -> tuple[Any, int]:
        """Handle 405 Method Not Allowed errors with a JSON response.

        Args:
            exc: The exception that triggered the error handler.

        Returns:
            A tuple of (JSON response, 405 status code).
        """
        return jsonify({"error": "Method not allowed."}), 405

    @app.errorhandler(500)
    def internal_error(exc: Exception) -> tuple[Any, int]:
        """Handle 500 Internal Server Error with a JSON response.

        Args:
            exc: The exception that triggered the error handler.

        Returns:
            A tuple of (JSON response, 500 status code).
        """
        logger.exception("Unhandled internal error: %s", exc)
        return jsonify({"error": "Internal server error."}), 500


def main() -> None:
    """Entry point for the ``dc_impact`` command-line script.

    Starts the Flask development server on ``0.0.0.0:5000``.
    """
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
