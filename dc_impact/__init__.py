"""dc_impact package — environmental footprint calculator for AI workloads.

This package exposes a Flask application factory (``create_app``) that wires
together the calculator engine, static data tables, and report generation
modules into a self-contained web application.

Typical usage::

    from dc_impact import create_app

    app = create_app()
    app.run()
"""

from dc_impact.app import create_app

__all__ = ["create_app"]
__version__ = "0.1.0"
