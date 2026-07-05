"""
Error-path tests: the generic exception handler and 500 bodies.

The handler is copied from the real app (same object), so these assert
production behavior, not a test-app approximation.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _app_with_real_handlers():
    import api.main as main_module

    app = FastAPI()
    for exc_class, handler in main_module.app.exception_handlers.items():
        app.add_exception_handler(exc_class, handler)

    @app.get("/boom")
    def boom():
        raise RuntimeError(
            "postgresql://postgres:password@db-internal:5432/postgres exploded"
        )

    return app


class TestGenericExceptionHandler:

    def test_unhandled_exception_returns_500_json(self):
        client = TestClient(_app_with_real_handlers(), raise_server_exceptions=False)
        resp = client.get("/boom")
        assert resp.status_code == 500
        assert resp.json() == {"detail": "Internal server error"}

    def test_unhandled_exception_leaks_nothing(self):
        """Exception text (connection strings, hosts, paths) must never
        reach the client."""
        client = TestClient(_app_with_real_handlers(), raise_server_exceptions=False)
        resp = client.get("/boom")
        body = resp.text
        assert "postgresql://" not in body
        assert "db-internal" not in body
        assert "RuntimeError" not in body


class TestClientFixtureParity:
    """The shared `client` fixture must expose the same handler set as the
    real app — otherwise error-path tests assert fictional behavior."""

    def test_client_app_has_generic_handler(self, client):
        import api.main as main_module
        test_app = client.app
        assert Exception in test_app.exception_handlers
        assert (
            test_app.exception_handlers[Exception]
            is main_module.app.exception_handlers[Exception]
        )
