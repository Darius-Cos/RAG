from __future__ import annotations
import os
import base64
from contextlib import contextmanager


class _NoOpObs:
    def update(self, **kwargs): pass
    def end(self): pass


class _NoOpClient:
    @contextmanager
    def start_as_current_observation(self, **kwargs):
        yield _NoOpObs()
    def flush(self): pass


_client = None


def get_langfuse_client():
    global _client
    if _client is not None:
        return _client

    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not pk or not sk or "..." in pk:
        _client = _NoOpClient()
        return _client

    try:
        token = base64.b64encode(f"{pk}:{sk}".encode()).decode()
        os.environ["LANGFUSE_HOST"] = host
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host}/api/public/otel"
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {token}"
        from langfuse import get_client
        _client = get_client()
    except Exception as e:
        print(f"Langfuse error: {e}")
        _client = _NoOpClient()

    return _client
