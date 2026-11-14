from time import time
from fastapi.concurrency import run_in_threadpool
import logging
from starlette.middleware.base import BaseHTTPMiddleware

from app.auth.store import log_usage
from app.api import deps

log = logging.getLogger(__name__)   # -> "app.middleware.usage"

# app/middleware/usage.py
class UsageMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        t0 = time.perf_counter()
        status = 500                      # if call_next raises, it's a 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            if request.url.path.startswith("/v1/"):
                try:
                    u = getattr(request.state, "usage", {})
                    await run_in_threadpool(
                        log_usage,
                        getattr(request.state, "key_id", None),
                        request.url.path,
                        u.get("label", ""),
                        u.get("confidence", 0.0),
                        int((time.perf_counter() - t0) * 1000),
                        status,
                        deps.client_ip(request),
                    )
                except Exception:
                    log.warning("usage log failed", exc_info=True)