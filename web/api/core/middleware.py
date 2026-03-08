"""
core/middleware.py
------------------
Custom Starlette middleware for DAPTA API.
"""

from __future__ import annotations
import logging
import traceback

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("dapta.api")


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Catches any unhandled exception that escapes a route handler and
    returns a safe JSON response instead of leaking a stack trace.

    In development, the error detail is included in the response body.
    In production, only a generic message is returned — full trace is logged.
    """

    def __init__(self, app, debug: bool = False) -> None:
        super().__init__(app)
        self.debug = debug

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(
                "Unhandled exception on %s %s\n%s",
                request.method, request.url, tb,
            )
            body = {"detail": "An internal server error occurred."}
            if self.debug:
                body["debug"] = str(exc)
            return JSONResponse(status_code=500, content=body)
