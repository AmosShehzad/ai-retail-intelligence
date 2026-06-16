"""
Global error handlers for the FastAPI application.

Why custom handlers:
- FastAPI's default errors return technical tracebacks
- Custom handlers return clean JSON your frontend can display
- Catches errors in one place instead of every endpoint
"""

import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

log = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    """Registers all custom error handlers onto the app."""

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError
    ):
        """
        Handles Pydantic validation errors (wrong data types sent by client).
        Default FastAPI returns a complex nested error — this simplifies it.
        Example: client sends string where int expected → clean error message
        """
        errors = []
        for error in exc.errors():
            errors.append({
                "field"  : " → ".join(str(e) for e in error["loc"]),
                "message": error["msg"],
                "type"   : error["type"],
            })

        log.warning("Validation error on %s: %s", request.url, errors)

        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error"  : "Validation failed",
                "details": errors,
            }
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """
        Handles ValueError raised in your analytics functions.
        Example: get_revenue_by_period() called with invalid period string
        """
        log.error("ValueError on %s: %s", request.url, str(exc))
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error"  : "Bad request",
                "detail" : str(exc),
            }
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        """
        Catches ALL other unhandled exceptions.
        Prevents raw Python tracebacks from reaching the client.
        Logs full error server-side for debugging.
        """
        log.exception("Unhandled exception on %s", request.url)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error"  : "Internal server error",
                "detail" : "Something went wrong. Check server logs.",
            }
        )