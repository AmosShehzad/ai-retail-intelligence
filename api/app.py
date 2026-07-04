"""
Day 8: FastAPI Application Factory

This file:
1. Creates the FastAPI app instance
2. Configures CORS (so Streamlit can call this API)
3. Registers global error handlers
4. Registers all routers (analytics, inventory, rag)
5. Sets up startup/shutdown lifecycle events
6. Exposes health check endpoint
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import analytics, inventory, rag, products, purchase_orders

from database.db_manager import initialize_database
from api.error_handlers import register_error_handlers

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)


# In api/app.py, update the lifespan function:

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("AI Retail Intelligence API — Starting Up")
    initialize_database()
    log.info("Database initialized and ready.")

    # Day 16: pre-warm the RAG pipeline at startup
    # Why: loading FAISS + connecting to Ollama takes a few seconds.
    # Doing it now means the FIRST user request isn't slow —
    # the pipeline is already loaded and ready by the time anyone asks.
    from rag.pipeline import get_rag_pipeline
    pipeline = get_rag_pipeline()
    log.info("RAG Pipeline pre-warmed | ready=%s", pipeline.is_ready())

    yield

    log.info("AI Retail Intelligence API — Shutting Down")

# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application.
    Using a factory function (instead of bare module-level app)
    makes testing easier — you can create fresh app instances in tests.
    """

    app = FastAPI(
        title       = "AI Retail Intelligence API",
        description = (
            "Backend API for the Pakistani Kiryana Store AI Assistant. "
            "Provides sales analytics, inventory intelligence, "
            "and RAG-powered natural language query endpoints."
        ),
        version     = "1.0.0",
        lifespan    = lifespan,
        docs_url    = "/docs",      # Swagger UI at localhost:8000/docs
        redoc_url   = "/redoc",     # ReDoc at localhost:8000/redoc
    )

    # ── CORS Configuration ────────────────────────────────────────────────────
    # Why: Streamlit runs on port 8501, FastAPI on 8000.
    # Without CORS, the browser blocks cross-port requests.
    # In production (AWS), replace localhost with your actual domain.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",    # Streamlit default port
            "http://localhost:3000",    # In case you add a React frontend later
            "http://127.0.0.1:8501",
            "*",                        # Allow all during development
                                        # REMOVE "*" in production
        ],
        allow_credentials = True,
        allow_methods     = ["GET", "POST", "PUT", "DELETE"],
        allow_headers     = ["*"],
    )

    # ── Request timing middleware ─────────────────────────────────────────────
    # Logs how long each request takes — useful for spotting slow endpoints
    @app.middleware("http")
    async def log_request_time(request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = round((time.time() - start) * 1000, 2)
        log.info("%s %s — %dms", request.method, request.url.path, duration)
        return response

    # ── Register error handlers ───────────────────────────────────────────────
    register_error_handlers(app)

    # ── Register routers ──────────────────────────────────────────────────────
    # prefix="/api/v1" means all routes become:
    # /api/v1/analytics/...
    # /api/v1/inventory/...
    # /api/v1/rag/...
    # Versioning lets you add /api/v2 later without breaking existing clients
    app.include_router(analytics.router, prefix="/api/v1")
    app.include_router(inventory.router, prefix="/api/v1")
    app.include_router(rag.router,       prefix="/api/v1")
    app.include_router(products.router,        prefix="/api/v1")
    app.include_router(purchase_orders.router, prefix="/api/v1")
    # ── Root health check ─────────────────────────────────────────────────────
    @app.get("/", tags=["Health"])
    async def root():
        """
        Root endpoint — confirms API is running.
        Used by Docker health checks and AWS load balancer probes.
        """
        return {
            "status" : "online",
            "service": "AI Retail Intelligence API",
            "version": "1.0.0",
            "docs"   : "/docs",
        }

    @app.get("/health", tags=["Health"])
    async def health_check():
        try:
            from database.db_manager import get_connection
            conn   = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM products WHERE is_active=1")
            product_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM sales")
            sales_count = cursor.fetchone()[0]
            conn.close()

            # LangSmith status
            from rag.langsmith_config import TRACING_ENABLED, LANGSMITH_PROJECT
            langsmith_status = "enabled" if TRACING_ENABLED else "disabled"

            return JSONResponse(status_code=200, content={
                "status"          : "healthy",
                "database"        : "connected",
                "products_count"  : product_count,
                "sales_count"     : sales_count,
                "langsmith"       : langsmith_status,
                "langsmith_project": LANGSMITH_PROJECT,
            })
        except Exception as e:
            return JSONResponse(status_code=503, content={
                "status": "unhealthy", "error": str(e)
            })

    log.info("App created with %d routes", len(app.routes))
    return app


# ── Create the app instance ───────────────────────────────────────────────────
# This is what Uvicorn imports: uvicorn api.app:app
app = create_app()


# ── Entry point for direct execution ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,   # auto-restart on file changes during development
        log_level="info",
    )