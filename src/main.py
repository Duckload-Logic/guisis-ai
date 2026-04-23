"""
Application entry point for the Ogos AI FastAPI service.
Initializes middleware, routing, and pre-loads machine learning models.
"""

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v1.endpoints import router as api_v1_router
from src.core.config import settings
from src.infrastructure.model_loader import ModelLoader

# Configure logging with a standardized format for production traceability.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Factory function to configure and return the FastAPI application instance.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "Text classification for student concerns and "
            "appointment urgency."
        )
    )

    # CORS configuration to allow local development traffic.
    origins = ["http://localhost:8080"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        """
        Warm up the system by pre-loading resource-heavy models into memory.
        """
        logger.info("[Main] Starting up and pre-loading model...")
        try:
            ModelLoader.load_model()
            ModelLoader.load_ocr_engine()
            logger.info("[Main] Model pre-loaded successfully")
        except Exception as e:
            logger.error(f"[Main] Critical failure during startup: {e}")

    # Register API versioned routers.
    app.include_router(
        api_v1_router,
        prefix=settings.api_v1_prefix,
        tags=["Classification"]
    )

    @app.get("/health")
    async def health_check():
        """
        Basic endpoint to verify service availability.
        """
        return {"status": "healthy", "service": settings.app_name}

    return app


# Global application instance used by the ASGI server.
app = create_app()


if __name__ == "__main__":
    logger.info("[Main] Starting server...")
    uvicorn.run(app, host=settings.app_host, port=settings.app_port, workers=1)