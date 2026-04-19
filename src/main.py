from fastapi import FastAPI
from src.api.v1.endpoints import router as api_v1_router
from src.core.config import settings
from src.infrastructure.model_loader import ModelLoader
import logging
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Application factory for Ogos AI API."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="Text classification for student concerns and appointment urgency."
    )

    origins = [
        "http://localhost:8080",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Startup event to pre-load the model
    @app.on_event("startup")
    async def startup_event():
        logger.info("[Main] Starting up and pre-loading model...")
        try:
            ModelLoader.load_model()
            logger.info("[Main] Model pre-loaded successfully")
        except Exception as e:
            logger.error(f"[Main] Critical failure during startup: {e}")

    # Include routers
    app.include_router(
        api_v1_router,
        prefix=settings.api_v1_prefix,
        tags=["Classification"]
    )

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": settings.app_name}

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)