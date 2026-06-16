import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Constants for project roots
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseModel):
    """
    Application settings and configuration.
    Uses Pydantic for validation and type safety.
    """
    app_name: str = "GuiSIS AI Classification API"
    version: str = "1.0.0"

    allowed_api_key: str | None = os.getenv("ALLOWED_API_KEY")

    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))

    # API Config
    api_v1_prefix: str = "/api/v1"

    # Hugging Face Settings
    hf_token: str | None = os.getenv("HF_TOKEN")
    hf_classify_url: str | None = os.getenv("HF_CLASSIFY_URL")

    # Local Model Settings
    model_path: str = os.getenv(
        "MODEL_PATH", "./ai_models/distilbert/model/outputs"
    )


settings = Settings()

