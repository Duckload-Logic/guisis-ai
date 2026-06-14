import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Constants for project roots
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = os.getenv("MODEL_PATH", "ai_models/distilbert/model/outputs")

# Resolve model path: absolute path if local, else use repo ID directly
if os.path.exists(str(BASE_DIR / MODEL_PATH)):
    RESOLVED_MODEL_PATH = str(BASE_DIR / MODEL_PATH)
else:
    RESOLVED_MODEL_PATH = MODEL_PATH

class Settings(BaseModel):
    """
    Application settings and configuration.
    Uses Pydantic for validation and type safety.
    """
    app_name: str = "GuiSIS AI Classification API"
    version: str = "1.0.0"

    allowed_api_key: str = os.getenv("ALLOWED_API_KEY")

    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))

    # ML Model Config
    model_path: str = RESOLVED_MODEL_PATH
    device: str = "cpu"  # Force CPU for stability

    # API Config
    api_v1_prefix: str = "/api/v1"

settings = Settings()
