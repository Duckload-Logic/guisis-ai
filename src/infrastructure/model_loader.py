from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional, Tuple
import logging
from src.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Infrastructure layer for loading and caching the ML model.
    Implements a singleton-like pattern for the model and tokenizer.
    """
    _tokenizer: Optional[AutoTokenizer] = None
    _model: Optional[AutoModelForSequenceClassification] = None

    @classmethod
    def load_model(cls) -> Tuple[
        AutoTokenizer, AutoModelForSequenceClassification
    ]:
        """
        Loads the model and tokenizer from the configured path.
        Caches the results to avoid multiple loads.
        """
        if cls._tokenizer is not None and cls._model is not None:
            return cls._tokenizer, cls._model

        logger.info(f"[Loader] Loading model from {settings.model_path}")

        try:
            cls._tokenizer = AutoTokenizer.from_pretrained(settings.model_path)
            cls._model = AutoModelForSequenceClassification.from_pretrained(
                settings.model_path
            )
            cls._model.to(settings.device)
            cls._model.eval()  # Set to evaluation mode
            logger.info("[Loader] Model loaded successfully")
        except Exception as e:
            logger.error(f"[Loader] Failed to load model: {str(e)}")
            raise RuntimeError(f"Could not load classification model: {e}")

        return cls._tokenizer, cls._model

    @classmethod
    def clear_cache(cls):
        """Clears the cached model and tokenizer."""
        cls._tokenizer = None
        cls._model = None
