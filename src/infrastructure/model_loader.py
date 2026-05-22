"""
Infrastructure layer for loading and caching machine learning models.
"""

import logging
import os
import sys
from typing import Any, Optional, Tuple

from paddleocr import PaddleOCR

from src.core.config import settings

# Disable oneDNN/MKLDNN before any paddle imports to prevent Windows crashes.
# This avoids instabilities in fused_conv2d operators on Windows CPU builds.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_onednn"] = "0"

# Configure logging
logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Manages loading and caching of ML models to ensure single-instance access.
    """

    _tokenizer: Optional[Any] = None
    _model: Optional[Any] = None
    _ocr: Optional[PaddleOCR] = None

    @classmethod
    def load_model(
        cls,
    ) -> Tuple[Any, Any]:
        """
        Retrieves the classification model and tokenizer, loading them on demand.

        Returns:
            Tuple containing the tokenizer and the classification model.
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if cls._tokenizer is not None and cls._model is not None:
            return cls._tokenizer, cls._model

        logger.info(f"[Loader] Loading model from {settings.model_path}")

        try:
            cls._tokenizer = AutoTokenizer.from_pretrained(settings.model_path)
            cls._model = AutoModelForSequenceClassification.from_pretrained(
                settings.model_path
            )
            cls._model.to(settings.device)
            cls._model.eval()
            logger.info("[Loader] Model loaded successfully")
        except Exception as e:
            logger.error(f"[Loader] Failed to load model: {str(e)}")
            raise RuntimeError(f"Could not load classification model: {e}")

        return cls._tokenizer, cls._model

    @classmethod
    def load_ocr_engine(cls) -> PaddleOCR:
        """
        Initializes the PaddleOCR engine for text extraction.

        Returns:
            The global PaddleOCR instance.
        """
        if cls._ocr is not None:
            return cls._ocr

        logger.info("[Loader] Initializing PaddleOCR engine...")

        # PaddleOCR has known instability issues on Windows with Python 3.13.
        # Enforcing Python 3.11 ensures compatibility with available builds.
        if sys.version_info >= (3, 13):
            raise RuntimeError(
                "PaddleOCR runtime is unstable on Python 3.13 in this project. "
                "Use Python 3.11 and reinstall requirements."
            )

        try:
            cls._ocr = PaddleOCR(
                use_angle_cls=False,
                lang="en",
                use_gpu=False,
                enable_mkldnn=False,
            )
            logger.info("[Loader] OCR Engine loaded successfully")
        except Exception as e:
            logger.error(f"[Loader] Failed to load OCR engine: {str(e)}")
            raise RuntimeError(f"Could not load OCR engine: {e}")

        return cls._ocr

    @classmethod
    def clear_cache(cls):
        """
        Resets the singleton instances for testing or memory management.
        """
        cls._tokenizer = None
        cls._model = None
        cls._ocr = None
