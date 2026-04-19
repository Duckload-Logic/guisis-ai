import torch
import torch.nn.functional as F
import re
from typing import Dict
import logging
from src.infrastructure.model_loader import ModelLoader
from src.schemas.prediction import ClassificationRequest, ClassificationResponse
from src.core.config import settings

logger = logging.getLogger(__name__)

class ClassifierService:
    """
    Service layer for classification logic.
    Decouples the API layer from the model loading and inference logic.
    """

    CRISIS_KEYWORDS = [
        "tulong", "help", "emergency", "saklolo", "suko", "suicide",
        "ayoko na", "di ko na kaya", "mawala", "disappear", "ending it",
        "self harm", "hurting myself", "die", "mamatay"
    ]

    SCHEDULING_KEYWORDS = ["schedule", "appointment", "pa-schedule"]

    ROUTINE_KEYWORDS = [
        "library hours", "campus map", "student id", "paano", "how to",
        "where is", "saan", "tanong lang", "inquire", "appointment"
    ]

    HIGH_KEYWORDS = [
        "threatening", "scared", "natatakot", "nahimatay", "revoked",
        "graduation", "cannot pay", "overwhelmed", "sobrang", "suicidal",
        "scholarship", "tuition", "grades", "sis", "emergency", "fainted"
    ]

    def __init__(self):
        # Ensure model is ready on initialization
        self.tokenizer, self.model = ModelLoader.load_model()

    def classify(
        self, request: ClassificationRequest
    ) -> ClassificationResponse:
        """
        Performs inference on the provided text.
        """
        logger.info(f"[Service] Classifying text: {request.text[:50]}...")

        try:
            # Preprocessing
            inputs = self.tokenizer(
                request.text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )

            # Move to device
            inputs = {k: v.to(settings.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Post-processing
            probabilities = F.softmax(logits, dim=-1)
            confidence, predicted_idx = torch.max(probabilities, dim=-1)

            # Convert to scalar
            confidence_score = confidence.item()
            label_id = predicted_idx.item()

            # Label mapping from model config
            # Transformers id2label keys can be either int or str depending on
            # how they were loaded
            level = self.model.config.id2label.get(label_id) or \
                    self.model.config.id2label.get(str(label_id)) or \
                    f"UNKNOWN_{label_id}"

            # Apply internal business rules for refinement
            result = ClassificationResponse(
                level=level,
                confidence=float(confidence_score),
                metadata={
                    "model_type": self.model.config.model_type,
                    "device": str(settings.device)
                }
            )

            return self._apply_business_rules(request.text, result)

        except Exception as e:
            logger.error(f"[Service] Inference error: {str(e)}")
            raise RuntimeError(f"Failed to perform classification: {e}")

    def _apply_business_rules(
        self, text: str, response: ClassificationResponse
    ) -> ClassificationResponse:
        """
        Adjusts prediction based on explicit keywords and university business
        logic.
        """
        text_lower = text.lower().strip()
        level = response.level
        confidence = response.confidence

        # Short phrases with crisis keywords are almost always CRITICAL
        if any(
            re.search(rf"\b{re.escape(kw)}\b", text_lower)
            for kw in self.CRISIS_KEYWORDS
        ) and len(text_lower.split()) <= 7:
            response.level = "CRITICAL"
            response.confidence = max(confidence, 0.95)
            response.metadata["rule_applied"] = "crisis_safety_rail"
            return response

        is_scheduling = any(kw in text_lower for kw in self.SCHEDULING_KEYWORDS)
        is_routine_inquiry = any(
            re.search(rf"\b{re.escape(kw)}\b", text_lower)
            for kw in self.ROUTINE_KEYWORDS
        )

        # Don't downgrade if it's a "class schedule conflict" (which is MEDIUM)
        is_collision = "conflict" in text_lower

        if (is_routine_inquiry or (is_scheduling and not is_collision)) and \
                level == "MEDIUM":
            response.level = "LOW"
            response.metadata["rule_applied"] = "routine_downgrade"
            return response

        if any(
            re.search(rf"\b{re.escape(kw)}\b", text_lower)
            for kw in self.HIGH_KEYWORDS
        ) and level == "MEDIUM":
            response.level = "HIGH"
            response.metadata["rule_applied"] = "urgency_upgrade"
            return response

        return response
