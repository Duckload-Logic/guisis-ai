import re
import os
import json
import logging
import httpx
from fastapi import HTTPException
from src.schemas.prediction import (
    ClassificationRequest,
    ClassificationResponse,
)
from src.core.config import settings

logger = logging.getLogger(__name__)


class ClassifierService:
    """
    Service layer for classification logic.
    Routes queries to Hugging Face and applies business rules locally.
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

    # Class-level variables for lazy loading/caching of local model
    _tokenizer = None
    _model = None
    _device = None
    _label_mapping = None

    def __init__(self):
        pass

    @classmethod
    def _load_local_model(cls):
        """
        Lazily loads the local tokenizer and model.
        """
        if cls._model is not None:
            return

        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )

        model_path = settings.model_path
        if os.path.exists(model_path):
            # Load local model
            mapping_path = os.path.join(model_path, "label_mapping.json")
            with open(mapping_path, "r") as f:
                cls._label_mapping = json.load(f)

            cls._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            cls._tokenizer = AutoTokenizer.from_pretrained(model_path)
            cls._model = AutoModelForSequenceClassification.from_pretrained(
                model_path
            ).to(cls._device)
            cls._model.eval()
        elif "/" in model_path:
            # Load from Hugging Face Hub (Private or Public Repository)
            if settings.hf_token:
                import huggingface_hub
                huggingface_hub.login(settings.hf_token)

            try:
                from huggingface_hub import hf_hub_download
                mapping_file = hf_hub_download(
                    repo_id=model_path,
                    filename="label_mapping.json",
                    token=settings.hf_token,
                )
                with open(mapping_file, "r") as f:
                    cls._label_mapping = json.load(f)
            except Exception as e:
                logger.warning(
                    f"[ClassifierService] {{LocalInference}}: "
                    f"Could not load label_mapping.json from Hub. "
                    f"Using default mapping. Error: {str(e)}"
                )
                cls._label_mapping = {
                    "id2label": {
                        "0": "LOW",
                        "1": "MEDIUM",
                        "2": "HIGH",
                        "3": "CRITICAL"
                    }
                }

            cls._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            cls._tokenizer = AutoTokenizer.from_pretrained(model_path)
            cls._model = AutoModelForSequenceClassification.from_pretrained(
                model_path
            ).to(cls._device)
            cls._model.eval()
        else:
            logger.error(
                f"[ClassifierService] {{LocalInference}}: "
                f"Model path not found: {model_path}"
            )
            raise FileNotFoundError(
                f"Model path not found: {model_path}"
            )

    def _classify_locally(self, text: str) -> ClassificationResponse:
        """
        Performs local inference using PyTorch and Hugging Face Transformers.
        """
        import torch

        self._load_local_model()

        # Tokenize inputs
        inputs = self._tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            pred_id = int(torch.argmax(logits, dim=-1).cpu().item())

        # Retrieve label mapping details
        id2label = self._label_mapping.get("id2label", {})
        # Map predictions
        label = id2label.get(str(pred_id), "LOW")

        # Softmax to get confidence/score
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        confidence = float(probabilities[pred_id])

        return ClassificationResponse(
            level=label,
            confidence=confidence,
            metadata={
                "model_type": "local-distilbert",
                "device": str(self._device),
            },
        )

    def _anonymize_text(self, text: str) -> str:
        """
        Strips sensitive student metadata (names, student numbers, emails)
        from payload before sending to HF.
        """
        # Redact student numbers (e.g. 2023-12345-MN-0)
        student_num_pattern = r"\b\d{4}-\d{5}-[A-Z]{2,3}-\d\S*\b"
        clean = re.sub(student_num_pattern, "[STUDENT_NUMBER_REDACTED]", text)

        # Redact emails
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        clean = re.sub(email_pattern, "[EMAIL_REDACTED]", clean)

        # Redact section info
        clean = re.sub(
            r"(?i)\b(section)\s+([a-zA-Z0-9\-]+)\b",
            r"\1 [REDACTED]",
            clean
        )

        # Redact name indicators (e.g., "my name is X", "ako si X")
        clean = re.sub(
            r"\b((?i:my name is|ako si|i am|hi,? I'm|hello,? I'm))\s+"
            r"([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)",
            r"\1 [NAME_REDACTED]",
            clean
        )

        return clean

    async def classify(
        self, request: ClassificationRequest
    ) -> ClassificationResponse:
        """
        Performs inference using HF Inference API and local business rules.
        """
        try:
            # 1. Anonymize metadata in-transit
            clean_text = self._anonymize_text(request.text)

            # Check if local model or HF Hub model is configured
            if (
                os.path.exists(settings.model_path)
                or "/" in settings.model_path
            ):
                try:
                    result = self._classify_locally(clean_text)
                    return self._apply_business_rules(request.text, result)
                except Exception as local_err:
                    logger.warning(
                        f"[ClassifierService] {{LocalInference}}: "
                        f"Local inference failed, falling back to API: "
                        f"{str(local_err)}"
                    )

            # 2. Call Hugging Face Serverless Inference API (Fallback)
            headers = {}
            if settings.hf_token:
                headers["Authorization"] = f"Bearer {settings.hf_token}"

            payload = {"inputs": clean_text}

            # Enforce 10s maximum client execution timeout
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    settings.hf_classify_url,
                    json=payload,
                    headers=headers
                )

            # 3. Handle non-200 responses gracefully
            if response.status_code != 200:
                logger.error(
                    f"[ClassifierService] {{CallHFInferenceAPI}}: "
                    f"HF Inference API returned status "
                    f"{response.status_code}: {response.text}"
                )
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "AI Classification Service (Hugging Face) failed: "
                        f"Status {response.status_code}"
                    )
                )

            predictions = response.json()
            if not isinstance(predictions, list) or not predictions:
                raise ValueError("Invalid response format from Inference API")

            target_list = (
                predictions[0]
                if isinstance(predictions[0], list)
                else predictions
            )

            # Find predictions with highest confidence score
            best_pred = max(target_list, key=lambda x: x.get("score", 0.0))
            level = best_pred.get("label", "LOW")
            confidence = float(best_pred.get("score", 0.0))

            result = ClassificationResponse(
                level=level,
                confidence=confidence,
                metadata={
                    "model_type": "huggingface-serverless",
                    "device": "cloud-serverless"
                }
            )

            # 4. Apply local business rules
            return self._apply_business_rules(request.text, result)

        except httpx.RequestError as e:
            logger.error(
                f"[ClassifierService] {{CallHFInferenceAPI}}: "
                f"Connection error: {str(e)}"
            )
            raise HTTPException(
                status_code=502,
                detail=f"AI Classification Service connection failed: {e}"
            )
        except Exception as e:
            logger.error(
                f"[ClassifierService] {{Inference}}: "
                f"Inference error: {str(e)}"
            )
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(
                status_code=500,
                detail=f"Internal Server Error during classification: {e}"
            )

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

        is_scheduling = any(
            kw in text_lower for kw in self.SCHEDULING_KEYWORDS
        )
        is_routine_inquiry = any(
            re.search(rf"\b{re.escape(kw)}\b", text_lower)
            for kw in self.ROUTINE_KEYWORDS
        )

        # Don't downgrade if it's a "class schedule conflict" (which is MEDIUM)
        is_collision = "conflict" in text_lower

        if (
            is_routine_inquiry or (is_scheduling and not is_collision)
        ) and level == "MEDIUM":
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
