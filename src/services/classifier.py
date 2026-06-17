# src/services/classifier_service.py

import re
import os
import json
import logging
import copy
import httpx
from typing import Any, Dict, List, Optional, Union

from fastapi import HTTPException
from src.schemas.prediction import (
    ClassificationRequest,
    ClassificationResponse,
)
from src.core.config import settings
from src.utils.text_cleaning import anonymize_text

logger = logging.getLogger(__name__)

DEFAULT_RULES: Dict[str, Any] = {
    "crisis_keywords": [
        "tulong", "help", "emergency", "saklolo", "suko", "suicide",
        "ayoko na", "di ko na kaya", "mawala", "disappear", "ending it",
        "self harm", "hurting myself", "die", "mamatay"
    ],
    "crisis_short_word_limit": 7,
    "crisis_flag_long_messages": True,
    "scheduling_keywords": ["schedule", "appointment", "pa-schedule"],
    "routine_patterns": [
        {
            "pattern": r"\b(library\s+hours|campus\s+map|student\s+id)\b",
            "require_near": None,
            "window": 0,
            "exceptions": []
        },
        {
            "pattern": r"\b(paano|how\s+to|where\s+is|saan|inquire)\b",
            "require_near": None,
            "window": 0,
            # if "conflict" is present, don't downgrade
            "exceptions": ["conflict"]
        },
    ],
    "high_risk_patterns": [
        {
            "pattern": r"\bgraduation\b",
            "require_near": [r"\b(?:denied|revoked|cannot|won't|problem|issue|delayed)\b"],
            "window": 5,
        },
        {
            "pattern": r"\b(?:suicidal|suicide|ending\s+my\s+life|kill\s+myself)\b",
            "require_near": None,
            "window": 0,
        },
        {
            "pattern": r"\b(?:overwhelmed|sobrang)\b",
            "require_near": [r"\b(?:anxiety|stress|depress|pagod|di\s+ko\s+na\s+kaya)\b"],
            "window": 5,
        },
        {
            "pattern": r"\b(?:threatening|scared|natatakot|nahimatay)\b",
            "require_near": None,
            "window": 0,
        },
        {
            "pattern": r"\b(?:scholarship|tuition|grades|sis)\b",
            "require_near": [r"\b(?:problema|issue|cannot|di\s+ko|may\s+di)\b"],
            "window": 5,
        },
        {
            "pattern": r"\b(?:fainted|emergency)\b",
            "require_near": None,
            "window": 0,
        },
    ],
    # Confidence thresholds for rules
    # (if model confidence >= threshold, rule is suppressed)
    "confidence_thresholds": {
        "urgency_upgrade": 0.80,
        "routine_downgrade": 0.65,
    }
}


class ClassifierService:
    """
    Service layer for classification logic.
    Routes queries to the local DistilBERT model (or Hugging Face fallback)
    and applies context‑aware business rules.
    """

    # Class‑level variables for lazy loading/caching of local model
    _tokenizer = None
    _model = None
    _device = None
    _label_mapping = None

    # Rules loaded once per process
    _rules: Optional[Dict[str, Any]] = None

    def __init__(self):
        pass

    @classmethod
    def load_rules(cls) -> Dict[str, Any]:
        """Load business rules from the built-in defaults."""
        if cls._rules is not None:
            return cls._rules

        cls._rules = copy.deepcopy(DEFAULT_RULES)
        logger.info("[Rules] Using built-in default rules.")
        return cls._rules

    @classmethod
    def reload_rules(cls) -> None:
        """Force reload of rules (useful for a management endpoint)."""
        cls._rules = None
        cls.load_rules()

    @staticmethod
    def get_device():
        import torch
        if torch.cuda.is_available():
            logger.info("[Device] CUDA GPU detected.")
            return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("[Device] Apple MPS detected.")
            return torch.device("mps")

        logger.info("[Device] No GPU found, using CPU.")
        return torch.device("cpu")

    @classmethod
    def _load_local_model(cls):
        """Lazily loads the tokenizer and DistilBERT model."""
        if cls._model is not None:
            return

        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )

        model_path = settings.model_path

        if os.path.exists(model_path):
            # Local directory
            mapping_path = os.path.join(model_path, "label_mapping.json")
            with open(mapping_path, "r") as f:
                cls._label_mapping = json.load(f)

            cls._device = cls.get_device()
            cls._tokenizer = AutoTokenizer.from_pretrained(model_path)
            cls._model = AutoModelForSequenceClassification.from_pretrained(
                model_path
            ).to(cls._device)
            cls._model.eval()

        elif "/" in model_path:
            # Hugging Face Hub repository
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
                    f"[ClassifierService] Could not load label_mapping.json "
                    f"from Hub. "
                    f"Using default mapping. Error: {e}"
                )
                cls._label_mapping = {
                    "id2label": {
                        "0": "LOW",
                        "1": "MEDIUM",
                        "2": "HIGH",
                        "3": "CRITICAL",
                    }
                }

            cls._device = cls.get_device()
            cls._tokenizer = AutoTokenizer.from_pretrained(model_path)
            cls._model = AutoModelForSequenceClassification.from_pretrained(
                model_path
            ).to(cls._device)
            cls._model.eval()
        else:
            logger.error(f"Model path not found: {model_path}")
            raise FileNotFoundError(f"Model path not found: {model_path}")

    def _classify_locally(self, text: str) -> ClassificationResponse:
        """Run inference on the local DistilBERT model."""
        import torch

        self._load_local_model()

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

        id2label = self._label_mapping.get("id2label", {})
        label = id2label.get(str(pred_id), "LOW")

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

    async def _classify_via_huggingface(
        self,
        text: str,
        original_text: str,
    ) -> ClassificationResponse:
        """Run inference through the Hugging Face Inference API."""
        headers = {}
        if settings.hf_token:
            headers["Authorization"] = f"Bearer {settings.hf_token}"

        payload = {"inputs": text}

        target_url = str(settings.hf_classify_url).strip()

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0)
        ) as client:
            response = await client.post(
                target_url,
                json=payload,
                headers=headers,
            )

        if response.status_code != 200:
            logger.error(
                f"[ClassifierService] HF API returned "
                f"{response.status_code}: {response.text}"
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    f"AI Classification Service (Hugging Face) "
                    f"failed: Status {response.status_code}"
                ),
            )

        predictions = response.json()
        if not isinstance(predictions, list) or not predictions:
            raise ValueError("Invalid response format from Inference API")

        target_list = (
            predictions[0]
            if isinstance(predictions[0], list)
            else predictions
        )
        best_pred = max(target_list, key=lambda x: x.get("score", 0.0))
        level = best_pred.get("label", "LOW")
        confidence = float(best_pred.get("score", 0.0))

        result = ClassificationResponse(
            level=level,
            confidence=confidence,
            metadata={
                "model_type": "huggingface-serverless",
                "device": "cloud-serverless",
            },
        )

        return self._apply_business_rules(original_text, result)

    def _anonymize_text(self, text: str) -> str:
        """Remove personally identifiable information before forwarding."""
        return anonymize_text(text)

    def _matches_contextual_pattern(
        self, text_lower: str, patterns: List[Dict[str, Any]]
    ) -> bool:
        """
        Returns True if any pattern matches within its required context.
        Each pattern dict may contain:
            - pattern (regex)
            - require_near: list of regexes that must appear near the match
            - window: number of words around the match to scan
        """
        words = text_lower.split()
        for pat in patterns:
            pattern = pat.get("pattern")
            if not pattern:
                continue

            matches = list(re.finditer(pattern, text_lower))
            if not matches:
                continue

            require_near = pat.get("require_near")
            if not require_near:
                return True  # no context required -> match

            window = pat.get("window", 5)
            for m in matches:
                # find word indices around the match
                start_word = len(text_lower[:m.start()].split())
                end_word = start_word + len(m.group().split())
                window_start = max(0, start_word - window)
                window_end = min(len(words), end_word + window)
                window_text = " ".join(words[window_start:window_end])
                if any(re.search(req, window_text) for req in require_near):
                    return True
        return False

    def _contains_crisis_keyword(self, text_lower: str) -> bool:
        rules = self.load_rules()
        crisis_words = rules.get(
            "crisis_keywords",
            DEFAULT_RULES["crisis_keywords"],
        )

        return any(
            re.search(rf"\b{re.escape(kw)}\b", text_lower)
            for kw in crisis_words
        )

    def _is_routine_inquiry(self, text_lower: str) -> bool:
        rules = self.load_rules()
        routine = rules.get(
            "routine_patterns",
            DEFAULT_RULES["routine_patterns"],
        )

        # First check basic scheduling keywords
        sched_words = rules.get(
            "scheduling_keywords",
            DEFAULT_RULES["scheduling_keywords"]
        )

        has_scheduling = any(kw in text_lower for kw in sched_words)

        if has_scheduling:
            # Check for exceptions
            # (e.g., "conflict" → not a simple scheduling query)
            for pat in routine:
                if "exceptions" in pat and any(
                    exc in text_lower for exc in pat["exceptions"]
                ):
                    return False
            return True

        # Otherwise check the contextual patterns
        return self._matches_contextual_pattern(text_lower, routine)

    def _matches_high_risk(self, text_lower: str) -> bool:
        rules = self.load_rules()
        high = rules.get(
            "high_risk_patterns",
            DEFAULT_RULES["high_risk_patterns"],
        )

        return self._matches_contextual_pattern(text_lower, high)

    def _apply_business_rules(
        self, text: str, response: ClassificationResponse
    ) -> ClassificationResponse:
        """
        Adjust prediction based on contextual rules, preserving original
        label/confidence for auditability.
        """
        text_lower = text.lower().strip()
        rules = self.load_rules()
        thresholds = rules.get(
            "confidence_thresholds",
            DEFAULT_RULES["confidence_thresholds"],
        )

        # Save original values before any modification
        original_level = response.level
        original_conf = response.confidence
        response.metadata["original_level"] = original_level
        response.metadata["original_confidence"] = original_conf

        # 1. Crisis detection – two‑tier
        if self._contains_crisis_keyword(text_lower):
            short_limit = rules.get("crisis_short_word_limit", 7)
            if len(text_lower.split()) <= short_limit:
                response.level = "CRITICAL"
                response.confidence = max(original_conf, 0.95)
                response.metadata["rule_applied"] = "crisis_safety_rail"
                return response
            else:
                # Flag for review, but don’t change label automatically
                response.metadata["crisis_flag"] = True

        # 2. Urgency upgrade (MEDIUM → HIGH) only
        # if model confidence is below threshold
        if original_level == "MEDIUM" and self._matches_high_risk(text_lower):
            threshold = thresholds.get("urgency_upgrade", 0.80)
            if original_conf < threshold:
                response.level = "HIGH"
                response.metadata["rule_applied"] = "urgency_upgrade"
            else:
                # Model is confident; just flag for review
                response.metadata["flag"] = "review_possible_high"

        # 3. Routine downgrade (MEDIUM → LOW) with multiple safeguards
        if original_level == "MEDIUM" and self._is_routine_inquiry(text_lower):
            threshold = thresholds.get("routine_downgrade", 0.65)
            # Only downgrade if model is not confident,
            # and there are no crisis/high-risk words
            if (
                original_conf < threshold
                and not self._contains_crisis_keyword(text_lower)
                and not self._matches_high_risk(text_lower)
            ):
                response.level = "LOW"
                response.metadata["rule_applied"] = "routine_downgrade"

        return response

    # ------------------------------------------------------------------
    # Main classify endpoint
    # ------------------------------------------------------------------
    async def classify(
        self, request: ClassificationRequest
    ) -> ClassificationResponse:
        """Full pipeline: anonymise → infer → apply business rules."""
        try:
            clean_text = self._anonymize_text(request.text)

            if settings.hf_classify_url:
                logger.info("[ClassifierService] Using Hugging Face inference API.")
                return await self._classify_via_huggingface(
                    clean_text,
                    request.text,
                )

            # Attempt local model first
            if (
                os.path.exists(settings.model_path)
                or "/" in settings.model_path
            ):
                try:
                    result = self._classify_locally(clean_text)
                    return self._apply_business_rules(request.text, result)
                except Exception as local_err:
                    logger.warning(
                        f"[ClassifierService] Local inference failed, "
                        f"falling back to API: {local_err}"
                    )

            raise FileNotFoundError(
                f"No usable local model found at {settings.model_path}"
            )

        except httpx.RequestError as e:
            logger.error(f"[ClassifierService] Connection error: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"AI Classification Service connection failed: {e}"
            )
        except Exception as e:
            logger.error(f"[ClassifierService] Inference error: {e}")
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(
                status_code=500,
                detail=f"Internal Server Error during classification: {e}"
            )
