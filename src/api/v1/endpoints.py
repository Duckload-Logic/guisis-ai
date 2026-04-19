from fastapi import APIRouter, HTTPException, Depends
from src.schemas.prediction import ClassificationRequest, ClassificationResponse
from src.services.classifier import ClassifierService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency for service injection
def get_classifier_service():
    return ClassifierService()

@router.post("/classify", response_model=ClassificationResponse)
async def post_predict(
    request: ClassificationRequest,
    service: ClassifierService = Depends(get_classifier_service)
):
    """
    Endpoint to classify the urgency of a student concern or appointment reason.
    """
    try:
        result = service.classify(request)
        return result
    except Exception as e:
        logger.error(f"[Endpoint] Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error during classification"
        )
