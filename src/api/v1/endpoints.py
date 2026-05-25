from src.schemas.ocr import CORResponse, CORValidationResponse
from fastapi import APIRouter, HTTPException, Depends
from src.schemas.prediction import ClassificationRequest, ClassificationResponse
from src.schemas.ocr import OCRResponse
from src.services.classifier import ClassifierService
from src.services.ocr import OCRService
from fastapi import UploadFile, File
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency for service injection
def get_classifier_service():
    return ClassifierService()

def get_ocr_service():
    return OCRService()

@router.get("/health")
async def health_check():
    return {"status": "ok"}

@router.post("/classify", response_model=ClassificationResponse)
async def post_predict(
    request: ClassificationRequest,
    service: ClassifierService = Depends(get_classifier_service)
):
    """
    Endpoint to classify the urgency of a student concern or appointment reason.
    """
    try:
        # Since service.classify is async
        result = await service.classify(request)
        return result
    except Exception as e:
        logger.error(f"[AI Endpoint] Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error during classification"
        )

@router.post("/ocr", response_model=OCRResponse)
async def post_ocr(
    file: UploadFile = File(...),
    service: OCRService = Depends(get_ocr_service)
):
    """
    Endpoint to perform OCR on a single image or document.
    """
    try:
        result = await service.process_document(file)
        return result
    except Exception as e:
        logger.error(f"[OCR Endpoint] Processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error during OCR processing: {str(e)}"
        )

@router.post("/ocr/validate", response_model=CORValidationResponse)
async def post_ocr_validate(
    file: UploadFile = File(...),
    service: OCRService = Depends(get_ocr_service)
):
    """
    Endpoint to perform OCR on a single image or document.
    """
    try:
        result = await service.validate_document(file)
        return result
    except Exception as e:
        logger.error(f"[OCR Endpoint] Processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error during OCR processing: {str(e)}"
        )

@router.post("/ocr/cor", response_model=CORResponse)
async def post_ocr_cor(
    file: UploadFile = File(...),
    service: OCRService = Depends(get_ocr_service)
):
    """
    Endpoint to perform OCR on a single image or document.
    """
    try:
        result = await service.process_cor(file)
        return result
    except Exception as e:
        logger.error(f"[OCR Endpoint] Processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error during OCR processing: {str(e)}"
        )
