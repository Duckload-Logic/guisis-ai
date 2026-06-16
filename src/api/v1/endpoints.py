import logging
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from src.schemas.ocr import (
    CORResponse,
    CORValidationResponse,
    OCRResponse,
)
from src.schemas.prediction import (
    ClassificationRequest,
    ClassificationResponse,
)
from src.services.classifier import ClassifierService
from src.services.ocr import OCRService

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
async def PostClassify(
    request: ClassificationRequest,
    service: ClassifierService = Depends(get_classifier_service)
):
    """
    Endpoint to classify the urgency of a student concern or
    appointment reason.
    """
    try:
        result = await service.classify(request)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"[AI Endpoint] Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error during classification"
        )


@router.post("/ocr", response_model=OCRResponse)
async def PostOCR(
    file: UploadFile = File(...),
    service: OCRService = Depends(get_ocr_service)
):
    """
    Endpoint to perform OCR on a single image or document.
    """
    try:
        result = await service.process_document(file)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"[OCR Endpoint] Processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error during OCR processing: {str(e)}"
        )


@router.post("/ocr/validate", response_model=CORValidationResponse)
async def PostOCRValidate(
    file: UploadFile = File(...),
    service: OCRService = Depends(get_ocr_service)
):
    """
    Endpoint to perform OCR validation on a single document.
    """
    try:
        result = await service.validate_document(file)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"[OCR Endpoint] Processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error during OCR processing: {str(e)}"
        )


@router.post("/ocr/cor", response_model=CORResponse)
async def PostOCRCor(
    file: UploadFile = File(...),
    service: OCRService = Depends(get_ocr_service)
):
    """
    Endpoint to extract structured data from a COR document.
    """
    try:
        result = await service.process_cor(file)
        return result
    except ValueError as e:
        logger.error(f"[OCR Endpoint] Processing failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data provided: {str(e)}"
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"[OCR Endpoint] Processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error during OCR processing: {str(e)}"
        )

