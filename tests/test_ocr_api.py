import sys
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.getcwd())

from src.main import app

def test_ocr_endpoint_invalid_file():
    """Verify that the API handles non-image files gracefully."""
    with TestClient(app) as client:
        # Send a text file as if it were an image
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/api/v1/ocr", files=files)
        
        # Should return 500 because our service raises ValueError for decoded image is None
        assert response.status_code == 500
        assert "Could not decode image" in response.json()["detail"]

def test_ocr_endpoint_success():
    """Verify the /ocr endpoint workflow with mocked service."""
    # Mock the OCRService.process_document method
    mock_response = {
        "filename": "test.png",
        "total_pages": 1,
        "full_text": "Sample OCR Content",
        "pages": [],
        "processing_time_ms": 100.0,
        "metadata": {"engine": "Mocked"}
    }
    
    with patch("src.services.ocr.OCRService.process_document", 
               return_value=mock_response):
        with TestClient(app) as client:
            # Create a dummy image
            dummy_image = b"fake-png-data"
            files = {"file": ("test.png", dummy_image, "image/png")}
            
            response = client.post("/api/v1/ocr", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert data["filename"] == "test.png"
            assert data["full_text"] == "Sample OCR Content"

if __name__ == "__main__":
    test_ocr_endpoint_success()
    test_ocr_endpoint_invalid_file()
    print("API Integration Tests Passed!")
