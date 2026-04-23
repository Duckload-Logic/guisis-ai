import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from src.services.ocr import OCRService
from src.schemas.ocr import OCRResponse

class TestOCRService(unittest.TestCase):
    def setUp(self):
        # Mock the PaddleOCR engine to avoid heavy loading
        self.mock_ocr_engine = MagicMock()
        
        # Patch the ModelLoader to return our mock engine
        with patch('src.infrastructure.model_loader.ModelLoader.load_ocr_engine', 
                   return_value=self.mock_ocr_engine):
            self.service = OCRService()

    def test_enhance_image_returns_ndarray(self):
        """Verify that pre-processing returns a valid numpy array."""
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        enhanced = self.service._enhance_image(dummy_img)
        
        self.assertIsInstance(enhanced, np.ndarray)
        self.assertEqual(len(enhanced.shape), 2) # Should be grayscale

    @patch('fastapi.UploadFile')
    def test_process_document_success(self, mock_file):
        """Verify OCR response mapping logic."""
        import asyncio
        asyncio.run(self._process_document_success(mock_file))

    async def _process_document_success(self, mock_file):
        # Setup mock file
        mock_file.filename = "test.png"
        mock_file.read.return_value = b"fake-image-bytes"
        
        # Mock cv2.imdecode to return a dummy image
        with patch('cv2.imdecode', return_value=np.zeros((10, 10, 3), dtype=np.uint8)):
            # Mock OCR results: [ [[ [x,y]... ], ("text", confidence) ]]
            self.mock_ocr_engine.ocr.return_value = [[
                [[[0,0], [1,0], [1,1], [0,1]], ("Hello World", 0.95)]
            ]]
            
            result = await self.service.process_document(mock_file)
            
            self.assertIsInstance(result, OCRResponse)
            self.assertEqual(result.filename, "test.png")
            self.assertEqual(result.full_text, "Hello World")
            self.assertEqual(len(result.pages), 1)
            self.assertEqual(result.pages[0].words[0].text, "Hello World")

if __name__ == "__main__":
    unittest.main()
