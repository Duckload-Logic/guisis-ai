# GuiSIS AI Service

This service provides machine learning capabilities for the PUP Student
Guidance System Capstone (GuiSIS), including student concern classification
and OCR-based Certificate of Registration (COR) processing.

For contributor and developer guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Tech Stack

- **Framework**: FastAPI (Python 3.11)
- **Machine Learning**: PyTorch + Transformers (DistilBERT)
- **OCR Engine**: Tesseract OCR
- **Deployment**: Docker / Hugging Face Spaces

## API Endpoints

All application routes require the `X-API-Key` header for authorization.

- **GET** `/health`
  System health check endpoint.
- **POST** `/api/v1/classify`
  Classifies student concerns into urgency levels (`LOW`, `MEDIUM`, `HIGH`,
  `CRITICAL`).
- **POST** `/api/v1/ocr`
  Performs raw OCR on uploaded documents.
- **POST** `/api/v1/ocr/validate`
  Runs structure validation tests on uploaded documents.
- **POST** `/api/v1/ocr/cor`
  Extracts structured text/student profile details from COR documents.

## Local Configuration

Create a `.env` file under `src/` directory:

```env
MODEL_PATH="johnalbet/guisis-urgency-distilbert"
MODEL_SOURCE="local"
HF_TOKEN="your_huggingface_access_token_here"
ALLOWED_API_KEY="your_api_key_for_gateway"
```

## Setup Instructions

### Prerequisites

- Python 3.11 installed
- Tesseract OCR engine installed locally
  - Windows: Add Tesseract path to System PATH.
  - Linux: `sudo apt-get install tesseract-ocr`

### Installation

1. Create a python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the development server:
   ```bash
   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   ```
