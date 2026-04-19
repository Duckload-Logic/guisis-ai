# PUPT-OGOS AI Classification API

## Overview

The PUPT-OGOS AI Classification API is a specialized machine learning service developed for the Polytechnic University of the Philippines Taguig (PUPT) Online Guidance and Office System (OGOS). It leverages Natural Language Processing (NLP) to classify student concerns and determine the urgency level of appointments, ensuring that critical student needs are prioritized.

The service uses a fine-tuned DistilBERT model as its core inference engine, complemented by a heuristic rule engine to handle safety-critical edge cases and administrative specificities.

## System Architecture

### 1. API Layer (FastAPI)

The entry point of the application, providing high-performance asynchronous endpoints.

- **Classification Endpoint**: `POST /api/v1/classify`
- **Health Check**: `GET /health`

### 2. Service Layer

The core business logic resides here, specifically in the `ClassifierService`.

- **Inference**: Invokes the DistilBERT model to get base probability distributions across urgency levels.
- **Heuristics Engine**: Applies post-inference rules to upgrade or downgrade urgency based on keyword detection (e.g., "crisis" keywords automatically trigger CRITICAL/HIGH levels).

### 3. Infrastructure Layer

Handles system-level concerns such as model loading and resource management.

- **Model Loader**: Implements a singleton pattern to ensure the heavy Transformer model is loaded into memory only once and shared across requests.

## Urgency Levels

- **LOW**: General inquiries, routine administrative questions (e.g., library hours).
- **MEDIUM**: Standard student concerns requiring guidance (e.g., elective conflicts).
- **HIGH**: Urgent administrative or personal issues (e.g., scholarship revocation, severe stress).
- **CRITICAL**: Immediate safety concerns or crisis situations (e.g., self-harm, threats).

## Technical Stack

- **Framework**: FastAPI
- **ML Engine**: Hugging Face Transformers (DistilBERT)
- **Validation**: Pydantic v2
- **Testing**: Unittest, Pytest
- **Environment**: Python 3.10+

## Getting Started

### Installation

1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements/dev.txt
   ```

### Configuration

Create a `.env` file in the root directory:

```env
MODEL_PATH=./ai_models/distilbert/model/outputs
```

### Running the Application

To start the development server:

```bash
uvicorn src.main:app --reload
```

## Testing and Verification

### Unit and Integration Tests

Run the comprehensive test suite to verify classification logic and business rules:

```bash
python tests/test_suite.py
```

### Interactive CLI Tool

Use the interactive CLI to test the model with real-time inputs:

```bash
python scripts/test_cli.py --interactive
```

## Contributor Guidelines

- **Type Safety**: Use Pydantic models for all data transfers.
- **Naming Convention**: Handlers follow `[Method][Resource]` (e.g., `PostClassification`).
- **Code Style**: Adhere to PEP 8; keep line lengths under 80 characters where possible.
- **Architecture**: Ensure a strict separation between the infrastructure and service layers.
