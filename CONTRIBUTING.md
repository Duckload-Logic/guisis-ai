# Contributor's Guide: GuiSIS AI Coding Standards

Welcome to the AI module of the Guidance System. To maintain code cleanliness
and avoid spaghetti or copy-paste engineering, please follow these guidelines.

## 1. Project Structure

The project is structured logically around the following layout:

```
src/
├── api/             # API routes and dependency injection
├── core/            # App configurations (Pydantic Settings), security
├── schemas/         # Request and Response models (Pydantic)
├── services/        # Business logic & machine learning model inferences
└── utils/           # Shared utility and helper functions
```

## 2. Strict Coding Standards

### 80-Character Line Limit
No line of Python code, comments, or documentation should exceed 80
characters. Use parentheses or backslashes `\` to wrap long statements.

### Structured Error Logging
Use the following format for all errors:
`[HandlerName] {Exact Step}: error message`
- Example: `[PostOCRValidate] {OCR Processing}: Tesseract failed to extract`

### Naming Convention
- Endpoint functions MUST follow the pattern: `[HTTPMethod][Resource]`
  - *Correct:* `PostClassify`, `PostOCRCor`
  - *Incorrect:* `handle_ocr`, `ocr_post`
- Use snake_case for other functions, methods, and variables.
- Use PascalCase for classes (e.g. `ClassifierService`, `OCRService`).

### Brutal DRY (Don't Repeat Yourself)
Do not duplicate helper functions, text cleaners, or database-like parsing
blocks across scripts and main codebase files. If you find duplicated code
blocks, extract them into helper functions inside `src/utils/`.

### Configuration Management
All configuration values must be defined in `src/core/config.py` using Pydantic
settings. Do not access `os.getenv` directly inside services or endpoints.
Always use `settings.variable_name`.
