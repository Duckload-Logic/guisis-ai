FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies needed by OCR at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install production python dependencies only.
COPY requirements.prod.txt /app/requirements.prod.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.prod.txt

# Copy project
COPY . /app/

# Expose port (local container port)
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
