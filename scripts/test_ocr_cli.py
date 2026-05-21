"""
CLI utility for testing OCR functionality against local or remote endpoints.
Supports both general document processing and specialized COR extraction.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from io import BytesIO
from pathlib import Path

import requests
from fastapi import UploadFile
from starlette.datastructures import Headers

# Ensure project root is in sys.path for local imports
sys.path.append(os.getcwd())

from src.services.ocr import OCRService


def ocr_remote(url: str, file_path: Path) -> dict:
    """
    Sends a file to a running FastAPI server for remote OCR processing.
    """
    endpoint = f"{url.rstrip('/')}/api/v1/ocr"

    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/octet-stream")}
            response = requests.post(endpoint, files=files)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"error": str(e)}


async def run_local_extraction(file_path: Path, is_cor: bool) -> dict:
    """
    Executes OCR logic using local services without network overhead.
    """
    service = OCRService()

    with open(file_path, "rb") as f:
        content = f.read()

        # Determine content type for the UploadFile mock
        ext = file_path.suffix.lower()
        mime = "application/pdf" if ext == ".pdf" else "image/png"

        upload = UploadFile(
            filename=file_path.name,
            file=BytesIO(content),
            size=len(content),
            headers=Headers({"content-type": mime}),
        )

        if is_cor:
            result = await service.process_cor(upload)
        else:
            result = await service.process_document(upload)

        return result.dict()


def ocr_local(file_path: Path, is_cor: bool = False) -> dict:
    """
    Synchronous wrapper for local OCR execution.
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            run_local_extraction(file_path, is_cor)
        )
        loop.close()
        return result
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


def handle_output(result: dict, file_path: Path):
    """
    Displays and optionally persists the OCR extraction results.
    """
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    duration = result.get("processing_time_ms", 0)
    print(f"Success! (Took {duration:.2f}ms)")
    print("-" * 30)
    print(f"FULL TEXT:\n{result.get('full_text', '')}")
    print("-" * 30)

    if input("Save full JSON to outputs/? (y/n): ").lower().strip() == "y":
        os.makedirs("./outputs", exist_ok=True)
        ts = int(time.time())
        out_path = f"./outputs/{file_path.name}-{ts}.json"

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {out_path}")


def run_interactive(args):
    """
    Continuous loop for processing multiple files without restarting the CLI.
    """
    print("=== GuiSIS AI OCR Interactive Mode ===")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        raw_path = input("\nEnter file path: ").strip().replace('"', "")
        if raw_path.lower() in ["exit", "quit"]:
            break

        if not raw_path:
            continue

        file_path = Path(raw_path)
        if not file_path.exists():
            print(f"Error: File not found at {raw_path}")
            continue

        print(f"Processing {file_path.name}...")
        if args.local:
            result = ocr_local(file_path, is_cor=args.cor)
        else:
            result = ocr_remote(args.url, file_path)

        handle_output(result, file_path)


def main():
    """
    Parsing logic and orchestration for single-file or interactive runs.
    """
    parser = argparse.ArgumentParser(description="Ogos AI OCR Testing CLI")
    parser.add_argument("--file", type=str, help="Path to image or PDF file")
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000", help="Remote API base URL"
    )
    parser.add_argument("--local", action="store_true", help="Local processing")
    parser.add_argument("--cor", action="store_true", help="Extract COR data")
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive mode"
    )

    args = parser.parse_args()

    if args.interactive:
        run_interactive(args)
        return

    if not args.file:
        parser.print_help()
        return

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found at {args.file}")
        return

    if args.local:
        result = ocr_local(file_path, is_cor=args.cor)
    else:
        result = ocr_remote(args.url, file_path)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
