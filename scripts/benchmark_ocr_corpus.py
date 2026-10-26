"""
Benchmark the generated N = 200 OCR perturbation corpus through OCRService.
Outputs confusion matrix metrics and exports benchmark_results.csv.
"""

import asyncio
import csv
import io
import os
import sys
import time
from pathlib import Path

from fastapi import UploadFile
from starlette.datastructures import Headers

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.ocr import OCRService

CORPUS_DIR = PROJECT_ROOT / "test_corpus"
RESULTS_CSV = CORPUS_DIR / "benchmark_results.csv"


def create_mock_upload(filepath: Path) -> UploadFile:
    content = filepath.read_bytes()
    mime = "image/png" if filepath.suffix == ".png" else "application/pdf"
    return UploadFile(
        filename=filepath.name,
        file=io.BytesIO(content),
        size=len(content),
        headers=Headers({"content-type": mime}),
    )


async def evaluate_file(service: OCRService, filepath: Path, expected: str):
    upload = create_mock_upload(filepath)
    t0 = time.perf_counter()
    status = "REJECTED"
    reason = ""

    try:
        val = await service.validate_document(upload)
        await upload.seek(0)
        if val.is_valid:
            cor = await service.process_cor(upload)
            status = "ACCEPTED"
            reason = f"Parsed ID: {cor.student_number}"
        else:
            status = "REJECTED"
            reason = val.message
    except Exception as exc:
        status = "REJECTED"
        reason = str(exc).splitlines()[0][:50]

    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "filename": filepath.name,
        "expected": expected,
        "observed": status,
        "latency_ms": round(latency_ms, 2),
        "reason": reason,
    }


async def run_benchmark():
    if not CORPUS_DIR.exists():
        print(f"[Error] Corpus not found at {CORPUS_DIR}. Run generate_ocr_corpus.py first.")
        return

    service = OCRService()
    valid_files = sorted((CORPUS_DIR / "legitimate").glob("*.png"))
    invalid_files = sorted((CORPUS_DIR / "defective").glob("*.png"))

    print(f"[*] Found {len(valid_files)} legitimate and {len(invalid_files)} defective files.")
    print("[*] Running benchmark through OCRService...")

    results = []
    # Evaluate legitimate files
    for f in valid_files:
        res = await evaluate_file(service, f, expected="ACCEPTED")
        results.append(res)

    # Evaluate defective files
    for f in invalid_files:
        res = await evaluate_file(service, f, expected="REJECTED")
        results.append(res)

    # Confusion matrix calculations
    tp = sum(1 for r in results if r["expected"] == "ACCEPTED" and r["observed"] == "ACCEPTED")
    fn = sum(1 for r in results if r["expected"] == "ACCEPTED" and r["observed"] == "REJECTED")
    tn = sum(1 for r in results if r["expected"] == "REJECTED" and r["observed"] == "REJECTED")
    fp = sum(1 for r in results if r["expected"] == "REJECTED" and r["observed"] == "ACCEPTED")

    total = len(results)
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

    # Save to CSV
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "expected", "observed", "latency_ms", "reason"])
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print(f"  OCR BENCHMARK EVALUATION REPORT (N = {total})")
    print("=" * 60)
    print(f"  True Positives (TP) : {tp:<4} | False Negatives (FN) : {fn:<4}")
    print(f"  True Negatives (TN) : {tn:<4} | False Positives (FP) : {fp:<4}")
    print("-" * 60)
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec * 100:.2f}%")
    print(f"  F1-Score  : {f1 * 100:.2f}%")
    print(f"  Results saved to : {RESULTS_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_benchmark())

