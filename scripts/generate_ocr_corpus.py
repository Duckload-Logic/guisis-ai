"""
Generate a reproducible N = 200 OCR perturbation corpus from the authentic
institutional template for compliance with RA 10173 (Data Privacy Act of 2012).
"""

import io
import os
import random
from pathlib import Path
import fitz
from PIL import Image, ImageFilter

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

BASE_PDF_CANDIDATES = [
    WORKSPACE_ROOT / "RegistrationCertificate (2).pdf",
    PROJECT_ROOT / "RegistrationCertificate (2).pdf",
    WORKSPACE_ROOT / "COR Test.pdf",
]

OUT_DIR = PROJECT_ROOT / "test_corpus"


def get_base_pdf() -> Path:
    for candidate in BASE_PDF_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate base COR PDF in candidates: {BASE_PDF_CANDIDATES}"
    )


def create_dataset():
    random.seed(42)
    base_pdf = get_base_pdf()
    print(f"[*] Base Authentic Template: {base_pdf}")

    valid_dir = OUT_DIR / "legitimate"
    invalid_dir = OUT_DIR / "defective"
    valid_dir.mkdir(parents=True, exist_ok=True)
    invalid_dir.mkdir(parents=True, exist_ok=True)

    raw_bytes = base_pdf.read_bytes()

    print("[*] Generating 120 Legitimate Perturbations...")
    # 1. 120 Legitimate Perturbations (DPI 100-300, rotation -4 to 4 deg)
    for i in range(120):
        dpi = random.choice([120, 150, 200, 300])
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        pix = doc[0].get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        deg = random.uniform(-4.0, 4.0)
        img = img.rotate(deg, expand=True, fillcolor=(255, 255, 255))
        img.save(valid_dir / f"valid_{i:03d}.png")

    print("[*] Generating 80 Defective Submissions...")
    # 2. 80 Defective Submissions (Blur, extreme skew, blank/corrupted)
    for i in range(80):
        target = invalid_dir / f"invalid_{i:03d}.png"
        if i < 30:  # Severe Optical Camera Blur (Defect)
            pix = fitz.open(stream=raw_bytes, filetype="pdf")[0].get_pixmap(
                dpi=150
            )
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            blur_radius = random.uniform(6.5, 11.0)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            img.save(target)
        elif i < 55:  # Extreme Skew (> 20 deg)
            pix = fitz.open(stream=raw_bytes, filetype="pdf")[0].get_pixmap(
                dpi=150
            )
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            skew_angle = random.choice([-35.0, -25.0, 25.0, 35.0])
            img = img.rotate(
                skew_angle, expand=True, fillcolor=(255, 255, 255)
            )
            img.save(target)
        else:  # Blank Canvas / Corrupted Upload
            img = Image.new("RGB", (800, 1000), color=(255, 255, 255))
            img.save(target)

    print(f"[+] Done! Generated {120 + 80} files under: {OUT_DIR}")


if __name__ == "__main__":
    create_dataset()

