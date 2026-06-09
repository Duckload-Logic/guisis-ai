"""
Service layer for Optical Character Recognition (OCR) using PaddleOCR.
Handles image enhancement, PDF rendering, and document-specific extraction.
"""

import logging
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import cv2
import fitz
import numpy as np
from fastapi import UploadFile

from src.infrastructure.model_loader import ModelLoader
from src.schemas.ocr import CORResponse, OCRPage, OCRResponse, OCRWord, CORValidationResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CORField:
    """
    Configuration for a specific document field to be extracted via OCR.
    """

    name: str
    bbox: List[List[float]]
    pattern: str
    post_process: Optional[Callable[[str], Any]] = None
    flags: re.RegexFlag = re.IGNORECASE


class OCRService:
    """
    Orchestrates OCR processes for general documents and registration forms.
    """

    def __init__(self) -> None:
        """
        Initializes the service and retrieves the singleton OCR engine.
        """
        self.ocr = ModelLoader.load_ocr_engine()

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Standardizes the image for the OCR engine to improve accuracy.

        Args:
            image: Original grayscale or BGR image.

        Returns:
            Processed grayscale image.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Avoid aggressive binarization (cv2.adaptiveThreshold) which turns
        # gradient shading and soft edges into noise and garbled characters.
        # Grayscale alone is cleaner and preserves anti-aliased font strokes.
        return gray

    def _process_image(
        self, img: np.ndarray, page_num: int
    ) -> Optional[OCRPage]:
        """
        Internal detection and recognition logic for a single image.

        Args:
            img: The image array to process.
            page_num: Human-readable page index.

        Returns:
            Structured page data or None if no content found.
        """
        enhanced = self._enhance_image(img)
        raw_result = self.ocr.ocr(enhanced)

        # Normalize various PaddleOCR output formats (None, nested list, tuple)
        if raw_result is None:
            logger.warning(f"Page {page_num}: OCR returned None")
            return None

        if isinstance(raw_result, tuple):
            if not raw_result:
                return None
            raw_result = raw_result[0]

        if (
            isinstance(raw_result, list) and
            raw_result and
            isinstance(raw_result[0], list)
        ):
            raw_result = raw_result[0]

        if not isinstance(raw_result, list) or not raw_result:
            return None

        page_text = []
        words = []

        for item in raw_result:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue

            bbox, text_conf = item[0], item[1]

            # Reconstruct text and confidence safely
            if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                text, conf = str(text_conf[0]), float(text_conf[1])
            elif isinstance(text_conf, str):
                text, conf = text_conf, 1.0
            else:
                continue

            page_text.append(text)
            words.append(OCRWord(text=text, confidence=conf, bounding_box=bbox))

        if not words:
            return None

        return OCRPage(
            page_number=page_num,
            text=" ".join(page_text),
            words=words
        )

    async def _extract_field_text(
        self, target_bbox: List[List[float]], words: List[OCRWord]
    ) -> Optional[str]:
        """
        Retrieves text fragments that spatially overlap with a target region.
        """
        t_x_min = min(p[0] for p in target_bbox)
        t_y_min = min(p[1] for p in target_bbox)
        t_x_max = max(p[0] for p in target_bbox)
        t_y_max = max(p[1] for p in target_bbox)

        found = []
        for word in words:
            if not word.bounding_box:
                continue

            w_x_min = min(p[0] for p in word.bounding_box)
            w_y_min = min(p[1] for p in word.bounding_box)
            w_x_max = max(p[0] for p in word.bounding_box)
            w_y_max = max(p[1] for p in word.bounding_box)

            # Check overlap using bounding box bounds
            if (w_x_min < t_x_max and w_x_max > t_x_min and
                    w_y_min < t_y_max and w_y_max > t_y_min):
                found.append(word)

        if not found:
            return None

        # Sort by vertical then horizontal position to reconstruct original flow
        found.sort(
            key=lambda x: (
                min(p[1] for p in x.bounding_box) // 15,
                min(p[0] for p in x.bounding_box)
            )
        )

        return " ".join([w.text for w in found])

    async def _apply_field_rules(
        self, field: CORField, words: List[OCRWord]
    ) -> str:
        """
        Validates extracted text against regex patterns and transformations.
        """
        raw = await self._extract_field_text(field.bbox, words)
        if raw is None:
            raise ValueError(f"No text detected in region for {field.name}")

        match = re.search(field.pattern, raw, field.flags)
        if not match:
            raise ValueError(
                f"Pattern {field.pattern} not found for {field.name}. "
                f"Region text: '{raw[:100]}...'"
            )

        val = match.group(1).strip()
        if field.post_process:
            processed = field.post_process(val)
            if processed is None:
                raise ValueError(f"Invalid {field.name} value: '{val}'")

            val = processed

        return val

    def _pdf_to_images(
        self,
        content: bytes,
        max_pages: Optional[int] = None,
        dpi: int = 300
    ) -> List[np.ndarray]:
        """
        High-fidelity rendering of PDF pages into image arrays.
        """
        images = []
        with fitz.open(stream=content, filetype="pdf") as doc:
            for idx, page in enumerate(doc):
                if max_pages is not None and idx >= max_pages:
                    break
                # Custom DPI scale (e.g., 300 for COR, 150 for verification)
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)

                # Convert buffer into standard BGR format
                cv_mode = (
                    cv2.COLOR_RGBA2BGR
                    if pix.alpha else
                    cv2.COLOR_RGB2BGR
                )

                img = np.frombuffer(
                    pix.samples, dtype=np.uint8
                ).reshape(
                    pix.h, pix.w, 4 if pix.alpha else 3
                )
                images.append(cv2.cvtColor(img, cv_mode))

        return images

    def _extract_digital_pdf_text(
        self, content: bytes
    ) -> Optional[List[OCRPage]]:
        """
        Attempts to extract text directly from a digital PDF.
        Returns a list of OCRPages if successful, or None if scanned.
        """
        pages_data = []
        try:
            with fitz.open(stream=content, filetype="pdf") as doc:
                scale = 300.0 / 72.0
                for i, page in enumerate(doc):
                    words_raw = page.get_text("words")
                    # If any page is scanned/empty, fallback to standard OCR
                    if len(words_raw) < 15:
                        return None

                    page_text = []
                    words = []
                    for w in words_raw:
                        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
                        bbox = [
                            [x0 * scale, y0 * scale],
                            [x1 * scale, y0 * scale],
                            [x1 * scale, y1 * scale],
                            [x0 * scale, y1 * scale]
                        ]
                        page_text.append(text)
                        words.append(
                            OCRWord(
                                text=text,
                                confidence=1.0,
                                bounding_box=bbox
                            )
                        )

                    pages_data.append(
                        OCRPage(
                            page_number=i + 1,
                            text=" ".join(page_text),
                            words=words
                        )
                    )
            return pages_data
        except Exception as e:
            logger.warning(
                f"Digital PDF extraction failed, falling back: {e}"
            )
            return None

    async def process_document(
        self,
        file: UploadFile,
        max_pages: Optional[int] = None,
        dpi: int = 300
    ) -> OCRResponse:
        """
        Main entry point for generic document text extraction.
        """
        start = time.time()
        try:
            content = await file.read()
            pages_data = []
            is_pdf = file.filename.lower().endswith(".pdf")
            engine_name = "PaddleOCR v2.7.3"

            if is_pdf:
                # Try digital PDF fast extraction first!
                digital_pages = self._extract_digital_pdf_text(content)
                if digital_pages:
                    if max_pages is not None:
                        pages_data = digital_pages[:max_pages]
                    else:
                        pages_data = digital_pages
                    engine_name = "PyMuPDF Fast Extraction"
                else:
                    imgs = self._pdf_to_images(
                        content, max_pages=max_pages, dpi=dpi
                    )
                    for i, img in enumerate(imgs):
                        stats = self._process_image(img, i + 1)
                        if stats:
                            pages_data.append(stats)
            else:
                buf = np.frombuffer(content, np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Failed to decode uploaded image")

                # Downscale high-resolution images for validation speed
                if dpi < 300:
                    h, w = img.shape[:2]
                    if w > 1500 or h > 1500:
                        scale = dpi / 300.0
                        img = cv2.resize(
                            img,
                            (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA
                        )

                stats = self._process_image(img, 1)
                if stats:
                    pages_data.append(stats)

            return OCRResponse(
                filename=file.filename,
                total_pages=len(pages_data),
                full_text="\n".join([p.text for p in pages_data]),
                pages=pages_data,
                processing_time_ms=(time.time() - start) * 1000,
                metadata={
                    "engine": engine_name,
                    "format": "pdf" if is_pdf else "image"
                }
            )
        except Exception:
            logger.error(f"OCR failed: {traceback.format_exc()}")
            raise

    async def process_cor(self, file: UploadFile) -> CORResponse:
        """
        Specialized pipeline for extracting student data from COR PDFs.
        """
        # Coordinate definitions (Assumes 300 DPI standardized layout)
        b_name = [
            [79.0, 412.0], [1500.0, 412.0], [1500.0, 500.0], [79.0, 500.0]
        ]
        b_no = [
            [76.0, 461.0], [440.0, 461.0], [440.0, 506.0], [76.0, 506.0]
        ]
        b_ay = [
            [2341.0, 461.0], [2451.0, 461.0], [2451.0, 509.0], [2341.0, 509.0]
        ]
        b_tm = [
            [2668.0, 457.0], [3149.0, 464.0], [3148.0, 510.0], [2667.0, 502.0]
        ]
        b_pg = [
            [89.0, 550.0], [1677.0, 550.0], [1677.0, 620.0], [89.0, 620.0]
        ]
        b_pc = [
            [2654.0, 550.0], [3114.0, 550.0], [3114.0, 620.0], [2654.0, 620.0]
        ]
        b_pd = [
            [70.0, 640.0], [3150.0, 640.0], [3150.0, 720.0], [70.0, 720.0]
        ]

        fields = [
            CORField("full_name", b_name, r"([A-Za-zÀ-ÿ\s,.\-]+)"),
            CORField(
                "student_number", b_no,
                r"(\d{4}-\d{5}-[A-Z]{2,3}-[01]\S*)"
            ),
            CORField("academic_year", b_ay, r"(\d{2,4}-?\d{2,4})"),
            CORField(
                "term", b_tm, r"TERM:\s*(First|Second)\s*Semester",
                post_process=lambda x: {"first": 1, "second": 2}.get(x.lower())
            ),
            CORField(
                "program_description", b_pg,
                r"PROGRAM DESCRIPTION:\s*(.*?)\s*\("
            ),
            CORField("program_code", b_pc, r"PROGRAM CODE:\s*([A-Z0-9]+)"),
            CORField(
                "year_level", b_pd,
                r"YEAR LEVEL:\s*(First|Second|Third|Fourth)\sYear",
                post_process=lambda x: {
                    "first": 1, "second": 2, "third": 3, "fourth": 4
                }.get(x.lower())
            ),
            CORField("campus", b_pd, r"Campus:\s*([A-Za-z\s-]+)"),
            CORField("section", b_pd, r"SECTION:\s*(\d+)"),
        ]

        ocr: OCRResponse = await self.process_document(file)
        if not ocr.pages:
            raise ValueError("Document contains no readable pages")

        words = ocr.pages[0].words
        # Reading order sort: y (line) then x (column)
        words.sort(
            key=lambda x: (
                min(p[1] for p in x.bounding_box) // 10,
                min(p[0] for p in x.bounding_box)
            )
        )

        ext = {}
        for f in fields:
            ext[f.name] = await self._apply_field_rules(f, words)

        # Reconstruct name for directory-style records
        full = ext["full_name"]
        last = full.split(",")[0].strip() if "," in full else full.split()[-1]

        # Year handling for both full (2023-2024) and short (2526) formats
        ay_r = ext["academic_year"]
        if "-" in ay_r:
            s_ay, e_ay = ay_r.split("-")
        elif len(ay_r) == 4:
            # 2526 -> 2025-2026
            s_ay, e_ay = "20" + ay_r[:2], "20" + ay_r[2:]
        else:
            s_ay, e_ay = ay_r, ay_r

        return CORResponse(
            filename=ocr.filename,
            last_name=last.upper(),
            full_name=full,
            student_number=ext["student_number"],
            start_academic_year=s_ay,
            end_academic_year=e_ay,
            term=int(ext["term"]),
            program_desc=ext["program_description"],
            program_code=ext["program_code"],
            year_level=int(ext["year_level"]),
            campus=ext["campus"],
            section=int(ext["section"]),
        )

    async def process_bulk_documents(
        self, files: List[UploadFile]
    ) -> List[OCRResponse]:
        """
        Sequentially processes multiple uploaded documents.
        """
        results = []
        for file in files:
            results.append(await self.process_document(file))

        return results

    async def validate_document(
        self, file: UploadFile
    ) -> CORValidationResponse:
        """
        Validates the document structure, confidence, and content.
        """
        ocr: OCRResponse = await self.process_document(
            file, max_pages=1, dpi=150
        )

        # Log basic metadata instead of dumping the entire OCR response
        logger.info(
            f"Validated document '{ocr.filename}' - "
            f"pages: {ocr.total_pages}, "
            f"time: {ocr.processing_time_ms:.2f}ms"
        )

        if not ocr.pages:
            return CORValidationResponse(
                is_valid=False,
                message="Document contains no readable pages."
            )

        for page in ocr.pages:
            # Filter words with high confidence and containing letters
            # (ignoring lone noise characters like '!', '1', etc.)
            valid_words = [
                w for w in page.words
                if w.confidence >= 0.5 and any(c.isalpha() for c in w.text)
            ]

            # A valid document must contain a minimum word count.
            # 5 words is a very conservative threshold to allow short letters,
            # but reject portraits or blank noise yielding 0-3 garbage words.
            if len(valid_words) < 5:
                return CORValidationResponse(
                    is_valid=False,
                    message=(
                        "No valid text detected. The image might be blurry, "
                        "too dark, or not a document."
                    )
                )

            # Check average confidence of identified valid words
            total_conf = sum(w.confidence for w in valid_words)
            avg_conf = total_conf / len(valid_words)
            if avg_conf < 0.60:
                return CORValidationResponse(
                    is_valid=False,
                    message=(
                        "Text clarity is too low. Please upload a clearer "
                        "copy."
                    )
                )

        return CORValidationResponse(
            is_valid=True,
            message="Document validated successfully"
        )