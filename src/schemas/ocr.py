"""
Schemas for Optical Character Recognition (OCR) results and document metadata.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class OCRWord(BaseModel):
    """
    Metadata for a single detected word/line.
    """

    text: str
    confidence: float
    bounding_box: Optional[List[List[float]]] = Field(
        None,
        description=(
            "Coordinates of the word polygon "
            "[[x,y], [x,y], [x,y], [x,y]]"
        )
    )


class OCRPage(BaseModel):
    """
    A single page of OCR output.
    """

    page_number: int
    text: str
    words: List[OCRWord]
    metadata: Dict = Field(default_factory=dict)


class OCRResponse(BaseModel):
    """
    The full result of a multi-page document OCR process.
    """

    filename: str
    total_pages: int
    full_text: str
    pages: List[OCRPage]
    processing_time_ms: float
    metadata: Dict = Field(default_factory=dict)

class CORResponse(BaseModel):
    """
    Extracted registration fields for Certificate of Registration (COR) files.
    """

    filename: str
    last_name: str
    full_name: str
    student_number: str
    start_academic_year: str
    end_academic_year: str
    term: int
    program_desc: str
    program_code: str
    year_level: int
    campus: str
    section: int
