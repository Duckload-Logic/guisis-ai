import re

NAME_REDACTION_PATTERN = re.compile(
    r"(?i)((?:my\s+name\s+is|ako\s+(?:po\s+)?si|i\s+am|hi,?\s*i'?m|hello,?\s*i'?m)\s+)"
    r"([A-ZÀ-ÖØ-ÞÑ][A-Za-zÀ-ÖØ-öø-ÿ'’-]*"
    r"(?:\s+(?:[A-ZÀ-ÖØ-ÞÑ][A-Za-zÀ-ÖØ-öø-ÿ'’-]*|"
    r"de\s+la|de\s+los|de\s+las|de|del|dela|van|von|bin|di|da|dos|das)){0,4})"
    r"(?=\b|[\.,;:!?])"
)

def anonymize_text(text: str) -> str:
    """Redact common PII patterns while preserving task-relevant text."""
    clean = re.sub(
        r"\b\d{4}-\d{5}-TG-\d\S*\b",
        "[STUDENT_NUMBER_REDACTED]",
        text,
    )

    clean = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "[EMAIL_REDACTED]",
        clean,
    )

    clean = re.sub(
        r"(?i)\b(section)\s+([a-zA-Z0-9\-]*\d[a-zA-Z0-9\-]*)\b",
        r"\1 [REDACTED]",
        clean,
    )

    return NAME_REDACTION_PATTERN.sub(r"\1[NAME_REDACTED]", clean)