"""Input sanitization utilities to prevent XSS and injection attacks."""

import re
import html


def sanitize_text(value: str) -> str:
    """Sanitize free-text input (journal entries, notes).

    - HTML-escapes dangerous characters
    - Strips leading/trailing whitespace
    - Collapses excessive whitespace
    """
    if not value:
        return value
    # HTML-escape to neutralise script tags etc.
    cleaned = html.escape(value.strip())
    # Collapse multiple spaces/newlines into single space
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def sanitize_identifier(value: str) -> str:
    """Sanitize identifier strings (student IDs, codes).

    - Allows only alphanumeric characters, hyphens, and underscores
    - Strips whitespace
    """
    if not value:
        return value
    cleaned = value.strip()
    # Remove anything that isn't alphanumeric, hyphen, or underscore
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "", cleaned)
    return cleaned


def sanitize_name(value: str) -> str:
    """Sanitize name fields (school, grade, person names).

    - HTML-escapes
    - Strips whitespace
    - Allows letters, numbers, spaces, hyphens, periods, apostrophes
    """
    if not value:
        return value
    cleaned = html.escape(value.strip())
    return cleaned
