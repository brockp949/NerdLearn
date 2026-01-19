"""
PII (Personally Identifiable Information) Filter
Strips potential PII from queries before they are sent to external LLMs.
"""
import re
from typing import List

class PIIFilter:
    """
    Utility to scrub PII from text
    """
    
    # Simple regex patterns for common PII
    PATTERNS = {
        "email": r"[\w\.-]+@[\w\.-]+\.\w+",
        "phone": r"(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}",
        "ssn": r"\d{3}-\d{2}-\d{4}",
        "credit_card": r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}",
    }

    @classmethod
    def scrub(cls, text: str, placeholder: str = "[PII_REMOVED]") -> str:
        """
        Scrub PII from text using regex patterns
        """
        scrubbed_text = text
        for pii_type, pattern in cls.PATTERNS.items():
            scrubbed_text = re.sub(pattern, placeholder, scrubbed_text)
        
        return scrubbed_text

    @classmethod
    def contains_pii(cls, text: str) -> bool:
        """
        Check if text contains any known PII patterns
        """
        for pattern in cls.PATTERNS.values():
            if re.search(pattern, text):
                return True
        return False
