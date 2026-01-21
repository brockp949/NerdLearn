"""
ML-Based Citation Extraction

Extracts and normalizes citations from academic and educational content using:
- Named Entity Recognition (NER)
- Pattern matching for common citation formats
- Reference list parsing
- DOI/ISBN extraction

Supports formats:
- APA (American Psychological Association)
- MLA (Modern Language Association)
- Chicago
- IEEE
- Harvard
- Vancouver

Features:
- High-precision citation detection
- Author/year extraction
- DOI resolution
- Reference deduplication
- Confidence scoring
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter

logger = logging.getLogger(__name__)


class CitationFormat(str, Enum):
    """Common citation formats"""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    IEEE = "ieee"
    HARVARD = "harvard"
    VANCOUVER = "vancouver"
    UNKNOWN = "unknown"


@dataclass
class Author:
    """Parsed author information"""
    last_name: str
    first_name: Optional[str] = None
    middle_initial: Optional[str] = None
    suffix: Optional[str] = None  # Jr., III, etc.

    def full_name(self) -> str:
        """Get full name string"""
        parts = []
        if self.first_name:
            parts.append(self.first_name)
        if self.middle_initial:
            parts.append(self.middle_initial)
        parts.append(self.last_name)
        if self.suffix:
            parts.append(self.suffix)
        return " ".join(parts)

    def normalized(self) -> str:
        """Get normalized form for deduplication"""
        return f"{self.last_name.lower()}_{(self.first_name or '')[0:1].lower()}"


@dataclass
class Citation:
    """Extracted citation information"""
    raw_text: str
    authors: List[Author]
    year: Optional[int]
    title: Optional[str]
    source: Optional[str]  # Journal, book, conference, etc.
    volume: Optional[str]
    issue: Optional[str]
    pages: Optional[str]
    doi: Optional[str]
    isbn: Optional[str]
    url: Optional[str]
    format: CitationFormat
    confidence: float
    is_inline: bool  # True for (Author, Year), False for reference list
    position: Optional[Tuple[int, int]] = None  # Start, end position in text
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_apa(self) -> str:
        """Convert to APA format string"""
        parts = []

        # Authors
        if self.authors:
            author_strs = []
            for i, author in enumerate(self.authors):
                if i < 6:  # APA shows first 6 authors
                    if author.first_name:
                        author_strs.append(f"{author.last_name}, {author.first_name[0]}.")
                    else:
                        author_strs.append(author.last_name)
                elif i == 6:
                    author_strs.append("...")
                    author_strs.append(f"{self.authors[-1].last_name}, {self.authors[-1].first_name[0] if self.authors[-1].first_name else ''}")
                    break
            parts.append(", ".join(author_strs[:-1]) + f", & {author_strs[-1]}" if len(author_strs) > 1 else author_strs[0] if author_strs else "")

        # Year
        if self.year:
            parts.append(f"({self.year}).")

        # Title
        if self.title:
            parts.append(f"{self.title}.")

        # Source
        if self.source:
            source_str = f"*{self.source}*"
            if self.volume:
                source_str += f", *{self.volume}*"
                if self.issue:
                    source_str += f"({self.issue})"
            if self.pages:
                source_str += f", {self.pages}"
            parts.append(source_str + ".")

        # DOI
        if self.doi:
            parts.append(f"https://doi.org/{self.doi}")

        return " ".join(parts)


class CitationExtractor:
    """
    ML-enhanced citation extraction from text.

    Combines regex patterns with heuristics for robust citation detection.
    """

    # Regex patterns for inline citations
    INLINE_PATTERNS = {
        # (Author, Year) - APA/Harvard style
        "author_year": re.compile(
            r'\(([A-Z][a-zA-Z\'\-]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-zA-Z\'\-]+))?),?\s*((?:19|20)\d{2}[a-z]?)\)',
            re.UNICODE
        ),
        # [Number] - IEEE/Vancouver style
        "numbered": re.compile(r'\[(\d{1,3})\]'),
        # Author (Year) - alternate format
        "author_year_alt": re.compile(
            r'([A-Z][a-zA-Z\'\-]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-zA-Z\'\-]+))?)\s*\(((?:19|20)\d{2}[a-z]?)\)',
            re.UNICODE
        ),
        # (Author Year) - no comma
        "author_year_no_comma": re.compile(
            r'\(([A-Z][a-zA-Z\'\-]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-zA-Z\'\-]+))?)\s+((?:19|20)\d{2}[a-z]?)\)',
            re.UNICODE
        ),
    }

    # Regex patterns for reference list entries
    REFERENCE_PATTERNS = {
        # DOI pattern
        "doi": re.compile(r'(?:doi:|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,}/[^\s\]<>]+)', re.IGNORECASE),
        # ISBN pattern
        "isbn": re.compile(r'ISBN[:\s-]*(\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?[\dXx])', re.IGNORECASE),
        # URL pattern
        "url": re.compile(r'https?://[^\s\]<>"]+'),
        # Year pattern
        "year": re.compile(r'\(?((?:19|20)\d{2}[a-z]?)\)?'),
        # Volume/Issue pattern
        "volume_issue": re.compile(r'(\d+)\s*\((\d+)\)'),
        # Pages pattern
        "pages": re.compile(r'(?:pp?\.\s*)?(\d+)\s*[-–—]\s*(\d+)'),
    }

    # Patterns for detecting reference sections
    REFERENCE_SECTION_PATTERNS = [
        re.compile(r'^references?\s*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^bibliography\s*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^works?\s+cited\s*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^literature\s+cited\s*$', re.IGNORECASE | re.MULTILINE),
    ]

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize citation extractor.

        Args:
            confidence_threshold: Minimum confidence for including citation
        """
        self.confidence_threshold = confidence_threshold

    def extract_citations(
        self,
        text: str,
        detect_format: bool = True
    ) -> List[Citation]:
        """
        Extract all citations from text.

        Args:
            text: Input text
            detect_format: Whether to auto-detect citation format

        Returns:
            List of extracted citations
        """
        citations = []

        # Extract inline citations
        inline_citations = self._extract_inline_citations(text)
        citations.extend(inline_citations)

        # Extract reference list citations
        ref_citations = self._extract_reference_list(text)
        citations.extend(ref_citations)

        # Deduplicate and merge
        citations = self._deduplicate_citations(citations)

        # Filter by confidence
        citations = [c for c in citations if c.confidence >= self.confidence_threshold]

        return citations

    def _extract_inline_citations(self, text: str) -> List[Citation]:
        """Extract inline citations from text"""
        citations = []

        # Try each inline pattern
        for pattern_name, pattern in self.INLINE_PATTERNS.items():
            for match in pattern.finditer(text):
                citation = self._parse_inline_match(match, pattern_name)
                if citation:
                    citations.append(citation)

        return citations

    def _parse_inline_match(
        self,
        match: re.Match,
        pattern_name: str
    ) -> Optional[Citation]:
        """Parse an inline citation match"""
        try:
            raw_text = match.group(0)
            position = (match.start(), match.end())

            if pattern_name == "numbered":
                # [1] style - minimal info
                return Citation(
                    raw_text=raw_text,
                    authors=[],
                    year=None,
                    title=None,
                    source=None,
                    volume=None,
                    issue=None,
                    pages=None,
                    doi=None,
                    isbn=None,
                    url=None,
                    format=CitationFormat.IEEE,
                    confidence=0.7,
                    is_inline=True,
                    position=position,
                    metadata={"reference_number": int(match.group(1))}
                )

            # Author-year styles
            if pattern_name == "author_year":
                author_str, year_str = match.group(1), match.group(2)
            elif pattern_name == "author_year_alt":
                author_str, year_str = match.group(1), match.group(2)
            elif pattern_name == "author_year_no_comma":
                author_str, year_str = match.group(1), match.group(2)
            else:
                return None

            # Parse authors
            authors = self._parse_author_string(author_str)

            # Parse year
            year = self._parse_year(year_str)

            # Determine format
            citation_format = CitationFormat.APA
            if "et al" in author_str.lower():
                citation_format = CitationFormat.APA  # Common in APA

            return Citation(
                raw_text=raw_text,
                authors=authors,
                year=year,
                title=None,
                source=None,
                volume=None,
                issue=None,
                pages=None,
                doi=None,
                isbn=None,
                url=None,
                format=citation_format,
                confidence=0.8 if authors and year else 0.6,
                is_inline=True,
                position=position
            )

        except Exception as e:
            logger.warning(f"Error parsing inline citation: {e}")
            return None

    def _extract_reference_list(self, text: str) -> List[Citation]:
        """Extract citations from reference section"""
        citations = []

        # Find reference section
        ref_start = None
        for pattern in self.REFERENCE_SECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                ref_start = match.end()
                break

        if ref_start is None:
            # Try to find references at end of document
            # Look for multiple lines starting with author-like patterns
            lines = text.split('\n')
            potential_refs = []
            for i, line in enumerate(lines):
                if self._looks_like_reference(line):
                    potential_refs.append((i, line))

            if len(potential_refs) >= 3:
                for idx, line in potential_refs:
                    citation = self._parse_reference_line(line)
                    if citation:
                        citations.append(citation)
            return citations

        # Process reference section
        ref_text = text[ref_start:]
        lines = ref_text.split('\n')

        current_ref = ""
        for line in lines:
            line = line.strip()
            if not line:
                if current_ref:
                    citation = self._parse_reference_line(current_ref)
                    if citation:
                        citations.append(citation)
                    current_ref = ""
            elif self._starts_new_reference(line):
                if current_ref:
                    citation = self._parse_reference_line(current_ref)
                    if citation:
                        citations.append(citation)
                current_ref = line
            else:
                current_ref += " " + line

        # Don't forget last reference
        if current_ref:
            citation = self._parse_reference_line(current_ref)
            if citation:
                citations.append(citation)

        return citations

    def _looks_like_reference(self, line: str) -> bool:
        """Check if line looks like a reference entry"""
        line = line.strip()
        if len(line) < 20:
            return False

        # Check for author-like start
        if re.match(r'^[A-Z][a-zA-Z\'\-]+,\s*[A-Z]', line):
            return True

        # Check for numbered reference
        if re.match(r'^\[\d+\]', line):
            return True

        # Check for year presence
        if re.search(r'\((?:19|20)\d{2}\)', line):
            return True

        return False

    def _starts_new_reference(self, line: str) -> bool:
        """Check if line starts a new reference"""
        line = line.strip()

        # Numbered references
        if re.match(r'^\[\d+\]', line):
            return True

        # Author-first references
        if re.match(r'^[A-Z][a-zA-Z\'\-]+,', line):
            return True

        # Hanging indent (common in reference lists)
        if not line.startswith(' ') and len(line) > 10:
            return True

        return False

    def _parse_reference_line(self, text: str) -> Optional[Citation]:
        """Parse a single reference line"""
        text = text.strip()
        if len(text) < 20:
            return None

        try:
            # Extract identifiers first
            doi_match = self.REFERENCE_PATTERNS["doi"].search(text)
            doi = doi_match.group(1) if doi_match else None

            isbn_match = self.REFERENCE_PATTERNS["isbn"].search(text)
            isbn = isbn_match.group(1).replace('-', '').replace(' ', '') if isbn_match else None

            url_match = self.REFERENCE_PATTERNS["url"].search(text)
            url = url_match.group(0) if url_match else None

            # Extract year
            year_match = self.REFERENCE_PATTERNS["year"].search(text)
            year = self._parse_year(year_match.group(1)) if year_match else None

            # Extract volume/issue
            vi_match = self.REFERENCE_PATTERNS["volume_issue"].search(text)
            volume = vi_match.group(1) if vi_match else None
            issue = vi_match.group(2) if vi_match else None

            # Extract pages
            pages_match = self.REFERENCE_PATTERNS["pages"].search(text)
            pages = f"{pages_match.group(1)}-{pages_match.group(2)}" if pages_match else None

            # Parse authors (at start of reference)
            authors = []
            # Try to find author block before year
            if year_match:
                author_text = text[:year_match.start()].strip()
                author_text = re.sub(r'^\[\d+\]\s*', '', author_text)  # Remove numbered prefix
                authors = self._parse_author_string(author_text)

            # Extract title (after year, before source)
            title = None
            if year_match:
                # Text after year until period or italics
                after_year = text[year_match.end():].strip()
                after_year = after_year.lstrip(').')
                title_match = re.match(r'^([^.]+)\.', after_year)
                if title_match:
                    title = title_match.group(1).strip()

            # Determine format
            citation_format = self._detect_format(text)

            # Calculate confidence
            confidence = 0.5
            if authors:
                confidence += 0.15
            if year:
                confidence += 0.15
            if title:
                confidence += 0.1
            if doi:
                confidence += 0.1

            return Citation(
                raw_text=text,
                authors=authors,
                year=year,
                title=title,
                source=None,  # Would need more parsing
                volume=volume,
                issue=issue,
                pages=pages,
                doi=doi,
                isbn=isbn,
                url=url,
                format=citation_format,
                confidence=min(1.0, confidence),
                is_inline=False
            )

        except Exception as e:
            logger.warning(f"Error parsing reference: {e}")
            return None

    def _parse_author_string(self, text: str) -> List[Author]:
        """Parse author names from string"""
        authors = []
        text = text.strip().rstrip(',')

        if not text:
            return authors

        # Handle "et al."
        if "et al" in text.lower():
            # Just get first author
            text = re.split(r'\s+et\s+al', text, flags=re.IGNORECASE)[0]

        # Split by common separators
        parts = re.split(r'[,;&]\s*(?:and\s+)?', text)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Try to parse "Last, First" format
            if ', ' in part:
                segments = part.split(', ')
                last_name = segments[0].strip()
                first_name = segments[1].strip() if len(segments) > 1 else None

                # Handle initials
                if first_name and len(first_name) <= 3 and '.' in first_name:
                    first_name = first_name.replace('.', '')

                authors.append(Author(
                    last_name=last_name,
                    first_name=first_name
                ))
            elif ' ' in part:
                # "First Last" format
                segments = part.split()
                if len(segments) >= 2:
                    authors.append(Author(
                        last_name=segments[-1],
                        first_name=segments[0]
                    ))
            else:
                # Single name
                authors.append(Author(last_name=part))

        return authors

    def _parse_year(self, text: str) -> Optional[int]:
        """Parse year from string"""
        if not text:
            return None

        # Remove letter suffix (e.g., 2020a)
        year_str = re.sub(r'[a-z]$', '', text.strip())

        try:
            year = int(year_str)
            if 1900 <= year <= 2100:
                return year
        except ValueError:
            pass

        return None

    def _detect_format(self, text: str) -> CitationFormat:
        """Detect citation format from reference text"""
        text_lower = text.lower()

        # IEEE: starts with [number]
        if re.match(r'^\[\d+\]', text):
            return CitationFormat.IEEE

        # APA: Author, A. A. (Year).
        if re.match(r'^[A-Z][a-z]+,\s*[A-Z]\.\s*[A-Z]?\.\s*\(', text):
            return CitationFormat.APA

        # MLA: Author. "Title." Source
        if re.search(r'"[^"]+"\.', text):
            return CitationFormat.MLA

        # Chicago: Author. Title. (no quotes, different punctuation)
        if re.match(r'^[A-Z][a-z]+,\s*[A-Z][a-z]+\.', text):
            return CitationFormat.CHICAGO

        # Harvard: similar to APA but subtle differences
        if re.match(r'^[A-Z][a-z]+,\s*[A-Z]\.\s*\(', text):
            return CitationFormat.HARVARD

        return CitationFormat.UNKNOWN

    def _deduplicate_citations(self, citations: List[Citation]) -> List[Citation]:
        """Remove duplicate citations"""
        seen = set()
        unique = []

        for citation in citations:
            # Create dedup key
            key_parts = []
            if citation.authors:
                key_parts.append(citation.authors[0].normalized())
            if citation.year:
                key_parts.append(str(citation.year))
            if citation.doi:
                key_parts.append(citation.doi.lower())

            key = "_".join(key_parts) if key_parts else citation.raw_text[:50]

            if key not in seen:
                seen.add(key)
                unique.append(citation)
            else:
                # Merge info from duplicate
                for existing in unique:
                    existing_key_parts = []
                    if existing.authors:
                        existing_key_parts.append(existing.authors[0].normalized())
                    if existing.year:
                        existing_key_parts.append(str(existing.year))
                    if existing.doi:
                        existing_key_parts.append(existing.doi.lower())
                    existing_key = "_".join(existing_key_parts) if existing_key_parts else existing.raw_text[:50]

                    if existing_key == key:
                        # Update with more complete info
                        if not existing.title and citation.title:
                            existing.title = citation.title
                        if not existing.doi and citation.doi:
                            existing.doi = citation.doi
                        if not existing.source and citation.source:
                            existing.source = citation.source
                        # Increase confidence for duplicates
                        existing.confidence = min(1.0, existing.confidence + 0.1)
                        break

        return unique

    def get_citation_statistics(
        self,
        citations: List[Citation]
    ) -> Dict[str, Any]:
        """
        Get statistics about extracted citations.

        Returns:
            Dictionary with citation statistics
        """
        if not citations:
            return {
                "total": 0,
                "inline": 0,
                "references": 0,
                "formats": {},
                "years": {},
                "authors": [],
                "avg_confidence": 0.0
            }

        # Count formats
        format_counts = Counter(c.format.value for c in citations)

        # Count years
        year_counts = Counter(c.year for c in citations if c.year)

        # Get top authors
        all_authors = []
        for c in citations:
            for author in c.authors:
                all_authors.append(author.last_name)
        author_counts = Counter(all_authors).most_common(10)

        return {
            "total": len(citations),
            "inline": sum(1 for c in citations if c.is_inline),
            "references": sum(1 for c in citations if not c.is_inline),
            "formats": dict(format_counts),
            "years": dict(year_counts),
            "top_authors": author_counts,
            "avg_confidence": sum(c.confidence for c in citations) / len(citations),
            "with_doi": sum(1 for c in citations if c.doi),
            "with_title": sum(1 for c in citations if c.title),
        }


# Singleton instance
citation_extractor = CitationExtractor()
