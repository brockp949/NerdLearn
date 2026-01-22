"""
Quality Metrics Service

Calculates and tracks content quality metrics including:
- Transcript/content confidence scores
- Readability metrics (Flesch-Kincaid, SMOG, etc.)
- Content complexity analysis
- Engagement quality indicators
- Accessibility scores

Features:
- Multiple readability formulas
- Language-agnostic complexity metrics
- Real-time quality scoring
- Quality threshold enforcement
"""

import re
import math
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter

logger = logging.getLogger(__name__)


# ==================== Enums and Data Classes ====================

class ReadabilityLevel(str, Enum):
    """Readability grade levels"""
    ELEMENTARY = "elementary"      # Grade 1-5
    MIDDLE_SCHOOL = "middle_school"  # Grade 6-8
    HIGH_SCHOOL = "high_school"    # Grade 9-12
    COLLEGE = "college"            # Grade 13-16
    GRADUATE = "graduate"          # Grade 17+


class ContentQualityGrade(str, Enum):
    """Overall content quality grades"""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"           # 75-89
    ACCEPTABLE = "acceptable"  # 60-74
    NEEDS_IMPROVEMENT = "needs_improvement"  # 40-59
    POOR = "poor"           # 0-39


@dataclass
class ReadabilityMetrics:
    """Readability analysis results"""
    flesch_reading_ease: float  # 0-100 (higher = easier)
    flesch_kincaid_grade: float  # US grade level
    gunning_fog_index: float
    smog_index: float
    automated_readability_index: float
    coleman_liau_index: float
    average_grade_level: float
    readability_level: ReadabilityLevel

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flesch_reading_ease": round(self.flesch_reading_ease, 2),
            "flesch_kincaid_grade": round(self.flesch_kincaid_grade, 2),
            "gunning_fog_index": round(self.gunning_fog_index, 2),
            "smog_index": round(self.smog_index, 2),
            "automated_readability_index": round(self.automated_readability_index, 2),
            "coleman_liau_index": round(self.coleman_liau_index, 2),
            "average_grade_level": round(self.average_grade_level, 2),
            "readability_level": self.readability_level.value
        }


@dataclass
class TranscriptQuality:
    """Transcript quality metrics"""
    confidence_score: float  # 0-1 from ASR/transcription
    word_error_rate: Optional[float]  # Estimated WER
    completeness: float  # Percentage of content transcribed
    speaker_identification: bool
    timestamp_accuracy: float
    punctuation_quality: float
    formatting_quality: float
    overall_quality: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence_score": round(self.confidence_score, 3),
            "word_error_rate": round(self.word_error_rate, 3) if self.word_error_rate else None,
            "completeness": round(self.completeness, 3),
            "speaker_identification": self.speaker_identification,
            "timestamp_accuracy": round(self.timestamp_accuracy, 3),
            "punctuation_quality": round(self.punctuation_quality, 3),
            "formatting_quality": round(self.formatting_quality, 3),
            "overall_quality": round(self.overall_quality, 3)
        }


@dataclass
class ContentComplexity:
    """Content complexity metrics"""
    vocabulary_diversity: float  # Type-token ratio
    avg_word_length: float
    avg_sentence_length: float
    complex_word_percentage: float  # 3+ syllables
    technical_term_density: float
    concept_density: float  # Concepts per 100 words
    structural_complexity: float
    overall_complexity: float  # 0-1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocabulary_diversity": round(self.vocabulary_diversity, 3),
            "avg_word_length": round(self.avg_word_length, 2),
            "avg_sentence_length": round(self.avg_sentence_length, 2),
            "complex_word_percentage": round(self.complex_word_percentage, 2),
            "technical_term_density": round(self.technical_term_density, 3),
            "concept_density": round(self.concept_density, 2),
            "structural_complexity": round(self.structural_complexity, 3),
            "overall_complexity": round(self.overall_complexity, 3)
        }


@dataclass
class AccessibilityScore:
    """Accessibility quality metrics"""
    alt_text_coverage: float  # Images with alt text
    heading_structure: float  # Proper heading hierarchy
    link_descriptiveness: float  # Links with descriptive text
    color_contrast: float  # Color contrast compliance
    reading_order: float  # Logical reading order
    caption_availability: float  # Video captions
    overall_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alt_text_coverage": round(self.alt_text_coverage, 3),
            "heading_structure": round(self.heading_structure, 3),
            "link_descriptiveness": round(self.link_descriptiveness, 3),
            "color_contrast": round(self.color_contrast, 3),
            "reading_order": round(self.reading_order, 3),
            "caption_availability": round(self.caption_availability, 3),
            "overall_score": round(self.overall_score, 3)
        }


@dataclass
class ContentQualityReport:
    """Complete content quality report"""
    content_id: str
    readability: ReadabilityMetrics
    complexity: ContentComplexity
    accessibility: AccessibilityScore
    transcript_quality: Optional[TranscriptQuality]
    overall_grade: ContentQualityGrade
    overall_score: float
    recommendations: List[str]
    analyzed_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_id": self.content_id,
            "readability": self.readability.to_dict(),
            "complexity": self.complexity.to_dict(),
            "accessibility": self.accessibility.to_dict(),
            "transcript_quality": self.transcript_quality.to_dict() if self.transcript_quality else None,
            "overall_grade": self.overall_grade.value,
            "overall_score": round(self.overall_score, 2),
            "recommendations": self.recommendations,
            "analyzed_at": self.analyzed_at
        }


# ==================== Quality Metrics Service ====================

class QualityMetricsService:
    """
    Analyzes and tracks content quality metrics.
    """

    # Quality thresholds
    QUALITY_THRESHOLDS = {
        ContentQualityGrade.EXCELLENT: 90,
        ContentQualityGrade.GOOD: 75,
        ContentQualityGrade.ACCEPTABLE: 60,
        ContentQualityGrade.NEEDS_IMPROVEMENT: 40,
        ContentQualityGrade.POOR: 0,
    }

    # Common complex words to exclude from complexity calculation
    COMMON_COMPLEX_WORDS = {
        "important", "different", "understand", "information", "development",
        "government", "environment", "international", "organization"
    }

    def __init__(self):
        self._cache: Dict[str, ContentQualityReport] = {}

    def analyze_content(
        self,
        content_id: str,
        text: str,
        has_transcript: bool = False,
        transcript_confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContentQualityReport:
        """
        Analyze content quality.

        Args:
            content_id: Content identifier
            text: Text content to analyze
            has_transcript: Whether content has transcript
            transcript_confidence: ASR confidence if available
            metadata: Additional content metadata

        Returns:
            Complete quality report
        """
        from datetime import datetime

        # Calculate readability
        readability = self.calculate_readability(text)

        # Calculate complexity
        complexity = self.calculate_complexity(text)

        # Calculate accessibility (simplified without actual content structure)
        accessibility = self.estimate_accessibility(text, metadata or {})

        # Calculate transcript quality if applicable
        transcript_quality = None
        if has_transcript:
            transcript_quality = self.estimate_transcript_quality(
                text, transcript_confidence
            )

        # Calculate overall score
        scores = [
            self._readability_to_score(readability),
            1 - complexity.overall_complexity,  # Lower complexity is better for general content
            accessibility.overall_score,
        ]
        if transcript_quality:
            scores.append(transcript_quality.overall_quality)

        overall_score = (sum(scores) / len(scores)) * 100

        # Determine grade
        overall_grade = self._score_to_grade(overall_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            readability, complexity, accessibility, transcript_quality
        )

        report = ContentQualityReport(
            content_id=content_id,
            readability=readability,
            complexity=complexity,
            accessibility=accessibility,
            transcript_quality=transcript_quality,
            overall_grade=overall_grade,
            overall_score=overall_score,
            recommendations=recommendations,
            analyzed_at=datetime.utcnow().isoformat()
        )

        # Cache result
        self._cache[content_id] = report

        return report

    def calculate_readability(self, text: str) -> ReadabilityMetrics:
        """
        Calculate readability metrics.

        Uses multiple formulas for comprehensive analysis.
        """
        # Preprocess text
        sentences = self._split_sentences(text)
        words = self._get_words(text)
        syllables = [self._count_syllables(w) for w in words]

        if not sentences or not words:
            return ReadabilityMetrics(
                flesch_reading_ease=0,
                flesch_kincaid_grade=0,
                gunning_fog_index=0,
                smog_index=0,
                automated_readability_index=0,
                coleman_liau_index=0,
                average_grade_level=0,
                readability_level=ReadabilityLevel.GRADUATE
            )

        total_sentences = len(sentences)
        total_words = len(words)
        total_syllables = sum(syllables)
        total_characters = sum(len(w) for w in words)

        # Complex words (3+ syllables)
        complex_words = sum(1 for s in syllables if s >= 3)

        # Flesch Reading Ease
        fre = 206.835 - (1.015 * (total_words / total_sentences)) - (84.6 * (total_syllables / total_words))
        fre = max(0, min(100, fre))

        # Flesch-Kincaid Grade Level
        fkgl = (0.39 * (total_words / total_sentences)) + (11.8 * (total_syllables / total_words)) - 15.59

        # Gunning Fog Index
        fog = 0.4 * ((total_words / total_sentences) + 100 * (complex_words / total_words))

        # SMOG Index
        if total_sentences >= 30:
            smog = 1.0430 * math.sqrt(complex_words * (30 / total_sentences)) + 3.1291
        else:
            smog = 1.0430 * math.sqrt(complex_words * 30) + 3.1291

        # Automated Readability Index
        ari = (4.71 * (total_characters / total_words)) + (0.5 * (total_words / total_sentences)) - 21.43

        # Coleman-Liau Index
        L = (total_characters / total_words) * 100
        S = (total_sentences / total_words) * 100
        cli = (0.0588 * L) - (0.296 * S) - 15.8

        # Average grade level
        avg_grade = (fkgl + fog + smog + ari + cli) / 5

        # Determine readability level
        if avg_grade <= 5:
            level = ReadabilityLevel.ELEMENTARY
        elif avg_grade <= 8:
            level = ReadabilityLevel.MIDDLE_SCHOOL
        elif avg_grade <= 12:
            level = ReadabilityLevel.HIGH_SCHOOL
        elif avg_grade <= 16:
            level = ReadabilityLevel.COLLEGE
        else:
            level = ReadabilityLevel.GRADUATE

        return ReadabilityMetrics(
            flesch_reading_ease=fre,
            flesch_kincaid_grade=fkgl,
            gunning_fog_index=fog,
            smog_index=smog,
            automated_readability_index=ari,
            coleman_liau_index=cli,
            average_grade_level=avg_grade,
            readability_level=level
        )

    def calculate_complexity(
        self,
        text: str,
        technical_terms: Optional[List[str]] = None
    ) -> ContentComplexity:
        """
        Calculate content complexity metrics.
        """
        words = self._get_words(text)
        sentences = self._split_sentences(text)

        if not words or not sentences:
            return ContentComplexity(
                vocabulary_diversity=0,
                avg_word_length=0,
                avg_sentence_length=0,
                complex_word_percentage=0,
                technical_term_density=0,
                concept_density=0,
                structural_complexity=0,
                overall_complexity=0
            )

        # Vocabulary diversity (Type-Token Ratio)
        unique_words = set(w.lower() for w in words)
        ttr = len(unique_words) / len(words) if words else 0

        # Average word length
        avg_word_len = sum(len(w) for w in words) / len(words)

        # Average sentence length
        avg_sent_len = len(words) / len(sentences)

        # Complex word percentage (3+ syllables, excluding common words)
        syllables = [self._count_syllables(w) for w in words]
        complex_count = sum(
            1 for w, s in zip(words, syllables)
            if s >= 3 and w.lower() not in self.COMMON_COMPLEX_WORDS
        )
        complex_pct = (complex_count / len(words)) * 100

        # Technical term density
        technical_terms = technical_terms or []
        if technical_terms:
            tech_count = sum(
                1 for w in words if w.lower() in [t.lower() for t in technical_terms]
            )
            tech_density = tech_count / len(words)
        else:
            # Estimate: words with unusual patterns
            tech_density = sum(
                1 for w in words
                if len(w) > 8 or (w[0].isupper() and len(w) > 3)
            ) / len(words)

        # Concept density (estimated by capitalized terms and multi-word phrases)
        capitalized = sum(1 for w in words if w[0].isupper() and len(w) > 2)
        concept_density = (capitalized / len(words)) * 100

        # Structural complexity (sentence length variation)
        if len(sentences) >= 2:
            sent_lengths = [len(s.split()) for s in sentences]
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            structural = min(1.0, math.sqrt(variance) / 20)
        else:
            structural = 0.5

        # Overall complexity (normalized 0-1)
        overall = (
            0.2 * (1 - ttr) +  # Lower diversity = more complexity
            0.2 * min(1.0, avg_word_len / 10) +
            0.2 * min(1.0, avg_sent_len / 40) +
            0.2 * (complex_pct / 30) +
            0.2 * structural
        )

        return ContentComplexity(
            vocabulary_diversity=ttr,
            avg_word_length=avg_word_len,
            avg_sentence_length=avg_sent_len,
            complex_word_percentage=complex_pct,
            technical_term_density=tech_density,
            concept_density=concept_density,
            structural_complexity=structural,
            overall_complexity=overall
        )

    def estimate_transcript_quality(
        self,
        text: str,
        asr_confidence: Optional[float] = None
    ) -> TranscriptQuality:
        """
        Estimate transcript quality metrics.
        """
        # Base confidence from ASR or estimate
        confidence = asr_confidence if asr_confidence else 0.85

        # Check for quality indicators
        words = self._get_words(text)
        sentences = self._split_sentences(text)

        # Completeness (estimate by sentence structure)
        incomplete_sentences = sum(
            1 for s in sentences
            if not s.strip().endswith(('.', '!', '?', '"'))
        )
        completeness = 1 - (incomplete_sentences / len(sentences)) if sentences else 0.5

        # Speaker identification (check for speaker labels)
        speaker_labels = bool(re.search(r'\[Speaker \d+\]|\[SPEAKER_\d+\]|Speaker:', text))

        # Timestamp accuracy (check for timestamp patterns)
        timestamps = len(re.findall(r'\[\d{2}:\d{2}(:\d{2})?\]', text))
        timestamp_accuracy = min(1.0, timestamps / max(1, len(sentences) / 10))

        # Punctuation quality
        punctuation_ratio = sum(1 for c in text if c in '.,!?;:') / max(1, len(words))
        expected_ratio = 0.1  # Roughly 1 punctuation per 10 words
        punctuation_quality = min(1.0, punctuation_ratio / expected_ratio)

        # Formatting quality (paragraphs, capitalization)
        paragraphs = text.count('\n\n') + 1
        expected_paragraphs = len(sentences) / 5
        formatting = min(1.0, paragraphs / max(1, expected_paragraphs))

        # Word error rate estimation (simplified)
        # Check for common ASR errors
        error_patterns = [
            r'\b(?:gonna|wanna|kinda|gotta)\b',  # Informal speech
            r'\s{2,}',  # Multiple spaces
            r'[^\w\s]{3,}',  # Repeated punctuation
        ]
        error_count = sum(len(re.findall(p, text)) for p in error_patterns)
        estimated_wer = min(0.5, error_count / max(1, len(words)))

        # Overall quality
        overall = (
            0.3 * confidence +
            0.2 * completeness +
            0.15 * punctuation_quality +
            0.15 * formatting +
            0.1 * timestamp_accuracy +
            0.1 * (1 - estimated_wer)
        )

        return TranscriptQuality(
            confidence_score=confidence,
            word_error_rate=estimated_wer,
            completeness=completeness,
            speaker_identification=speaker_labels,
            timestamp_accuracy=timestamp_accuracy,
            punctuation_quality=punctuation_quality,
            formatting_quality=formatting,
            overall_quality=overall
        )

    def estimate_accessibility(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> AccessibilityScore:
        """
        Estimate accessibility score.

        In production, would analyze actual HTML/content structure.
        """
        # Heading structure (check for markdown headers or capitalized lines)
        headings = len(re.findall(r'^#{1,6}\s', text, re.MULTILINE))
        heading_score = min(1.0, headings / 5) if headings else 0.5

        # Alt text coverage (from metadata or estimate)
        image_count = metadata.get("image_count", 0)
        alt_text_count = metadata.get("images_with_alt", 0)
        alt_coverage = alt_text_count / max(1, image_count) if image_count else 1.0

        # Link descriptiveness (check for generic link text)
        links = re.findall(r'\[([^\]]+)\]\([^\)]+\)', text)
        generic_links = ["click here", "here", "link", "more", "read more"]
        descriptive_links = sum(
            1 for link in links
            if link.lower() not in generic_links
        )
        link_score = descriptive_links / max(1, len(links)) if links else 1.0

        # Color contrast (would need actual design analysis)
        color_contrast = metadata.get("color_contrast_score", 0.8)

        # Reading order
        reading_order = 0.9  # Assume good order for text content

        # Caption availability
        has_captions = metadata.get("has_captions", False)
        has_video = metadata.get("has_video", False)
        caption_score = 1.0 if has_captions or not has_video else 0.0

        # Overall accessibility
        overall = (
            0.2 * alt_coverage +
            0.2 * heading_score +
            0.15 * link_score +
            0.15 * color_contrast +
            0.15 * reading_order +
            0.15 * caption_score
        )

        return AccessibilityScore(
            alt_text_coverage=alt_coverage,
            heading_structure=heading_score,
            link_descriptiveness=link_score,
            color_contrast=color_contrast,
            reading_order=reading_order,
            caption_availability=caption_score,
            overall_score=overall
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_words(self, text: str) -> List[str]:
        """Extract words from text"""
        return re.findall(r'\b[a-zA-Z]+\b', text)

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        prev_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        # Adjust for silent e
        if word.endswith('e'):
            count -= 1

        # Ensure at least one syllable
        return max(1, count)

    def _readability_to_score(self, readability: ReadabilityMetrics) -> float:
        """Convert readability to 0-1 score"""
        # Target: middle school to high school level (grades 6-12)
        target_grade = 9
        grade = readability.average_grade_level

        # Penalize both too easy and too hard
        deviation = abs(grade - target_grade)
        score = max(0, 1 - (deviation / 10))

        return score

    def _score_to_grade(self, score: float) -> ContentQualityGrade:
        """Convert score to quality grade"""
        for grade, threshold in sorted(
            self.QUALITY_THRESHOLDS.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if score >= threshold:
                return grade
        return ContentQualityGrade.POOR

    def _generate_recommendations(
        self,
        readability: ReadabilityMetrics,
        complexity: ContentComplexity,
        accessibility: AccessibilityScore,
        transcript: Optional[TranscriptQuality]
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        # Readability recommendations
        if readability.average_grade_level > 12:
            recommendations.append("Consider simplifying sentence structure for broader accessibility")
        elif readability.average_grade_level < 6:
            recommendations.append("Content may be too simple for target audience")

        if readability.flesch_reading_ease < 50:
            recommendations.append("Use shorter sentences and simpler words to improve readability")

        # Complexity recommendations
        if complexity.vocabulary_diversity < 0.3:
            recommendations.append("Increase vocabulary variety to maintain reader interest")

        if complexity.avg_sentence_length > 25:
            recommendations.append("Break up long sentences for better comprehension")

        if complexity.complex_word_percentage > 20:
            recommendations.append("Define or simplify technical terms for clarity")

        # Accessibility recommendations
        if accessibility.alt_text_coverage < 0.8:
            recommendations.append("Add descriptive alt text to all images")

        if accessibility.heading_structure < 0.5:
            recommendations.append("Improve heading structure for better navigation")

        if accessibility.link_descriptiveness < 0.7:
            recommendations.append("Use descriptive link text instead of generic phrases")

        # Transcript recommendations
        if transcript:
            if transcript.confidence_score < 0.8:
                recommendations.append("Review transcript for potential transcription errors")

            if not transcript.speaker_identification:
                recommendations.append("Add speaker labels for multi-speaker content")

            if transcript.timestamp_accuracy < 0.5:
                recommendations.append("Add timestamps for easier navigation")

        if not recommendations:
            recommendations.append("Content meets quality standards")

        return recommendations

    def get_cached_report(self, content_id: str) -> Optional[ContentQualityReport]:
        """Get cached quality report"""
        return self._cache.get(content_id)

    def clear_cache(self, content_id: Optional[str] = None):
        """Clear quality report cache"""
        if content_id:
            self._cache.pop(content_id, None)
        else:
            self._cache.clear()


# Singleton instance
quality_metrics_service = QualityMetricsService()
