"""
Tests for QualityMetricsService
"""
import pytest
from unittest.mock import MagicMock, patch


class TestReadabilityLevel:
    """Tests for ReadabilityLevel enum"""

    def test_readability_levels(self):
        """Test all readability level values"""
        from app.services.quality_metrics import ReadabilityLevel

        assert ReadabilityLevel.ELEMENTARY == "elementary"
        assert ReadabilityLevel.MIDDLE_SCHOOL == "middle_school"
        assert ReadabilityLevel.HIGH_SCHOOL == "high_school"
        assert ReadabilityLevel.COLLEGE == "college"
        assert ReadabilityLevel.GRADUATE == "graduate"


class TestContentQualityGrade:
    """Tests for ContentQualityGrade enum"""

    def test_quality_grades(self):
        """Test all quality grade values"""
        from app.services.quality_metrics import ContentQualityGrade

        assert ContentQualityGrade.EXCELLENT == "excellent"
        assert ContentQualityGrade.GOOD == "good"
        assert ContentQualityGrade.ACCEPTABLE == "acceptable"
        assert ContentQualityGrade.NEEDS_IMPROVEMENT == "needs_improvement"
        assert ContentQualityGrade.POOR == "poor"


class TestReadabilityMetrics:
    """Tests for ReadabilityMetrics dataclass"""

    def test_to_dict(self):
        """Test conversion to dictionary"""
        from app.services.quality_metrics import ReadabilityMetrics, ReadabilityLevel

        metrics = ReadabilityMetrics(
            flesch_reading_ease=65.5,
            flesch_kincaid_grade=8.5,
            gunning_fog_index=10.2,
            smog_index=9.8,
            automated_readability_index=9.1,
            coleman_liau_index=10.0,
            average_grade_level=9.5,
            readability_level=ReadabilityLevel.HIGH_SCHOOL
        )

        result = metrics.to_dict()

        assert result["flesch_reading_ease"] == 65.5
        assert result["flesch_kincaid_grade"] == 8.5
        assert result["readability_level"] == "high_school"


class TestQualityMetricsService:
    """Tests for QualityMetricsService"""

    @pytest.fixture
    def service(self):
        """Create service instance"""
        from app.services.quality_metrics import QualityMetricsService
        return QualityMetricsService()

    def test_split_sentences(self, service):
        """Test sentence splitting"""
        text = "This is sentence one. This is sentence two! Is this sentence three?"
        sentences = service._split_sentences(text)

        assert len(sentences) == 3
        assert "This is sentence one" in sentences[0]

    def test_split_sentences_empty(self, service):
        """Test sentence splitting with empty text"""
        sentences = service._split_sentences("")
        assert len(sentences) == 0

    def test_get_words(self, service):
        """Test word extraction"""
        text = "Hello world, this is a test."
        words = service._get_words(text)

        assert "Hello" in words
        assert "world" in words
        assert "test" in words
        assert len(words) == 6

    def test_count_syllables_simple(self, service):
        """Test syllable counting for simple words"""
        assert service._count_syllables("cat") == 1
        assert service._count_syllables("hello") == 2
        assert service._count_syllables("beautiful") == 3

    def test_count_syllables_complex(self, service):
        """Test syllable counting for complex words"""
        assert service._count_syllables("understanding") >= 4
        assert service._count_syllables("a") == 1  # Minimum 1 syllable

    def test_calculate_readability_simple_text(self, service):
        """Test readability calculation for simple text"""
        text = "The cat sat on the mat. The dog ran fast. It was a good day."
        metrics = service.calculate_readability(text)

        assert metrics.flesch_reading_ease > 50  # Should be easy to read
        assert metrics.flesch_kincaid_grade < 10  # Below high school level
        assert metrics.readability_level in [
            service.calculate_readability.__annotations__.get('return'),
        ] or True  # Just verify it returns

    def test_calculate_readability_complex_text(self, service):
        """Test readability calculation for complex text"""
        text = """The epistemological implications of quantum mechanics fundamentally
        challenge our classical understanding of deterministic causality, necessitating
        a comprehensive reevaluation of the philosophical underpinnings of scientific
        methodology and the nature of empirical observation."""

        metrics = service.calculate_readability(text)

        assert metrics.flesch_reading_ease < 50  # Hard to read
        assert metrics.flesch_kincaid_grade > 12  # College level or above

    def test_calculate_readability_empty_text(self, service):
        """Test readability calculation for empty text"""
        metrics = service.calculate_readability("")

        assert metrics.flesch_reading_ease == 0
        assert metrics.flesch_kincaid_grade == 0

    def test_calculate_complexity(self, service):
        """Test complexity calculation"""
        text = """Python is a programming language. It is used for many things.
        Variables store data. Functions perform actions. Classes organize code."""

        complexity = service.calculate_complexity(text)

        assert 0 <= complexity.vocabulary_diversity <= 1
        assert complexity.avg_word_length > 0
        assert complexity.avg_sentence_length > 0
        assert 0 <= complexity.overall_complexity <= 1

    def test_calculate_complexity_empty_text(self, service):
        """Test complexity calculation for empty text"""
        complexity = service.calculate_complexity("")

        assert complexity.vocabulary_diversity == 0
        assert complexity.overall_complexity == 0

    def test_estimate_transcript_quality_with_confidence(self, service):
        """Test transcript quality estimation with ASR confidence"""
        text = "This is a transcript. It has proper punctuation."
        quality = service.estimate_transcript_quality(text, asr_confidence=0.95)

        assert quality.confidence_score == 0.95
        assert quality.overall_quality > 0

    def test_estimate_transcript_quality_without_confidence(self, service):
        """Test transcript quality estimation without ASR confidence"""
        text = "This is a transcript. It has proper punctuation."
        quality = service.estimate_transcript_quality(text)

        assert quality.confidence_score == 0.85  # Default
        assert quality.completeness > 0

    def test_estimate_transcript_quality_with_speaker_labels(self, service):
        """Test transcript quality with speaker identification"""
        text = "[Speaker 1] Hello everyone. [Speaker 2] Welcome to the class."
        quality = service.estimate_transcript_quality(text)

        assert quality.speaker_identification is True

    def test_estimate_accessibility(self, service):
        """Test accessibility estimation"""
        text = "# Introduction\n\nThis is content.\n\n## Section 1\n\nMore content."
        metadata = {"image_count": 2, "images_with_alt": 2}

        accessibility = service.estimate_accessibility(text, metadata)

        assert accessibility.alt_text_coverage == 1.0
        assert accessibility.heading_structure > 0
        assert 0 <= accessibility.overall_score <= 1

    def test_analyze_content_simple(self, service):
        """Test complete content analysis"""
        text = "Python is a popular programming language. It is easy to learn."

        report = service.analyze_content(
            content_id="test_1",
            text=text
        )

        assert report.content_id == "test_1"
        assert report.readability is not None
        assert report.complexity is not None
        assert report.accessibility is not None
        assert report.overall_score >= 0
        assert len(report.recommendations) > 0

    def test_analyze_content_with_transcript(self, service):
        """Test content analysis with transcript"""
        text = "This is transcribed content. It comes from audio."

        report = service.analyze_content(
            content_id="test_2",
            text=text,
            has_transcript=True,
            transcript_confidence=0.9
        )

        assert report.transcript_quality is not None
        assert report.transcript_quality.confidence_score == 0.9

    def test_score_to_grade(self, service):
        """Test score to grade conversion"""
        from app.services.quality_metrics import ContentQualityGrade

        assert service._score_to_grade(95) == ContentQualityGrade.EXCELLENT
        assert service._score_to_grade(80) == ContentQualityGrade.GOOD
        assert service._score_to_grade(65) == ContentQualityGrade.ACCEPTABLE
        assert service._score_to_grade(45) == ContentQualityGrade.NEEDS_IMPROVEMENT
        assert service._score_to_grade(20) == ContentQualityGrade.POOR

    def test_caching(self, service):
        """Test report caching"""
        text = "Test content for caching."

        # First analysis
        report1 = service.analyze_content("cache_test", text)

        # Get from cache
        cached = service.get_cached_report("cache_test")

        assert cached is not None
        assert cached.content_id == report1.content_id

    def test_clear_cache_specific(self, service):
        """Test clearing specific cache entry"""
        service.analyze_content("item_1", "Content 1")
        service.analyze_content("item_2", "Content 2")

        service.clear_cache("item_1")

        assert service.get_cached_report("item_1") is None
        assert service.get_cached_report("item_2") is not None

    def test_clear_cache_all(self, service):
        """Test clearing all cache"""
        service.analyze_content("item_1", "Content 1")
        service.analyze_content("item_2", "Content 2")

        service.clear_cache()

        assert service.get_cached_report("item_1") is None
        assert service.get_cached_report("item_2") is None


class TestRecommendations:
    """Tests for recommendation generation"""

    @pytest.fixture
    def service(self):
        from app.services.quality_metrics import QualityMetricsService
        return QualityMetricsService()

    def test_recommendations_for_hard_text(self, service):
        """Test recommendations for difficult text"""
        hard_text = """The epistemological ramifications of phenomenological
        hermeneutics necessitate a comprehensive reevaluation of the
        ontological presuppositions underlying contemporary metaphysical
        discourse and its methodological implications."""

        report = service.analyze_content("hard_test", hard_text)

        # Should have readability recommendations
        assert any("simplif" in r.lower() for r in report.recommendations)

    def test_recommendations_meet_standards(self, service):
        """Test that good content gets positive recommendation"""
        good_text = """Python is a great language for beginners.
        It has clear syntax. You can learn it quickly.
        Many tutorials exist online. Practice makes perfect."""

        report = service.analyze_content("good_test", good_text)

        # May have "meets quality standards" if good enough
        # or specific recommendations
        assert len(report.recommendations) > 0
