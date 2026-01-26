import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

# Example service class to mock
class LearningAnalyticsService:
    def process_student_progress(self, student_id: str, scores: list[float]) -> Dict[str, Any]:
        if not scores:
            raise ValueError("Scores cannot be empty")
        
        avg_score = sum(scores) / len(scores)
        # Handle floating point precision for the boundary case 0.8
        # e.g. 2.4 / 3 = 0.799999... < 0.8
        avg_score = round(avg_score, 10) 
        
        mastery_status = "mastered" if avg_score >= 0.8 else "learning"
        
        return {
            "student_id": student_id,
            "average": avg_score,
            "status": mastery_status,
            "samples": len(scores)
        }

class TestMockVariations:
    
    @pytest.fixture
    def service(self):
        return LearningAnalyticsService()

    def test_mock_external_dependency(self, service):
        """
        Demonstrates simple mocking of a method within the class 
        or an external dependency if it were imported.
        """
        # Scenario: We want to test process_student_progress but mocking the calculation logic 
        # (simulating a complex external call) isn't directly applicable here without a separate dependency.
        # So we'll mock a hypothetical external logger or notification system commonly used.
        
        with patch('logging.info') as mock_log:
            # Let's assume the real code would log something (it doesn't in our simple example, 
            # so we'll just manually call it to demonstrate assertion on mocks).
            import logging
            logging.info("Processing started")
            
            result = service.process_student_progress("student_123", [0.9, 0.85])
            
            mock_log.assert_called_with("Processing started")
            assert result["status"] == "mastered"

    @pytest.mark.parametrize("scores,expected_status,expected_avg", [
        ([0.9, 0.9], "mastered", 0.9),
        ([0.7, 0.7], "learning", 0.7),
        ([0.8, 0.8], "mastered", 0.8),
        ([0.95, 0.85, 0.6], "mastered", 0.8), # (0.95+0.85+0.6)/3 = 2.4/3 = 0.8
    ])
    def test_parameterized_scoring_rules(self, service, scores, expected_status, expected_avg):
        """
        Demonstrates parameterized testing for business logic variations.
        """
        result = service.process_student_progress("student_p", scores)
        
        assert result["status"] == expected_status
        assert result["average"] == pytest.approx(expected_avg)

    def test_error_handling_exception(self, service):
        """
        Demonstrates asserting that specific exceptions are raised.
        """
        with pytest.raises(ValueError, match="Scores cannot be empty"):
            service.process_student_progress("student_err", [])

    def test_mock_side_effects(self):
        """
        Demonstrates using side_effects to simulate failures in dependencies.
        """
        mock_db = Mock()
        # Simulate db raising error on connection
        mock_db.connect.side_effect = ConnectionError("DB Down")
        
        with pytest.raises(ConnectionError, match="DB Down"):
            mock_db.connect()
