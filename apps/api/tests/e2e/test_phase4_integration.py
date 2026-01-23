from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from app.main import app
from app.schemas.social_agent import DebateFormat, PanelPreset

client = TestClient(app)

def test_coding_challenge_flow():
    """
    Test the full coding challenge API flow.
    """
    # 1. Init Challenges (Mock)
    resp = client.post("/api/social/challenges/init?current_user_id=1")
    assert resp.status_code == 200
    
    # 2. List Challenges
    with patch("app.services.social_agent_service.SocialAgentService.get_challenges", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = [{
            "challenge_id": "test_1",
            "title": "Test Challenge",
            "description": "Desc",
            "difficulty": "beginner",
            "category": "Test",
            "concepts": [],
            "function_name": "test_func",
            "parameters": [],
            "return_type": "int",
            "test_cases": [],
            "estimated_minutes": 10,
            "language": "python"
        }]
        
        resp = client.get("/api/social/coding-challenges?current_user_id=1")
        assert resp.status_code == 200
        data = resp.json()
        assert "challenges" in data
        assert len(data["challenges"]) == 1

    # 3. Evaluate Submission
    with patch("app.services.social_agent_service.SocialAgentService.evaluate_code", new_callable=AsyncMock) as mock_eval:
        mock_eval.return_value = {
            "passed": True, 
            "overall_score": 100, 
            "tests_passed": 1, 
            "tests_total": 1, 
            "execution_time_ms": 10, 
            "dimension_scores": {}, 
            "feedback": [],
            "concepts_demonstrated": [],
            "concepts_to_review": [],
            "runtime_errors": []
        }
        
        payload = {
            "challenge_id": "test_1",
            "user_id": "1",
            "code": "def solution(): pass"
        }
        resp = client.post("/api/social/challenges/evaluate", json=payload)
        assert resp.status_code == 200
        assert resp.json()["passed"] is True

def test_debate_flow():
    """
    Test the debate session API flow.
    """
    # 1. Start Debate
    with patch("app.services.social_agent_service.SocialAgentService.start_debate", new_callable=AsyncMock) as mock_start:
        mock_start.return_value = {
            "session_id": "sess_1",
            "topic": "AI",
            "format": "roundtable",
            "participants": [{"name": "Alice", "role": "advocate"}],
            "current_round": 1,
            "max_rounds": 5,
            "opening_statements": []
        }
        
        payload = {
            "topic": "AI",
            "format": "roundtable",
            "panel_preset": "technical_pros_cons",
            "max_rounds": 5
        }
        resp = client.post("/api/social/debates/start", json=payload)
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "sess_1"

def test_teaching_flow():
    """
    Test the teachable agent API flow.
    """
    # 1. Start Teaching
    with patch("app.services.social_agent_service.SocialAgentService.start_teaching_session", new_callable=AsyncMock) as mock_start:
        mock_start.return_value = {
            "session_id": "teach_1",
            "persona": "novice",
            "opening_question": "What is it?",
            "comprehension": 0.0,
            "comprehension_level": "lost"
        }
        
        payload = {"user_id": "1", "concept_name": "Recursion", "persona": "novice"}
        resp = client.post("/api/social/teaching/start", json=payload)
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "teach_1"
