"""
Locust load test for learning flow.

Usage:
    locust -f tests/performance/locustfiles/learning_flow.py --host=http://localhost:8000

This file defines load testing scenarios for the NerdLearn learning flow.
"""

# from locust import HttpUser, task, between
# Uncomment when locust is installed

# class LearningFlowUser(HttpUser):
#     """Simulates a learner going through a learning session."""
#
#     wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
#
#     def on_start(self):
#         """Login and setup before tasks."""
#         # Register/login user
#         self.user_data = {
#             "email": f"loadtest_{self.environment.runner.user_count}@example.com",
#             "password": "LoadTest123!"
#         }
#         # Login to get token
#         response = self.client.post("/api/auth/login", data=self.user_data)
#         if response.status_code == 200:
#             self.token = response.json().get("access_token")
#             self.headers = {"Authorization": f"Bearer {self.token}"}
#
#     @task(3)
#     def start_session(self):
#         """Start a learning session."""
#         self.client.post(
#             "/session/start",
#             json={"learner_id": "load_test_user", "limit": 10},
#             headers=self.headers
#         )
#
#     @task(10)
#     def answer_card(self):
#         """Answer a card in the session."""
#         # Would need session_id from start_session
#         pass
#
#     @task(1)
#     def check_health(self):
#         """Check API health."""
#         self.client.get("/health")


# Placeholder for future load testing implementation
def main():
    """Entry point for manual testing."""
    print("Locust load tests not yet configured.")
    print("Install locust: pip install locust")
    print("Run: locust -f tests/performance/locustfiles/learning_flow.py")


if __name__ == "__main__":
    main()
