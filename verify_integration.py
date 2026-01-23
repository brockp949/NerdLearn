import requests
import json
import time

# Configuration
ORCHESTRATOR_URL = "http://127.0.0.1:8005"
SCHEDULER_URL = "http://127.0.0.1:8001"
INFERENCE_URL = "http://127.0.0.1:8003"
CONTENT_URL = "http://127.0.0.1:8004"

def print_status(service, status, details=""):
    symbol = "✅" if status else "❌"
    print(f"{symbol} {service}: {details}")

def check_health():
    print("\n--- 1. Health Checks ---")
    services = {
        "Orchestrator": ORCHESTRATOR_URL,
        "Scheduler": SCHEDULER_URL,
        "Inference": INFERENCE_URL,
        "Content": CONTENT_URL
    }
    
    all_healthy = True
    for name, url in services.items():
        try:
            # Scheduler uses root for health
            path = "/" if name == "Scheduler" else "/health"
            res = requests.get(f"{url}{path}", timeout=5)
            if res.status_code == 200:
                print_status(name, True, "Operational")
            else:
                print_status(name, False, f"Returned {res.status_code}")
                all_healthy = False
        except Exception as e:
            print_status(name, False, f"Connection Failed: {str(e)}")
            all_healthy = False
            

    return all_healthy

def seed_data(learner_id):
    print(f"\n--- 1.5 Seeding Data for {learner_id} ---")
    try:
        res = requests.post(f"{ORCHESTRATOR_URL}/test/seed", params={"learner_id": learner_id}, timeout=5)
        if res.status_code == 200:
            print_status("Seeding", True, "Test data seeded successfully")
            return True
        else:
            print_status("Seeding", False, f"Failed: {res.status_code} - {res.text}")
            return False
    except Exception as e:
        print_status("Seeding", False, f"Error: {str(e)}")
        return False


def test_communication(learner_id):
    print("\n--- 2. Inter-Service Communication (via Orchestrator) ---")
    
    # Using seeded learner ID
    print(f"Using Learner ID: {learner_id}") 
    
    # 2.1 Start Session (Orchestrator -> Scheduler)
    print("\nTesting Session Start (Orchestrator -> Scheduler)...")
    try:
        # This might fail if user doesn't exist in Postgres, but the network request should happen
        payload = {"learner_id": learner_id, "limit": 5}
        res = requests.post(f"{ORCHESTRATOR_URL}/session/start", json=payload, timeout=5)
        
        if res.status_code == 200:
            print_status("Orchestrator -> Scheduler", True, "Session started successfully")
            session_data = res.json()
            return session_data
        elif res.status_code == 404 and "profile not found" in res.text.lower():
            print_status("Orchestrator -> Scheduler", False, "Learner profile not found (Expected if DB empty)")
            print("   -> Tip: We need to seed a learner profile first.")
            return None
        else:
            print_status("Orchestrator -> Scheduler", False, f"Failed: {res.status_code} - {res.text}")
            return None
            
    except Exception as e:
        print_status("Orchestrator -> Scheduler", False, f"Error: {str(e)}")
        return None

def main():
    print("Starting Microservices Integration Test...")
    if not check_health():
        print("\n⚠️  Some services are unhealthy. Proceeding with caution...")
    
    
    learner_id = "test_user_123"
    if seed_data(learner_id):
        session = test_communication(learner_id)
        if session:
            print(f"\n✅ End-to-End Flow Verified (Session ID: {session.get('session_id')})")
    else:
        print("\n❌ Seeding failed, skipping session test.")
    
if __name__ == "__main__":
    main()
