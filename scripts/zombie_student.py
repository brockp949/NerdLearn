import asyncio
import httpx
import random
import time
import sys

# Configuration
API_URL = "http://localhost:8000/api"
STUDENT_ID = "zombie_001"
ITERATIONS = 50
DELAY_SECONDS = 0.1

async def run_zombie_student():
    print(f"🧟 Zombie Student '{STUDENT_ID}' rising from the grave...")
    print(f"🎯 Target: {API_URL}")
    print(f"🔄 Iterations: {ITERATIONS}")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Start Session
        print("\n[1] Starting Session...")
        start_payload = {
            "learner_id": STUDENT_ID,
            "domain": "Python",
            "limit": 50
        }
        
        try:
            resp = await client.post(f"{API_URL}/session/start", json=start_payload)
            resp.raise_for_status()
            session = resp.json()
            session_id = session["session_id"]
            print(f"✅ Session Started: {session_id}")
            print(f"   Initial Card: {session['current_card']['title']}")
        except Exception as e:
            print(f"❌ Failed to start session: {e}")
            return

        # 2. Learning Loop
        print("\n[2] Entering Feeding Frenzy (Learning Loop)...")
        
        current_card_id = session['current_card']['card_id']
        total_xp = 0
        streak = 0
        
        start_time = time.time()
        
        for i in range(ITERATIONS):
            # Simulate "thinking"
            # await asyncio.sleep(DELAY_SECONDS)
            
            # Decide on answer (Zombie is smart, getting smarter)
            # 80% chance of being correct
            is_good_day = random.random() < 0.8
            rating = "good" if is_good_day else "hard"
            
            payload = {
                "session_id": session_id,
                "card_id": current_card_id,
                "rating": rating,
                "dwell_time_ms": random.randint(1000, 5000),
                "hesitation_count": random.randint(0, 3)
            }
            
            try:
                ans_resp = await client.post(f"{API_URL}/session/answer", json=payload)
                ans_resp.raise_for_status()
                result = ans_resp.json()
                
                # Update State
                current_card_id = result['next_card']['card_id']
                total_xp = result['new_total_xp']
                correct = result['correct']
                
                status_icon = "🧠" if correct else "🧟"
                print(f"   Iter {i+1}/{ITERATIONS}: {status_icon} Answered '{rating}' -> XP: {total_xp} | Next: {result['next_card']['title']}")
                
            except Exception as e:
                print(f"❌ Error at iteration {i}: {e}")
                break
                
        duration = time.time() - start_time
        print(f"\n✅ Frenzy Complete in {duration:.2f}s ({ITERATIONS/duration:.2f} req/s)")
        print(f"📈 Final XP: {total_xp}")

if __name__ == "__main__":
    try:
        asyncio.run(run_zombie_student())
    except KeyboardInterrupt:
        print("\n🧟 Zombie returning to grave...")
