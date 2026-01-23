import asyncio
import websockets
import json
import time
import requests

async def verify_telemetry():
    uri = "ws://localhost:8002/ws/verify_user/verify_session"
    api_url = "http://localhost:8002"
    
    print(f"🔌 Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to Telemetry Service")
            
            # Send a sequence of mouse events to simulate activity
            print("🖱️ Sending mouse trajectory...")
            for i in range(10):
                event = {
                    "user_id": "verify_user",
                    "session_id": "verify_session",
                    "event_type": "mouse_move",
                    "timestamp": time.time(),
                    "data": {"x": 100 + i*10, "y": 200 + i*5}
                }
                await websocket.send(json.dumps(event))
                await asyncio.sleep(0.05)
            
            # Send interaction
            await websocket.send(json.dumps({
                "user_id": "verify_user",
                "session_id": "verify_session",
                "event_type": "content_interaction",
                "timestamp": time.time(),
                "data": {"action": "click"}
            }))
            
            # Wait for processing
            print("⏳ Waiting for processing...")
            await asyncio.sleep(1)
            
            # Check Analysis Endpoints via HTTP
            print(f"🔍 Checking Analysis at {api_url}/analysis/mouse/verify_user/verify_session")
            resp = requests.get(f"{api_url}/analysis/mouse/verify_user/verify_session")
            if resp.status_code == 200:
                print(f"✅ Mouse Analysis: {resp.json()}")
            else:
                print(f"❌ Mouse Analysis Failed: {resp.status_code} - {resp.text}")

            print(f"🔍 Checking Engagement at {api_url}/analysis/engagement/verify_user/verify_session")
            resp = requests.get(f"{api_url}/analysis/engagement/verify_user/verify_session")
            if resp.status_code == 200:
                print(f"✅ Engagement Score: {resp.json()}")
            else:
                print(f"❌ Engagement Analysis Failed: {resp.status_code} - {resp.text}")
                
    except Exception as e:
        print(f"❌ Verification FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(verify_telemetry())
