import asyncio
import websockets
import json
import time

async def verify_telemetry():
    uri = "ws://localhost:8002/ws/verify_user/verify_session"
    
    print(f"🔌 Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to Telemetry Service")
            
            # Send a test event
            event = {
                "user_id": "verify_user",
                "session_id": "verify_session",
                "event_type": "mouse_move",
                "timestamp": time.time(),
                "data": {"x": 100, "y": 200}
            }
            
            print(f"📤 Sending event: {event['event_type']}")
            await websocket.send(json.dumps(event))
            
            # Wait for acknowledgment
            response = await websocket.recv()
            print(f"cx Received: {response}")
            
            if "status" in json.loads(response):
                print("✅ Verification PASSED: Event acknowledged")
            else:
                print("❌ Verification FAILED: Unexpected response")
                
    except Exception as e:
        print(f"❌ Verification FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(verify_telemetry())
