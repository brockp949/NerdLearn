from fastapi import APIRouter, WebSocket

router = APIRouter()


@router.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """WebSocket endpoint for stealth assessment telemetry"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            # Process telemetry data
            await websocket.send_json({"status": "received"})
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
