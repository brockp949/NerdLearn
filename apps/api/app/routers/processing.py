"""
Processing Status Router
Check status of background processing tasks with real-time updates
"""
import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.models.course import Module
from app.services.worker_client import get_task_status, get_task_progress
from typing import Dict, Any, AsyncGenerator

router = APIRouter()


@router.get("/modules/{module_id}/processing-status")
async def get_module_processing_status(
    module_id: int, db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get processing status for a module"""
    # Get module
    result = await db.execute(select(Module).where(Module.id == module_id))
    module = result.scalar_one_or_none()

    if not module:
        raise HTTPException(status_code=404, detail="Module not found")

    # Get task status if task exists
    task_status = None
    task_progress = None
    if module.processing_task_id:
        task_status = get_task_status(module.processing_task_id)
        task_progress = get_task_progress(module.processing_task_id)

    return {
        "module_id": module_id,
        "processing_status": module.processing_status.value if module.processing_status else "pending",
        "is_processed": module.is_processed,
        "task_id": module.processing_task_id,
        "task_status": task_status,
        "task_progress": task_progress,
        "error": module.processing_error,
        "chunk_count": module.chunk_count,
        "concept_count": module.concept_count,
        "processed_at": module.processed_at,
    }


@router.get("/tasks/{task_id}/status")
async def get_task_status_endpoint(task_id: str) -> Dict[str, Any]:
    """
    Get status of a specific Celery task.

    Returns current state, progress, and any result/error.
    """
    status = get_task_status(task_id)
    progress = get_task_progress(task_id)

    if not status:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task_id,
        "status": status,
        "progress": progress,
    }


async def _generate_progress_events(
    task_id: str,
    poll_interval: float = 1.0,
    timeout: float = 3600.0,
) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events for task progress.

    Args:
        task_id: Celery task ID
        poll_interval: Seconds between status checks
        timeout: Maximum time to stream events

    Yields:
        SSE-formatted events
    """
    elapsed = 0.0
    last_status = None

    while elapsed < timeout:
        status = get_task_status(task_id)
        progress = get_task_progress(task_id)

        # Only send update if status changed
        current_state = (status, json.dumps(progress) if progress else None)
        if current_state != last_status:
            event_data = {
                "task_id": task_id,
                "status": status,
                "progress": progress,
                "elapsed_seconds": elapsed,
            }
            yield f"data: {json.dumps(event_data)}\n\n"
            last_status = current_state

        # Check if task completed
        if status in ("SUCCESS", "FAILURE", "REVOKED"):
            # Send final event
            final_data = {
                "task_id": task_id,
                "status": status,
                "progress": progress,
                "completed": True,
                "elapsed_seconds": elapsed,
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            break

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    # Timeout event
    if elapsed >= timeout:
        yield f"data: {json.dumps({'task_id': task_id, 'status': 'TIMEOUT', 'message': 'Stream timeout'})}\n\n"


@router.get("/tasks/{task_id}/progress-stream")
async def stream_task_progress(
    task_id: str,
    poll_interval: float = 1.0,
):
    """
    Stream task progress using Server-Sent Events (SSE).

    This endpoint provides real-time progress updates for long-running tasks.
    Connect using EventSource in JavaScript:

    ```javascript
    const eventSource = new EventSource('/api/processing/tasks/{task_id}/progress-stream');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Progress:', data.progress);
        if (data.completed) {
            eventSource.close();
        }
    };
    ```
    """
    return StreamingResponse(
        _generate_progress_events(task_id, poll_interval),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.websocket("/tasks/{task_id}/ws")
async def websocket_task_progress(
    websocket: WebSocket,
    task_id: str,
):
    """
    WebSocket endpoint for real-time task progress.

    Provides bidirectional communication for task monitoring.
    Clients can send commands like {"action": "cancel"} to control the task.
    """
    await websocket.accept()

    try:
        # Start background progress sender
        async def send_progress():
            last_status = None
            while True:
                status = get_task_status(task_id)
                progress = get_task_progress(task_id)

                current_state = (status, json.dumps(progress) if progress else None)
                if current_state != last_status:
                    await websocket.send_json({
                        "type": "progress",
                        "task_id": task_id,
                        "status": status,
                        "progress": progress,
                    })
                    last_status = current_state

                    # Stop if completed
                    if status in ("SUCCESS", "FAILURE", "REVOKED"):
                        await websocket.send_json({
                            "type": "completed",
                            "task_id": task_id,
                            "status": status,
                            "progress": progress,
                        })
                        break

                await asyncio.sleep(1.0)

        # Create progress task
        progress_task = asyncio.create_task(send_progress())

        # Handle incoming messages
        try:
            while True:
                data = await websocket.receive_json()
                action = data.get("action")

                if action == "cancel":
                    # Cancel the Celery task
                    from app.services.worker_client import cancel_task
                    cancelled = cancel_task(task_id)
                    await websocket.send_json({
                        "type": "action_response",
                        "action": "cancel",
                        "success": cancelled,
                    })
                elif action == "ping":
                    await websocket.send_json({"type": "pong"})

        except WebSocketDisconnect:
            pass
        finally:
            progress_task.cancel()

    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e),
        })


@router.get("/batch/{batch_task_id}/status")
async def get_batch_processing_status(batch_task_id: str) -> Dict[str, Any]:
    """
    Get status of a batch processing task.

    Returns aggregated progress for all modules in the batch.
    """
    status = get_task_status(batch_task_id)
    progress = get_task_progress(batch_task_id)

    if not status:
        raise HTTPException(status_code=404, detail="Batch task not found")

    return {
        "batch_task_id": batch_task_id,
        "status": status,
        "progress": progress,
    }
