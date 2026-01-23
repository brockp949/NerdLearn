import asyncio
import logging
from datetime import datetime, timedelta
from app.adaptive.cognitive.frustration_detector import FrustrationDetector, InteractionEvent, FrustrationLevel, BehavioralSignal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def simulate_user_behavior():
    detector = FrustrationDetector()
    user_id = "sim_user_1"
    
    # Scene 1: Calm User (Thoughtful responses)
    logger.info("--- Simulating Calm User ---")
    events = []
    base_time = datetime.utcnow()
    for i in range(5):
        events.append(InteractionEvent(
            timestamp=base_time + timedelta(seconds=i*10),
            event_type="answer",
            correct=True,
            response_time_ms=5000 + (i*100) # > 3000ms (normal)
        ))
        
    estimate = detector.detect_frustration(user_id, events)
    if estimate.level == FrustrationLevel.NONE:
         logger.info(f"✅ Calm behavior detected properly. Level: {estimate.level}")
    else:
         logger.error(f"❌ unexpected level for calm user: {estimate.level}")

    # Scene 2: Rapid Guessing (Frustrated/Gaming)
    logger.info("--- Simulating Rapid Guessing ---")
    events = []
    for i in range(5):
        events.append(InteractionEvent(
            timestamp=base_time + timedelta(seconds=i*2),
            event_type="answer",
            correct=False,
            response_time_ms=500 # < 2000ms (rapid)
        ))
        
    estimate = detector.detect_frustration(user_id, events)
    
    # We expect some frustration level or specific signal
    if BehavioralSignal.RAPID_GUESSING in estimate.active_signals:
        logger.info(f"✅ Rapid Guessing detected! Signals: {estimate.active_signals}")
    else:
        logger.error(f"❌ Failed to detect rapid guessing. Signals: {estimate.active_signals}")
        
    # Scene 3: Wheel Spinning (Consecutive Errors w/ hints)
    logger.info("--- Simulating Wheel Spinning ---")
    events = []
    for i in range(6):
        events.append(InteractionEvent(
            timestamp=base_time + timedelta(seconds=i*5),
            event_type="answer",
            correct=False,
            response_time_ms=4000,
            hint_used=True
        ))
        
    estimate = detector.detect_frustration(user_id, events)
    
    if BehavioralSignal.CONSECUTIVE_ERRORS in estimate.active_signals or \
       BehavioralSignal.HELP_SEEKING_SPIKE in estimate.active_signals:
        logger.info(f"✅ Wheel Spinning signs detected! Level: {estimate.level}, Signals: {estimate.active_signals}")
    else:
        logger.error(f"❌ Failed to detect wheel spinning. Signals: {estimate.active_signals}")

    logger.info("Simulation Complete.")

if __name__ == "__main__":
    asyncio.run(simulate_user_behavior())
