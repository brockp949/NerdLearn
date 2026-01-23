
import joblib
import numpy as np
import os
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class AffectModel:
    """
    Wrapper for the Affect Detection Machine Learning Model.
    Predicts user state (Flow, Frustrated, Bored) based on telemetry.
    """
    
    LABELS = {0: "Flow", 1: "Frustrated", 2: "Bored"}
    
    def __init__(self, model_path: str = "apps/api/data/affect_model.joblib"):
        # Resolve absolute path relative to app root if needed
        # Assuming app is run from root or we find the file relative to this one
        self.model = None
        self.model_path = model_path
        self._load_model()
        
    def _load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded Affect Model from {self.model_path}")
            else:
                logger.warning(f"Affect Model not found at {self.model_path}. Using heuristic fallback.")
        except Exception as e:
            logger.error(f"Failed to load Affect Model: {e}")

    def predict(self, telemetry_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict affect state from telemetry features.
        Expected keys: velocity_mean, velocity_variance, click_rate, idle_time_ratio, scroll_depth_rate
        """
        if not self.model:
            return self._heuristic_fallback(telemetry_data)
            
        try:
            features = [
                telemetry_data.get("velocity_mean", 0.0),
                telemetry_data.get("velocity_variance", 0.0),
                telemetry_data.get("click_rate", 0.0),
                telemetry_data.get("idle_time_ratio", 0.0),
                telemetry_data.get("scroll_depth_rate", 0.0)
            ]
            
            # Reshape for single sample
            prediction = self.model.predict([features])[0]
            confidence = np.max(self.model.predict_proba([features])[0])
            
            label = self.LABELS.get(prediction, "Unknown")
            
            return {
                "state": label,
                "confidence": float(confidence),
                "model_version": "v1.0-rf"
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"state": "Error", "confidence": 0.0}

    def _heuristic_fallback(self, data: Dict[str, float]) -> Dict[str, Any]:
        """Simple rules if model is missing"""
        if data.get("click_rate", 0) > 15 or data.get("velocity_variance", 0) > 0.3:
            return {"state": "Frustrated", "confidence": 0.6, "method": "heuristic"}
        elif data.get("idle_time_ratio", 0) > 0.5:
            return {"state": "Bored", "confidence": 0.6, "method": "heuristic"}
        else:
            return {"state": "Flow", "confidence": 0.5, "method": "heuristic"}
