import torch
import torch.nn as nn

class ImplicitTelemetryLoss(nn.Module):
    def __init__(self, recall_weight=1.0, latency_weight=0.5, hesitation_weight=0.5):
        super().__init__()
        self.recall_weight = recall_weight
        self.latency_weight = latency_weight
        self.hesitation_weight = hesitation_weight
        
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict returned by NeuralMemoryODE.predict_all()
                - probs: (T, B, 1)
                - latency: (T, B, 1)
                - hesitation: (T, B, 1)
            targets: dict of ground truth
                - recall: (T, B, 1) [0 or 1]
                - latency: (T, B, 1) [seconds]
                - hesitation: (T, B, 1) [normalized score]
                - mask: (T, B, 1) [Which time steps have valid data]
                
        Returns:
            loss: scalar tensor
        """
        # Unpack predictions
        pred_probs = predictions["probs"]
        pred_latency = predictions["latency"]
        pred_hesitation = predictions["hesitation"]
        
        # Unpack targets
        target_recall = targets["recall"]
        target_latency = targets["latency"]
        target_hesitation = targets["hesitation"]
        mask = targets.get("mask", torch.ones_like(target_recall))
        
        # Apply mask
        # We only compute loss where we have valid observations
        
        # 1. Binary Recall Loss (The "What")
        # BCE requires inputs in [0,1], pred_probs is sigmoided
        loss_recall = self.bce(pred_probs * mask, target_recall * mask)
        
        # 2. Latency Loss (The "How" - Speed)
        loss_latency = self.mse(pred_latency * mask, target_latency * mask)
        
        # 3. Hesitation Loss (The "How" - Certainty)
        loss_hesitation = self.mse(pred_hesitation * mask, target_hesitation * mask)
        
        total_loss = (self.recall_weight * loss_recall + 
                      self.latency_weight * loss_latency + 
                      self.hesitation_weight * loss_hesitation)
                      
        return total_loss, {
            "recall": loss_recall.item(),
            "latency": loss_latency.item(),
            "hesitation": loss_hesitation.item()
        }
