"""
Variable Reward Engine - Implements research-backed reward schedules.

Based on "Variable Reward Schedules in Educational Gamification" research.
Supports:
- Variable Ratio (VR) schedules for high engagement.
- Mastery-linked reward density (fading rewards as proficiency increases).
- Loot drop tables with rarity distributions.
"""

import random
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class RewardRarity(str, Enum):
    COMMON = "common"
    RARE = "rare"
    EPIC = "epic"
    LEGENDARY = "legendary"

class LootItem(BaseModel):
    id: str
    name: str
    rarity: RewardRarity
    base_probability: float  # Chance when triggered
    reward_type: str  # "xp", "badge", "streak_shield", "cosmetic"
    value: Any

class VariableRewardEngine:
    """
    Manages variable reward distributions to prevent overjustification 
    and maintain long-term engagement.
    """
    
    LOOT_TABLE = [
        LootItem(id="xp_boost_small", name="Small XP Surge", rarity=RewardRarity.COMMON, base_probability=0.6, reward_type="xp", value=25),
        LootItem(id="xp_boost_med", name="Medium XP Surge", rarity=RewardRarity.RARE, base_probability=0.2, reward_type="xp", value=75),
        LootItem(id="streak_shield", name="Streak Shield", rarity=RewardRarity.EPIC, base_probability=0.1, reward_type="streak_shield", value=1),
        LootItem(id="mastery_badge", name="Mastery Insight", rarity=RewardRarity.LEGENDARY, base_probability=0.05, reward_type="cosmetic", value="insight_icon_unlocked"),
    ]

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def trigger_reward(self, mastery_level: float, probability_override: Optional[float] = None) -> Optional[LootItem]:
        """
        Roll for a variable reward.
        
        Research Foundation (Fading):
        Reward density should decrease as mastery increases to transition 
        from extrinsic to intrinsic motivation.
        """
        # Base probability of triggering ANY reward
        # High mastery = lower probability of extrinsic reward (Fading)
        trigger_prob = probability_override if probability_override is not None else max(0.1, 0.5 - (mastery_level * 0.4))
        
        if random.random() > trigger_prob:
            return None

        # Roll for specific item in loot table
        # Normalize probabilities
        active_items = self.LOOT_TABLE
        probs = [item.base_probability for item in active_items]
        total_prob = sum(probs)
        normalized_probs = [p / total_prob for p in probs]
        
        return random.choices(active_items, weights=normalized_probs, k=1)[0]

    def get_reward_feedback(self, reward: LootItem, age_group: str) -> Dict[str, Any]:
        """
        Tailor feedback based on age-appropriate guidelines.
        """
        feedback = {
            "item": reward.dict(),
            "visual_effect": "standard_sparkle",
            "audio_cue": "positive_chime"
        }
        
        if age_group in ["early_childhood", "toddler"]:
            feedback["visual_effect"] = "explosion_of_stars"
            feedback["audio_cue"] = "celebration_trumpet"
            feedback["message"] = f"Wow! You found a {reward.name}!"
        elif age_group == "adolescent":
            feedback["visual_effect"] = "sleek_glow"
            feedback["message"] = f"Reward acquired: {reward.name}. Your autonomy score increases."
        else:
            feedback["message"] = f"You earned a {reward.name}."
            
        return feedback
