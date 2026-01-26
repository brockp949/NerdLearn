"""
Example: Fuel Meter Cost Controls

Demonstrates strict token and step limits.

From PDF:
"Deploy the 'Fuel Meter': Implement strict token/step limits... 
to prevent runaway costs while allowing for 'Deep Think' capabilities."
"""

import pytest
import time
from apps.testing.agents.fuel_meter import FuelMeter, FuelBudget

def test_token_exhaustion():
    """Test token limit enforcement"""
    budget = FuelBudget(max_tokens=100)
    meter = FuelMeter(budget, agent_name="HighSpender")
    
    meter.start()
    
    # Simulate heavy usage
    meter.record_tokens(50)
    assert not meter.budget.is_exhausted()
    
    meter.record_tokens(60)
    assert meter.budget.is_exhausted()
    
    report = meter.stop()
    assert report.exhaustion_reason == "token_limit"
    print(f"\n⛽ Token exhaustion detected: {report.tokens_used}/{report.tokens_budget}")

def test_deep_think_reserve():
    """Test deep think reserve logic"""
    # 1000 tokens total, 20% reserve = 200 tokens
    budget = FuelBudget(max_tokens=1000, deep_think_reserve=0.2)
    meter = FuelMeter(budget, agent_name="DeepThinker")
    
    meter.start()
    
    # Use 700 tokens (300 remaining > 200 reserve)
    meter.record_tokens(700)
    assert meter.enter_deep_think() == True, "Should allow Deep Think (300 reserve > 200)"
    
    # Use 150 more (150 remaining < 200 reserve)
    meter.record_tokens(150)
    meter.exit_deep_think()
    
    # Should define if we can allow NEW deep think session
    # Now 150 < 200, so NO
    assert meter.enter_deep_think() == False, "Should deny Deep Think (150 remaining < 200 required)"
    
    print("\n🧠 Deep Think reserve logic validated")

if __name__ == "__main__":
    test_token_exhaustion()
    test_deep_think_reserve()
