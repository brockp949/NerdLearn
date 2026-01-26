import pytest
from packages.swarm.core.fuel_meter import FuelMeter, FuelLimit, FuelType

def test_fuel_initialization():
    meter = FuelMeter(limit=100)
    assert meter.limit == 100
    assert meter.current_usage == 0
    assert not meter.is_empty()

def test_fuel_spending():
    meter = FuelMeter(limit=100)
    meter.spend(50, FuelType.TOKEN)
    assert meter.current_usage == 50
    assert meter.check_remaining() == 50

def test_fuel_depletion_error():
    meter = FuelMeter(limit=100)
    meter.spend(100)
    assert meter.is_empty()
    
    with pytest.raises(RuntimeError, match="FUEL DEPLETED"):
        meter.spend(1)

def test_reset():
    meter = FuelMeter(limit=100)
    meter.spend(50)
    meter.reset()
    assert meter.current_usage == 0
