import pytest
import math
from app.adaptive.fsrs.fsrs_algorithm import FSRSAlgorithm, FSRSCard, Rating
from app.adaptive.bkt.bayesian_kt import BayesianKnowledgeTracer

def test_fsrs_math():
    """Verify research-grade exponential decay in FSRS"""
    fsrs = FSRSAlgorithm()
    
    # Test R = 0.9^(t/S)
    # If t=S, R should be exactly 0.9
    stability = 10.0
    elapsed = 10.0
    retrievability = fsrs.calculate_retrievability(elapsed, stability)
    assert math.isclose(retrievability, 0.9, rel_tol=1e-5)
    
    # Test interval calculation: R_target = 0.9^(I/S) => I = S * ln(R_target)/ln(0.9)
    # If target retention is 0.9, interval should be equal to stability
    fsrs.params["request_retention"] = 0.9
    interval = fsrs.next_interval(stability)
    assert interval == 10

def test_bkt_calibration():
    """Verify BKT parameter self-calibration"""
    bkt = BayesianKnowledgeTracer()
    initial_p_g = bkt.p_g
    initial_p_s = bkt.p_s
    
    # History of "Lucky Guesses": mastery is low but user is correct
    history = [(0.1, True)] * 10
    bkt.calibrate_parameters(history, learning_rate=1.0)
    
    # P(G) should increase
    assert bkt.p_g > initial_p_g
    
    # History of "Slips": mastery is high but user is incorrect
    history = [(0.9, False)] * 10
    bkt.calibrate_parameters(history, learning_rate=1.0)
    
    # P(S) should increase
    assert bkt.p_s > initial_p_s

if __name__ == "__main__":
    test_fsrs_math()
    test_bkt_calibration()
    print("All adaptive math tests passed!")
