from app.services.code_execution.executor import SafePythonExecutor

def test_safe_python_executor():
    executor = SafePythonExecutor(timeout=1.0)
    
    # Test 1: Valid Code
    code = """
def add(a, b):
    return a + b
"""
    test_cases = [
        {"input": [1, 2], "expected": 3},
        {"input": [5, 5], "expected": 10}
    ]
    
    result = executor.execute(code, test_cases, "add")
    assert result.passed
    assert len(result.test_results) == 2
