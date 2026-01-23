import multiprocessing
import sys
import io
import time
import traceback
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ExecutionResult:
    passed: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    test_results: List[Dict[str, Any]] = None

class SafePythonExecutor:
    """
    Executes Python code in a separate process with timeout and restriction.
    WARNING: This is a basic isolation. For production, use Docker/nsjail.
    """
    
    def __init__(self, timeout: float = 2.0):
        self.timeout = timeout

    def execute(
        self, 
        code: str, 
        test_cases: List[Dict[str, Any]], 
        function_name: str
    ) -> ExecutionResult:
        """
        Execute code against test cases.
        """
        # Create a queue for result communication
        queue = multiprocessing.Queue()
        
        # Prepare the worker process
        process = multiprocessing.Process(
            target=self._worker,
            args=(code, test_cases, function_name, queue)
        )
        
        start_time = time.time()
        process.start()
        
        # Wait for completion or timeout
        process.join(self.timeout)
        
        execution_time = time.time() - start_time
        
        if process.is_alive():
            process.terminate()
            return ExecutionResult(
                passed=False,
                output="",
                error="Timeout: Execution exceeded safe limits.",
                execution_time=execution_time,
                test_results=[]
            )
            
        if not queue.empty():
            return queue.get()
        else:
            return ExecutionResult(
                passed=False,
                output="",
                error="Execution failed with no result.",
                execution_time=execution_time,
                test_results=[]
            )

    @staticmethod
    def _worker(code: str, test_cases: List[Dict[str, Any]], function_name: str, queue: multiprocessing.Queue):
        """
        Worker function to run in separate process.
        """
        # Redirect stdout
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        result_data = {
            "passed": True,
            "test_results": [],
            "error": None
        }
        
        try:
            # Create a safe globals dictionary
            # RestrictedPython could be used here for more security
            safe_globals = {}
            
            # Exec the user code definition
            exec(code, safe_globals)
            
            # Check if function exists
            if function_name not in safe_globals:
                raise ValueError(f"Function '{function_name}' not found in code.")
                
            func = safe_globals[function_name]
            
            # Run test cases
            for tc in test_cases:
                input_val = tc.get("input")
                expected = tc.get("expected")
                
                # Handle multiple arguments if input is a list/tuple and function accepts multiple
                # Simple heuristic: unpack if list/tuple
                try:
                    if isinstance(input_val, (list, tuple)):
                        actual = func(*input_val)
                    else:
                        actual = func(input_val)
                except TypeError:
                    # Fallback: maybe it expects a single list arg
                    actual = func(input_val)
                
                # Compare
                # Simple equality check
                passed = (actual == expected)
                
                result_data["test_results"].append({
                    "input": input_val,
                    "expected": expected,
                    "actual": actual,
                    "passed": passed
                })
                
                if not passed:
                    result_data["passed"] = False
                    
        except Exception:
            result_data["passed"] = False
            result_data["error"] = traceback.format_exc()
            
        finally:
            # Restore stdout
            sys.stdout = old_stdout
            
        # Send result back
        queue.put(ExecutionResult(
            passed=result_data["passed"],
            output=redirected_output.getvalue(),
            error=result_data["error"],
            test_results=result_data["test_results"]
        ))
