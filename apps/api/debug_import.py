
import sys
import traceback

print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")

try:
    print("Attempting to import causallearn...")
    import causallearn
    print(f"Successfully imported causallearn. Version: {causallearn.__version__}")
except ImportError:
    print("Failed to import causallearn.")
    traceback.print_exc()
except Exception:
    print("An error occurred during import.")
    traceback.print_exc()

try:
    print("Attempting to import causallearn.search.ScoreBased.NOTEARS...")
    from causallearn.search.ScoreBased.NOTEARS import notears_linear
    print("Successfully imported notears_linear")
except ImportError:
    print("Failed to import notears_linear.")
    traceback.print_exc()
except Exception:
    print("An error occurred during import of notears_linear.")
    traceback.print_exc()
