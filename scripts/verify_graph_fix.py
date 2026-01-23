import os
import asyncio
import sys
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
api_root = os.path.join(project_root, "apps", "api")
sys.path.append(api_root)

# Import models to register mappers
try:
    from app.models import *
except ImportError as e:
    print(f"Warning: Model import failed, might be okay if not using ORM relationships directly in this script. Error: {e}")

from app.services.graph_service import AsyncGraphService
from app.core.database import AsyncSessionLocal

async def verify_fix():
    print("Starting verification of graph service fix...")
    
    async with AsyncSessionLocal() as db:
        service = AsyncGraphService(db)
        
        # Test 0: Minimal Cypher
        print("\nTest 0: Minimal connection check (RETURN 1)...")
        try:
             res = await service.run_cypher("RETURN 1", "res agtype")
             print(f"SUCCESS: RETURN 1 result: {res}")
        except Exception as e:
             print(f"FAILED: RETURN 1 error: {e}")
             return

        # Test 1: create_course_node (verify datetime issue is gone)
        print("\nTest 1: Creating Course Node via Service...")
        try:
             res = await service.create_course_node(999, "Verification Course")
             print(f"SUCCESS: created course node via Service. Result: {res}")
        except Exception as e:
            print(f"FAILED: create_course_node error: {e}")
            return

        # Test 2: create_module_node (verify syntax issue is gone)
        print("\nTest 2: Creating Module Node via Service...")
        try:
            await service.create_module_node(999, 9991, "Mod 1", 1)
            print("SUCCESS: created module node via Service.")
        except Exception as e:
            print(f"FAILED: create_module_node error: {e}")
            import traceback
            traceback.print_exc()

        # Test 3: Check ingested data for Course 1
        print("\nTest 3: Checking Ingested Graph for Course 1...")
        try:
            result = await service.get_course_graph(1)
            print(f"SUCCESS: Course 1 Graph Nodes: {len(result['nodes'])}")
            print(f"SUCCESS: Course 1 Graph Edges: {len(result['edges'])}")
        except Exception as e:
             print(f"FAILED: Course 1 check error: {e}")

if __name__ == "__main__":
    asyncio.run(verify_fix())
