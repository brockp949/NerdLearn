"""
Verification script for DSPy Style Transfer
"""
import sys
import os
import asyncio

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from app.core.config import settings
from app.core.dspy_config import configure_dspy
from app.services.dspy_optimizer import StyleTransferModule

async def test_style_transfer():
    print("Testing DSPy Style Transfer Pipeline...")
    
    # 1. Configure (will fail if no API key, so we check settings)
    if not settings.OPENAI_API_KEY:
        print("SKIPPING: No OPENAI_API_KEY found in settings")
        return

    configure_dspy()
    
    # 2. Initialize Module
    pipeline = StyleTransferModule()
    
    # 3. Test Content
    content = """
    Photosynthesis is the process used by plants, algae and certain bacteria to harness energy from sunlight and turn it into chemical energy. 
    Here, carbon dioxide and water are converted into glucose (sugar) and oxygen.
    The process happens in the chloroplasts, using the green pigment chlorophyll.
    """
    
    # 4. Run Socratic Transformation
    print("\nOriginal Content:\n", content.strip())
    print("\nTransforming to 'Socratic' persona...")
    
    try:
        result = pipeline.forward(
            content=content,
            persona="Socratic",
            target_audience="high school student"
        )
        
        print("\nTransformed Content (Socratic):\n")
        print(result.transformed_content)
        
        print("\nExtracted Facts:\n")
        for fact in result.facts:
            print(f"- {fact}")
            
        print("\nSUCCESS: Pipeline executed successfully")
        
    except Exception as e:
        print(f"\nFAILURE: Pipeline failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(test_style_transfer())
