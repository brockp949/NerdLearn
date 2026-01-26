from tests.swarm.core.llm_factory import LLMFactory, LLMProvider
from tests.swarm.agents.refiner_agent import RefinerAgent
import os

def test_imports():
    print("Starting Multi-LLM Instantiation Test...")
    
    # We use dummy keys just to test instantiation. 
    # Validations won't work, but it proves we can creating the objects.
    
    print("\n1. Testing OpenAI instantiation...")
    try:
        # Using a recognizable fake key format if libs validate format
        llm = LLMFactory.create_llm(LLMProvider.OPENAI, "gpt-4o", api_key="sk-proj-dummy-key")
        agent = RefinerAgent(llm=llm)
        print("PASS: OpenAI Agent instantiated.")
    except Exception as e:
        print(f"FAIL: OpenAI instantiation error: {e}")

    print("\n2. Testing Anthropic (Claude) instantiation...")
    try:
        llm = LLMFactory.create_llm(LLMProvider.ANTHROPIC, "claude-3-opus-20240229", api_key="sk-ant-dummy-key")
        agent = RefinerAgent(llm=llm)
        print("PASS: Anthropic Agent instantiated.")
    except Exception as e:
        print(f"FAIL: Anthropic instantiation error: {e}")

    print("\n3. Testing Google (Gemini) instantiation...")
    try:
        llm = LLMFactory.create_llm(LLMProvider.GEMINI, "gemini-pro", api_key="dummy-google-key")
        agent = RefinerAgent(llm=llm)
        print("PASS: Gemini Agent instantiated.")
    except Exception as e:
        print(f"FAIL: Gemini instantiation error: {e}")

if __name__ == "__main__":
    test_imports()
