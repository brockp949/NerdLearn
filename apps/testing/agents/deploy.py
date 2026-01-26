"""
Deployment Script for Agentic Testing Agents

This script deploys and tests the agentic testing framework with
both Claude and Gemini providers.

Usage:
    python -m apps.testing.agents.deploy --provider gemini
    python -m apps.testing.agents.deploy --provider claude
    python -m apps.testing.agents.deploy --test-all
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from apps.testing.agents.llm_client import (
    create_llm_client, 
    LLMConfig, 
    LLMProvider,
    MockLLMClient
)
from apps.testing.agents.base_verifier import VerifierAgent, GoalVector
from apps.testing.agents.adversarial_peer import AdversarialPeer, PoisonType
from apps.testing.agents.refiner_agent import RefinerAgent
from apps.testing.agents.topological_auditor import TopologicalAuditor


async def test_verifier(client, provider_name: str):
    """Test Verifier Agent"""
    print(f"\n🔍 Testing Verifier Agent with {provider_name}...")
    
    goal = GoalVector(
        name="Semantic Accuracy",
        description="Output must be factually correct and contextually appropriate",
        constraints=["No hallucinations", "Proper context awareness"],
        pass_threshold=0.7
    )
    
    verifier = VerifierAgent(goal, client)
    
    test_output = {
        "entity": "Apple",
        "type": "Organization",
        "context": "Technology company that makes iPhones"
    }
    
    result = await verifier.verify(
        test_output,
        context={"domain": "technology", "expected_type": "Organization"}
    )
    
    print(f"  ✓ Passed: {result.passed}")
    print(f"  ✓ Confidence: {result.confidence:.2f}")
    print(f"  ✓ Reasoning: {result.reasoning[:100]}...")
    
    return result.passed


async def test_adversarial_peer(client, provider_name: str):
    """Test Adversarial Peer"""
    print(f"\n🧨 Testing Adversarial Peer with {provider_name}...")
    
    peer = AdversarialPeer(client)
    
    scenarios = await peer.generate_poisoned_data(
        domain="learning_content",
        poison_type=PoisonType.CONFLICT,
        count=2
    )
    
    print(f"  ✓ Generated {len(scenarios)} adversarial scenarios")
    for i, scenario in enumerate(scenarios[:2]):
        if isinstance(scenario, dict):
            name = scenario.get('name', 'unnamed')
            desc = scenario.get('poison_description', 'N/A')
        else:
            name = getattr(scenario, 'name', 'unnamed')
            desc = getattr(scenario, 'poison_description', 'N/A')
            
        print(f"    {i+1}. {name}: {desc[:50]}...")
    
    return len(scenarios) > 0


async def test_refiner(client, provider_name: str):
    """Test Refiner Agent (TDFlow)"""
    print(f"\n📝 Testing Refiner Agent with {provider_name}...")
    
    refiner = RefinerAgent(client)
    
    component_spec = {
        "name": "LearningPathGenerator",
        "requirements": "Generate adaptive learning paths based on student mastery",
        "constraints": ["Must respect prerequisites", "Optimize for engagement"]
    }
    
    plan = await refiner.generate_test_plan(component_spec)
    
    print(f"  ✓ Generated {len(plan.tests)} tests")
    print(f"  ✓ Estimated coverage: {plan.estimated_coverage:.0%}")
    for test in plan.tests[:3]:
        print(f"    - {test.name}: {test.description[:40]}...")
    
    return len(plan.tests) > 0


def test_topological_auditor():
    """Test Topological Auditor (no LLM needed)"""
    print(f"\n🕸️ Testing Topological Auditor...")
    
    auditor = TopologicalAuditor()
    
    # Valid DAG
    valid_nodes = [
        {"id": "algebra", "prerequisites": []},
        {"id": "calculus", "prerequisites": ["algebra"]},
        {"id": "physics", "prerequisites": ["calculus"]}
    ]
    
    auditor.load_graph(valid_nodes)
    result = auditor.audit()
    
    print(f"  ✓ Valid DAG: {result.is_valid_dag}")
    print(f"  ✓ Nodes: {result.node_count}, Depth: {result.max_depth}")
    print(f"  ✓ Connectivity: {result.connectivity_score:.0%}")
    
    # Invalid DAG (cycle)
    invalid_nodes = [
        {"id": "A", "prerequisites": ["C"]},
        {"id": "B", "prerequisites": ["A"]},
        {"id": "C", "prerequisites": ["B"]}
    ]
    
    auditor.load_graph(invalid_nodes)
    result = auditor.audit()
    
    print(f"  ✓ Cycle detected: {not result.is_valid_dag}")
    print(f"  ✓ Violations: {len(result.violations)}")
    
    return True


async def run_provider_tests(provider: LLMProvider):
    """Run all tests for a specific provider"""
    print(f"\n{'='*60}")
    print(f"🚀 DEPLOYING AGENTS WITH {provider.value.upper()}")
    print(f"{'='*60}")
    
    config = LLMConfig(provider=provider)
    
    # Load API key from environment
    if provider == LLMProvider.CLAUDE:
        config.api_key = os.getenv("ANTHROPIC_API_KEY")
        config.model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    elif provider == LLMProvider.GEMINI:
        config.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        config.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    if provider != LLMProvider.MOCK and not config.api_key:
        print(f"  ⚠️ No API key found for {provider.value}. Using mock client.")
        client = MockLLMClient()
    else:
        client = create_llm_client(config)
    
    results = []
    
    # Run tests
    results.append(("Verifier", await test_verifier(client, provider.value)))
    results.append(("Adversarial Peer", await test_adversarial_peer(client, provider.value)))
    results.append(("Refiner", await test_refiner(client, provider.value)))
    results.append(("Topological Auditor", test_topological_auditor()))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 DEPLOYMENT SUMMARY ({provider.value.upper()})")
    print(f"{'='*60}")
    
    all_passed = True
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False
    
    return all_passed


async def main():
    parser = argparse.ArgumentParser(description="Deploy Agentic Testing Agents")
    parser.add_argument(
        "--provider", 
        choices=["claude", "gemini", "mock"],
        default="mock",
        help="LLM provider to use"
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Test all available providers"
    )
    
    args = parser.parse_args()
    
    if args.test_all:
        # Test mock first (always works)
        await run_provider_tests(LLMProvider.MOCK)
        
        # Test real providers if keys available
        if os.getenv("ANTHROPIC_API_KEY"):
            await run_provider_tests(LLMProvider.CLAUDE)
        else:
            print("\n⚠️ Skipping Claude (no ANTHROPIC_API_KEY)")
        
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            await run_provider_tests(LLMProvider.GEMINI)
        else:
            print("\n⚠️ Skipping Gemini (no GOOGLE_API_KEY)")
    else:
        provider = LLMProvider(args.provider)
        await run_provider_tests(provider)


if __name__ == "__main__":
    asyncio.run(main())
