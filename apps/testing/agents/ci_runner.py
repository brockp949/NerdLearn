"""
CI Runner for Architect Agent

Script to run Architect Agent in CI/CD pipeline.
"""

import asyncio
import sys
import argparse
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from apps.testing.agents.architect_agent import ArchitectAgent


class MockLLMClient:
    """Mock LLM for CI testing"""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Return mock review"""
        # In production, use actual LLM client
        return '''```json
{
    "approved": true,
    "confidence": 0.85,
    "concerns": [],
    "architectural_impact": "Adds new testing infrastructure. No breaking changes to existing services.",
    "follow_cable_violations": [],
    "suggestions": ["Consider adding integration tests for new components"]
}
```'''


async def run_review(diff_file: str, pr_number: int):
    """
    Run architect review on PR.
    
    Args:
        diff_file: Path to file containing git diff
        pr_number: GitHub PR number
    """
    
    # Read diff
    with open(diff_file, 'r', encoding='utf-8') as f:
        diff = f.read()
    
    # Initialize agent
    # TODO: Replace with actual LLM client
    llm_client = MockLLMClient()
    architect = ArchitectAgent(llm_client)
    
    # Run review
    result = await architect.review_pr(
        diff=diff,
        pr_description=f"Pull Request #{pr_number}",
        pr_title=f"PR #{pr_number}"
    )
    
    # Output results
    print("=" * 80)
    print("ARCHITECT AGENT REVIEW")
    print("=" * 80)
    print(result.to_markdown_comment())
    print("=" * 80)
    
    # Exit with appropriate code
    if not result.approved:
        print(f"\n❌ REVIEW FAILED: {', '.join(result.concerns)}")
        sys.exit(1)
    elif result.requires_human_review:
        print(f"\n⚠️  HUMAN REVIEW REQUIRED (Risk: {result.risk_level})")
        sys.exit(2)
    else:
        print(f"\n✅ REVIEW PASSED (Confidence: {result.confidence:.0%})")
        sys.exit(0)


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description='Run Architect Agent review')
    parser.add_argument('--diff', required=True, help='Path to diff file')
    parser.add_argument('--pr-number', type=int, required=True, help='PR number')
    
    args = parser.parse_args()
    
    asyncio.run(run_review(args.diff, args.pr_number))


if __name__ == "__main__":
    main()
