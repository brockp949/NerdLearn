# Agentic Testing Developer Guide

This guide explains how to use the NerdLearn Agentic Testing Framework.

## 🎯 Core Concepts

The system uses AI agents to test the codebase, focusing on **semantic correctness** and **architectural integrity** rather than just syntax.

### The 4 Agents (Role Matrix)

1. **The Verifier** (`VerifierAgent`): Checks "Semantic Truth".
   - *Use for*: Checking LLM outputs, RAG retrievals, and entity extraction.
   - *Key Concept*: **Gravitational Biasing** - Ensures the test stays focused on the goal.

2. **The Peer** (`AdversarialPeer`): Chaos Monkey for resilience.
   - *Use for*: Stress testing, checking handling of conflicting data, robustness.
   - *Key Concept*: **Poisoned Data** - Deliberately injecting edge cases.

3. **The Architect** (`ArchitectAgent`): Validates Topology.
   - *Use for*: CI/CD PR reviews, checking curriculum graph structures.
   - *Key Concept*: **Follow the Cable** - Ensuring causal chains are intact.

4. **The Refiner** (`RefinerAgent`): TDFlow Test Generation.
   - *Use for*: Generating test plans BEFORE writing implementation code.
   - *Key Concept*: **Coverage Optimization** - Finding edge cases upfront.

---

## 🚀 How to Write Agentic Tests

### 1. Semantic Verification (Replacing Assertions)

Instead of checking exact string matches, use the `VerifierAgent` to check meaning.

```python
from apps.testing.agents import VerifierAgent, GoalVector

# 1. Define your Goal Vector (The "Intention")
goal = GoalVector(
    name="Helpful Tutor Response",
    description="Response should be encouraging but correct the error",
    constraints=["No harsh criticism", "Explain the concept", "Provide example"],
    pass_threshold=0.8
)

# 2. Run your code
response = ai_tutor.reply("2 + 2 = 5")

# 3. Verify semantically
verifier = VerifierAgent(goal, llm_client)
result = await verifier.verify(response)

assert result.passed
```

### 2. Adversarial Testing (Robustness)

Don't just test the happy path. Test the "Poisoned" path.

```python
from apps.testing.agents import AdversarialPeer, PoisonType

# 1. Generate unique edge cases
peer = AdversarialPeer(llm_client)
scenarios = await peer.generate_poisoned_data(
    domain="math_problems", 
    poison_type=PoisonType.SEMANTIC_TRAP
)

# 2. Run them against your system
for scenario in scenarios:
    result = solver.solve(scenario.input)
    # Ensure system didn't crash or hallucinate
    assert result.is_safe
```

---

## 🌊 TDFlow (Test-Driven Flow)

**Rule**: Before implementing a new feature, generate the test plan.

```python
from apps.testing.agents import RefinerAgent

refiner = RefinerAgent(llm_client)
plan = await refiner.generate_test_plan({
    "name": "NewFeature",
    "requirements": "..."
})

print(plan.tests) # <- Implement these!
```

---

## ⛽ Fuel Meter (Cost Control)

Always use the fuel meter when running agents in loops or background jobs.

```python
from apps.testing.agents import FuelMeter, FuelBudget

budget = FuelBudget(max_tokens=5000)
meter = FuelMeter(budget)

with meter:
    await run_complex_agent_task()
```

---

## 🏗️ Topology Audits

If you modify the Curriculum or Knowledge Graph structure, you MUST run a topological audit.

```bash
# Run locally
python -m apps.testing.agents.tests.example_topological_test
```

---

## ❓ Troubleshooting

- **Verification Failures**: Check the `result.reasoning` field. The agent usually explains why it failed.
- **Recursion Errors**: Typical in cyclic graphs. The Topological Auditor catches these.
- **Budget Exhaustion**: Increase `FuelBudget` limits or optimize prompt size.

## 📚 Reference

- **Design Doc**: `brain/.../implementation_plan.md`
- **Walkthrough**: `brain/.../walkthrough.md`
