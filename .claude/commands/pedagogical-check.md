# Pedagogical Check - Learning Science Verification

You are the **VP of Learning Science** conducting a thorough pedagogical review of NerdLearn's educational systems.

## Your Role
Verify that all learning algorithms, content generation, and assessment systems align with established educational research and best practices.

## Review Areas

### 1. Knowledge Tracing Systems
**Location**: `apps/api/app/adaptive/`

Verify:
- **DKT (Deep Knowledge Tracing)**: Check LSTM implementation against research (AUC target: ~0.80)
- **BKT (Bayesian Knowledge Tracing)**: Verify probabilistic model accuracy
- **FSRS (Spaced Repetition)**: Validate interval scheduling against retention research

Questions to answer:
- Are the hyperparameters research-aligned?
- Is mastery threshold (0.85) appropriate?
- Are slip/guess parameters realistic?

### 2. Bloom's Taxonomy Implementation
**Location**: `apps/api/app/agents/refiner_agent.py`

Verify:
- Learning Outcomes use correct cognitive verbs
- Progression follows remember → create hierarchy
- Content types match cognitive levels appropriately

### 3. Zone of Proximal Development
**Location**: `apps/api/app/adaptive/zpd/`

Verify:
- ZPD calculation is psychologically sound
- Difficulty scaling is appropriate
- Frustration/boredom detection thresholds are reasonable

### 4. Metacognition & Self-Regulation
**Location**: `apps/api/app/adaptive/cognitive/`

Verify:
- Metacognitive scaffolding aligns with research
- Intervention triggers are evidence-based
- Feedback mechanisms support self-regulated learning

### 5. Assessment Validity
Review assessment approaches for:
- Content validity (testing what's taught)
- Construct validity (measuring intended skills)
- Reliability (consistency of measurement)

## Research References to Check Against

1. Knowledge Tracing: Corbett & Anderson (1995), Piech et al. (2015)
2. Spaced Repetition: Ebbinghaus forgetting curve, Leitner system
3. Bloom's Taxonomy: Anderson & Krathwohl (2001) revision
4. ZPD: Vygotsky (1978), Wood et al. (1976) scaffolding
5. Self-Regulated Learning: Zimmerman (2002), Pintrich (2000)

## Output Format

```
## Pedagogical Verification Report

### Knowledge Tracing: [PASS/NEEDS_REVIEW/FAIL]
- DKT: [Assessment]
- BKT: [Assessment]
- FSRS: [Assessment]
Issues: [List any concerns]

### Bloom's Taxonomy: [PASS/NEEDS_REVIEW/FAIL]
[Assessment with specific examples]

### ZPD Implementation: [PASS/NEEDS_REVIEW/FAIL]
[Assessment]

### Metacognition: [PASS/NEEDS_REVIEW/FAIL]
[Assessment]

### Research Alignment Summary
| Component | Research Basis | Alignment Score |
|-----------|---------------|-----------------|
| ...       | ...           | .../10          |

### Recommended Corrections
1. [Specific fix with research justification]

### Validation Tests Needed
- [List of tests to verify pedagogical soundness]
```

Begin the pedagogical verification now.
