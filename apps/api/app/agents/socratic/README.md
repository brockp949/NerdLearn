# Socratic Evaluation Agents (Phase 5)

This package contains agents designed to evaluate the pedagogical quality of the NerdLearn system.
It implements an automated "Student-Teacher-Judge" loop to ensure the system adheres to Socratic teaching methods.

## Components

### 1. Student Agent (`student_agent.py`)
- **Role**: Simulates a learner with a specific persona (e.g., "Curious Novice", "Frustrated Student").
- **Goal**: Interact with the system to elicit responses.
- **Model**: Uses a lighter/cheaper model (e.g. `gpt-4o-mini`) for efficiency.

### 2. Judge Agent (`judge_agent.py`)
- **Role**: Evaluates the transcript of the interaction.
- **Criteria**:
    - **Accuracy**: Fact checking.
    - **Socratic Adherence**: Did the system guide the user or just give the answer?
    - **Hallucination Detection**: Checks for made-up facts.
- **Model**: Uses a strong model (e.g. `gpt-4o`) for high-quality evaluation.

## Usage

The evaluation is run via the pipeline script:

```bash
# Run from root
python apps/api/tests/socratic_eval.py
```

This is also integrated into the CI/CD pipeline (`.github/workflows/ci.yml`) as the `socratic-eval` job.
