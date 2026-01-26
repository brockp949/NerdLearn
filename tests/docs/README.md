# NerdLearn Documentation & Research

This folder contains all project documentation, research papers, and planning materials organized by category.

---

## Folder Structure

```
docs/
├── phases/                    # Implementation phase summaries
│   ├── PHASE_2_SUMMARY.md     # NotebookLM Ingestion Pipeline
│   ├── PHASE_3_SUMMARY.md     # Adaptive Engine
│   └── PHASE_4_SUMMARY.md     # Learning Interface
│
├── planning/                  # Project planning documents
│   ├── BONUS_FEATURES_PLAN.md # 24 future features with priority matrix
│   └── VERIFICATION_REPORT.md # Implementation verification report
│
├── operations/                # Deployment & operations
│   ├── DEPLOYMENT.md          # Production deployment guide
│   └── PRODUCTION_OPS.md      # Operations & maintenance
│
├── tracks/                    # Conductor tracking system
│   ├── tracks.md              # Central coordination hub
│   ├── 001-research-ingestion/# PDF ingestion track
│   ├── 002-causal-discovery/  # Causal discovery track (complete)
│   ├── 003-gamification/      # Gamification track (complete)
│   └── 004-frontend-phase4/   # Frontend phase 4 track
│
└── research/                  # Academic research papers (40+ PDFs)
    ├── adaptive-learning/     # Adaptive learning systems
    ├── algorithms/            # ML/AI algorithms (DKT, MAB, FSRS, etc.)
    ├── architecture/          # System architecture & infrastructure
    ├── assessment/            # Stealth assessment & psychometrics
    ├── cognitive-science/     # Cognitive load, ZPD, dual-process theory
    ├── gamification/          # Gamification & motivation
    ├── implicit-feedback/     # Telemetry & behavioral analysis
    ├── knowledge-graphs/      # Knowledge graph construction
    ├── practice-methods/      # Interleaving, spaced repetition
    └── misc/                  # Other research materials
```

---

## Quick Navigation

### By Purpose

| Need | Location |
|------|----------|
| Understand implementation | `phases/PHASE_*_SUMMARY.md` |
| Plan future work | `planning/BONUS_FEATURES_PLAN.md` |
| Track current work | `tracks/tracks.md` |
| Deploy to production | `operations/DEPLOYMENT.md` |
| Research algorithms | `research/algorithms/` |
| Research pedagogy | `research/cognitive-science/` |

### By Phase

| Phase | Summary | Status |
|-------|---------|--------|
| Phase 2 | NotebookLM Ingestion | Complete |
| Phase 3 | Adaptive Engine | Complete |
| Phase 4 | Learning Interface | Complete |

### By Track

| Track | Description | Status |
|-------|-------------|--------|
| 001 | Research PDF Ingestion | Pending |
| 002 | Causal Discovery | Complete |
| 003 | Gamification | Complete |
| 004 | Frontend Phase 4 | In Progress |

---

## Research Categories

### Adaptive Learning (`research/adaptive-learning/`)
Papers on adaptive learning systems, curriculum adaptation, and personalized learning.

### Algorithms (`research/algorithms/`)
Technical papers on:
- Deep Knowledge Tracing (DKT)
- Multi-Armed Bandits
- Spaced Repetition (FSRS)
- Neural ODEs for memory decay
- Reinforcement learning for interleaving

### Architecture (`research/architecture/`)
System design papers:
- Microservices architecture
- PostgreSQL unified database
- Privacy-preserving ML
- Generative UI

### Assessment (`research/assessment/`)
Assessment methodology:
- Stealth assessment
- Evidence-centered design
- Psychometric foundations

### Cognitive Science (`research/cognitive-science/`)
Learning theory:
- Cognitive load theory
- Zone of Proximal Development (ZPD)
- Dual-process theory
- Mental models

### Gamification (`research/gamification/`)
Motivation & engagement:
- Octalysis framework
- Age-appropriate mechanics
- Variable reward schedules

### Knowledge Graphs (`research/knowledge-graphs/`)
Graph-based learning:
- Knowledge graph construction
- GraphRAG community detection
- Causal discovery

---

## Adding New Documentation

1. **Phase summaries** → `phases/`
2. **Planning docs** → `planning/`
3. **Operations guides** → `operations/`
4. **Track plans** → `tracks/<track-number>-<name>/plan.md`
5. **Research papers** → `research/<category>/`

---

## Related

- [Testing Guide](../README.md) - How to run tests
- [Main README](../../README.md) - Project overview
