# 🧠 NerdLearn: Cognitive-Adaptive Learning Platform

<div align="center">

**Not just an LMS. A Cognitive Operating System.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)](https://www.typescriptlang.org/)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black)](https://nextjs.org/)

</div>

## 🎯 What is NerdLearn?

NerdLearn is a **research-backed, AI-powered learning platform** that adapts to your cognitive state in real-time. Unlike traditional LMS platforms that simply deliver content, NerdLearn acts as an **external regulator of learner cognition**, maintaining optimal challenge levels and maximizing learning efficiency.

### The Problem with Traditional Learning

- **Fixed difficulty**: Everyone gets the same content regardless of skill level
- **Disruptive testing**: Learning interrupted by assessments
- **Inefficient scheduling**: Generic review intervals ignore individual memory
- **Demotivating gamification**: Points and badges feel hollow
- **One-size-fits-all**: No adaptation to cognitive load or learning state

### The NerdLearn Solution

**Sub-100ms Cognitive Adaptation** powered by:
- 🧠 **Deep Knowledge Tracing** (SAINT+ architecture)
- 🎯 **Zone of Proximal Development Regulation**
- 📊 **Stealth Assessment** (Evidence-Centered Design)
- 🔄 **FSRS Scheduling** (99.6% more efficient than Anki)
- 🌳 **Knowledge Graph Navigation**
- 🎮 **Intrinsic Gamification** (Octalysis framework)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Next.js 14 Frontend                      │
│                    (React Server Components)                 │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ├──────────────┬──────────────┬──────────────┐
                 │              │              │              │
        ┌────────▼─────┐ ┌─────▼──────┐ ┌────▼─────────┐ ┌──▼────────┐
        │  Scheduler   │ │ Telemetry  │ │  Inference   │ │  Content  │
        │   Service    │ │  Service   │ │   Engine     │ │ Ingestion │
        │   (FSRS)     │ │  (Stealth) │ │  (DKT/ZPD)   │ │  Service  │
        └──────┬───────┘ └─────┬──────┘ └──────┬───────┘ └─────┬─────┘
               │               │               │               │
        ┌──────▼───────────────▼───────────────▼───────────────▼─────┐
        │                     Data Layer                              │
        ├─────────────┬────────────────┬───────────────┬─────────────┤
        │ PostgreSQL  │    Neo4j       │  TimescaleDB  │   Milvus    │
        │ (Relational)│ (Knowledge     │ (Time-series) │  (Vector)   │
        │             │  Graph)        │               │             │
        └─────────────┴────────────────┴───────────────┴─────────────┘
```

### Microservices

#### 1. **Scheduler Service** (Port 8001)
- **Algorithm**: FSRS (Free Spaced Repetition Scheduler)
- **Function**: Optimal spacing of review intervals
- **Tech**: FastAPI + Redis
- **Performance**: 99.6% more efficient than SM-2 (Anki)

#### 2. **Telemetry Service** (Port 8002)
- **Algorithm**: Evidence-Centered Design (ECD)
- **Function**: Stealth assessment via behavioral signals
- **Tech**: FastAPI + Redpanda (Kafka) + WebSockets
- **Latency**: <100ms event processing

#### 3. **Inference Engine** (Port 8003)
- **Algorithms**:
  - SAINT+ (Deep Knowledge Tracing)
  - ZPD Regulator (Vygotsky's theory)
- **Function**: Predicts knowledge state, regulates difficulty
- **Tech**: FastAPI + PyTorch
- **Model**: Transformer-based sequence modeling

### Databases

| Database | Purpose | Data |
|----------|---------|------|
| **PostgreSQL** | Relational data | Users, courses, enrollments |
| **Neo4j** | Knowledge Graph | Concepts, prerequisites, relationships |
| **TimescaleDB** | Time-series | Behavioral logs (mouse, clicks) |
| **Milvus** | Vector DB | Semantic embeddings for RAG |
| **Redis** | Cache/State | Session data, card states |

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+
- **Python** 3.11+
- **Docker** & **Docker Compose**
- **Git**

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/NerdLearn.git
cd NerdLearn

# Start all databases
docker-compose up -d

# Install dependencies
npm install

# Set up database
cd packages/db
cp .env.example .env
npm run db:push
cd ../..

# Install Python dependencies for services
cd services/scheduler
pip install -r requirements.txt
cd ../telemetry
pip install -r requirements.txt
cd ../inference
pip install -r requirements.txt
cd ../..

# Start development (all services)
npm run dev
```

### Development URLs

- **Frontend**: http://localhost:3000
- **Scheduler API**: http://localhost:8001/docs
- **Telemetry API**: http://localhost:8002/docs
- **Inference API**: http://localhost:8003/docs
- **Neo4j Browser**: http://localhost:7474
- **PostgreSQL**: localhost:5432

---

## 📚 Core Algorithms Explained

### 1. FSRS (Free Spaced Repetition Scheduler)

**Purpose**: Optimize review intervals to maximize retention with minimum study time.

**How it works**:
```python
# Core formula
S_new = S_old × (1 + e^(w8) × (11 - D) × S^(-w9) × (e^(w10 × (1 - R)) - 1))

# Where:
# S = Stability (days until retrievability drops to 90%)
# D = Difficulty (1-10)
# R = Retrievability (current probability of recall)
# w8-w10 = Learned weight parameters
```

**Why it's better than SM-2/Anki**:
- Uses **stochastic model** instead of fixed formulas
- Considers **retrievability** at review time
- **Self-optimizing** through parameter learning
- Research shows **99.6% efficiency improvement**

### 2. Deep Knowledge Tracing (DKT)

**Purpose**: Model learner knowledge evolution over time.

**Architecture**: SAINT+ (Separated Self-Attention)
```
Input: (exercise_t, response_t)
      ↓
Exercise Embedding + Response Embedding + Position Encoding
      ↓
Transformer Encoder (Multi-Head Attention)
      ↓
Output: P(correct | exercise_t+1)
```

**Advantages over traditional IRT**:
- Captures **temporal dependencies** in learning
- Models **non-monotonic** knowledge change (forgetting)
- Handles **sparse data** better
- **Transfer learning** across concepts

### 3. ZPD Regulator (Cognitive Thermostat)

**Purpose**: Maintain learner in "Goldilocks Zone" of challenge.

**Target Success Rate**: 35-70%

**Decision Logic**:
```
Success Rate < 35%  → FRUSTRATION ZONE
  Actions:
  - Provide scaffolding (worked examples)
  - Review prerequisites
  - Reduce difficulty
  - Enable blocked practice

Success Rate 35-70% → OPTIMAL ZONE (ZPD)
  Actions:
  - Maintain current difficulty
  - Fade scaffolding gradually
  - No major changes

Success Rate > 70%  → COMFORT ZONE
  Actions:
  - Remove all scaffolding
  - Increase difficulty
  - Introduce interleaving
  - Advance to next concept
```

### 4. Stealth Assessment (Evidence-Centered Design)

**Purpose**: Assess competency without explicit tests.

**Evidence Sources**:

| Signal | What it Measures | Threshold |
|--------|------------------|-----------|
| **Dwell Time** | Engagement validity | 60% of expected reading time |
| **Reading Rate** | Comprehension mode | 150-700 WPM = valid |
| **Mouse Velocity** | Cognitive load | Low velocity = high load |
| **Trajectory Entropy** | Confusion | High entropy = chaotic |
| **Saccade Count** | Uncertainty | High count = struggling |

---

## 📁 Project Structure

```
NerdLearn/
├── apps/
│   └── web/                    # Next.js 14 frontend
│       ├── src/
│       │   ├── app/           # App Router pages
│       │   ├── components/    # React components
│       │   └── lib/           # Utilities
│       └── package.json
│
├── services/
│   ├── scheduler/              # FSRS scheduling service
│   │   ├── main.py            # FastAPI app
│   │   ├── scheduler.py       # FSRS algorithm
│   │   └── requirements.txt
│   │
│   ├── telemetry/             # Stealth assessment service
│   │   ├── main.py            # FastAPI app
│   │   └── requirements.txt
│   │
│   └── inference/             # Knowledge Tracing + ZPD
│       ├── main.py            # FastAPI app
│       ├── dkt_model.py       # SAINT+ implementation
│       ├── zpd_regulator.py   # ZPD logic
│       └── requirements.txt
│
├── packages/
│   └── db/                     # Shared database package
│       ├── prisma/
│       │   └── schema.prisma  # Database schema
│       └── src/
│           ├── index.ts       # Prisma client
│           └── neo4j.ts       # Neo4j client
│
├── docker-compose.yml          # Infrastructure
├── package.json               # Monorepo root
└── turbo.json                 # Turborepo config
```

---

## 🛣️ Roadmap

### Phase 1: Core Infrastructure ✅ (COMPLETED)
- [x] Monorepo setup
- [x] Database schemas (PostgreSQL + Neo4j)
- [x] FSRS scheduler service
- [x] Telemetry service (stealth assessment)
- [x] DKT/ZPD inference engine
- [x] Next.js frontend skeleton
- [x] Docker infrastructure

### Phase 2: Content Pipeline (Next)
- [ ] PDF/video ingestion
- [ ] Knowledge Graph construction
- [ ] Concept extraction (BERT NER)
- [ ] Prerequisite mining
- [ ] Difficulty scoring

### Phase 3: Full Integration
- [ ] End-to-end learning flow
- [ ] Real-time adaptation
- [ ] Gamification layer
- [ ] Analytics dashboard

---

## 📄 License

MIT License - see [LICENSE](./LICENSE) for details.

---

<div align="center">

**Built with 🧠 by the NerdLearn Team**

*Making learning more human by making it more intelligent.*

</div>
