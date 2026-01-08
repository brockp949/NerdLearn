# NerdLearn

**AI-Powered Adaptive Learning Platform**

NerdLearn combines the best of Udemy's course management with NotebookLM's AI interactivity, powered by cutting-edge cognitive science. Learn smarter, not harder.

## Overview

NerdLearn is a full-stack learning platform that offers:

- **Udemy-Style Course Management**: Instructors can create, upload, and manage courses
- **NotebookLM AI Chat**: Chat with your course content using AI
- **Stealth Assessment**: Invisible learning analytics that track engagement
- **Adaptive Learning**: FSRS spaced repetition + Zone of Proximal Development optimization
- **Knowledge Graphs**: Visualize concept relationships and prerequisites
- **Audio Overviews**: AI-generated podcast-style course summaries

## Architecture

### Tech Stack

**Frontend (apps/web)**
- Next.js 15 (App Router)
- TypeScript
- TailwindCSS + Shadcn/UI
- Lucide React Icons
- Recharts for analytics
- React Dropzone for file uploads
- React Force Graph for knowledge visualization

**Backend (apps/api)**
- FastAPI (Python)
- SQLAlchemy (async)
- PostgreSQL (user/course metadata)
- Neo4j (knowledge graphs)
- Qdrant (vector embeddings)
- Redis (caching/sessions)
- MinIO (file storage)

**AI/ML Stack**
- OpenAI GPT (chat, content analysis)
- Whisper (video transcription)
- BERT (concept extraction)
- FSRS (spaced repetition)
- DKT (knowledge tracing)

## Project Structure

```
nerdlearn/
├── apps/
│   ├── web/              # Next.js frontend
│   │   ├── src/
│   │   │   ├── app/      # Next.js App Router pages
│   │   │   │   ├── studio/       # Instructor Studio
│   │   │   │   ├── learn/        # Learning interface
│   │   │   │   └── page.tsx      # Homepage
│   │   │   ├── components/       # React components
│   │   │   └── lib/              # Utilities & API client
│   │   └── package.json
│   │
│   ├── api/              # FastAPI backend
│   │   ├── app/
│   │   │   ├── core/             # Config, database
│   │   │   ├── models/           # SQLAlchemy models
│   │   │   ├── routers/          # API endpoints
│   │   │   ├── schemas/          # Pydantic schemas
│   │   │   ├── services/         # Business logic
│   │   │   └── main.py           # FastAPI app
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   └── worker/           # Background processing (Phase 2)
│
├── docker-compose.yml    # All infrastructure services
├── turbo.json            # Turborepo config
└── Makefile              # Development commands
```

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.11+
- Docker & Docker Compose

### Installation

1. **Clone the repository**
```bash
git clone <repo-url>
cd NerdLearn
```

2. **Install dependencies**
```bash
make install
# or manually:
npm install
cd apps/api && pip install -r requirements.txt
```

3. **Start infrastructure services**
```bash
make docker-up
# This starts: PostgreSQL, Neo4j, Qdrant, Redis, MinIO
```

4. **Configure environment**
```bash
cp apps/api/.env.example apps/api/.env
# Edit .env with your API keys
```

5. **Start development servers**
```bash
# Terminal 1: Backend
cd apps/api
uvicorn app.main:app --reload

# Terminal 2: Frontend
cd apps/web
npm run dev
```

6. **Access the platform**
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs
- MinIO Console: http://localhost:9001 (minioadmin/minioadmin)
- Neo4j Browser: http://localhost:7474 (neo4j/password)

## Development Status

### ✅ Phase 1: Core Infrastructure (COMPLETE)

**Phase 1.1: Monorepo & Services**
- ✅ Turborepo monorepo setup
- ✅ Next.js 15 frontend with TailwindCSS
- ✅ FastAPI backend with async support
- ✅ Docker Compose with 5 services (Postgres, Neo4j, Qdrant, Redis, MinIO)

**Phase 1.2: Instructor Studio**
- ✅ SQLAlchemy models (User, Course, Module, Instructor, Assessment)
- ✅ Full CRUD API for courses
- ✅ Module upload endpoints with MinIO integration
- ✅ Instructor Studio dashboard UI
- ✅ Drag-and-drop file upload (videos/PDFs)
- ✅ Course management interface

### ✅ Phase 2: NotebookLM Ingestion Pipeline (COMPLETE)

**Phase 2.1: Background Worker Infrastructure**
- ✅ Celery worker with Redis broker
- ✅ Task queues (documents, videos, processing)
- ✅ Docker integration with worker service
- ✅ Processing status tracking in database

**Phase 2.2: Content Processing**
- ✅ PDF processor with pdfplumber/PyPDF2
- ✅ Video transcription with OpenAI Whisper
- ✅ Semantic chunking with transformer tokenizers
- ✅ Vector storage with Qdrant embeddings
- ✅ Knowledge graph construction (Neo4j)
- ✅ Concept extraction and relationship detection

**Phase 2.3: Integration**
- ✅ API integration for triggering background tasks
- ✅ Processing status check endpoints
- ✅ Module metadata tracking (chunk count, concept count)
- ✅ Automatic processing on module upload

### ✅ Phase 3: Adaptive Engine (COMPLETE)

**Phase 3.1: FSRS Spaced Repetition**
- ✅ FSRS algorithm implementation with stability/difficulty tracking
- ✅ Spaced repetition card models and review logging
- ✅ Adaptive scheduling based on performance
- ✅ Next interval predictions for all rating options

**Phase 3.2: Stealth Assessment**
- ✅ WebSocket telemetry collector for behavioral data
- ✅ Evidence rules (dwell time, video engagement, chat queries)
- ✅ Real-time mastery updates from implicit feedback
- ✅ Multi-signal evidence aggregation

**Phase 3.3: Bayesian Knowledge Tracing**
- ✅ BKT probabilistic mastery estimation
- ✅ Bayesian updates from observations and evidence
- ✅ Performance prediction and mastery thresholds
- ✅ Sessions-to-mastery estimation

**Phase 3.4: Zone of Proximal Development**
- ✅ ZPD-based content difficulty regulation
- ✅ Prerequisite readiness checking
- ✅ Optimal challenge point calculation
- ✅ Content recommendations with success rate prediction
- ✅ Dynamic difficulty adjustment

### 🚧 Phase 4: Learning Interface (PENDING)

- [ ] Split-screen learning UI
- [ ] Context-aware chat
- [ ] Citation pills with video seeking
- [ ] Gamification (skill tree, streaks)

### 🚧 Phase 5: Production Ready (PENDING)

- [ ] Federated learning stub
- [ ] GitHub Actions CI/CD
- [ ] Deployment configurations

## API Endpoints

### Courses
- `GET /api/courses` - List all courses
- `POST /api/courses` - Create a course
- `GET /api/courses/{id}` - Get course details
- `PUT /api/courses/{id}` - Update course
- `DELETE /api/courses/{id}` - Delete course
- `POST /api/courses/{id}/publish` - Publish course

### Modules
- `GET /api/courses/{course_id}/modules` - List course modules
- `POST /api/courses/{course_id}/modules` - Upload module (multipart/form-data)
- `GET /api/modules/{id}` - Get module details
- `PUT /api/modules/{id}` - Update module
- `DELETE /api/modules/{id}` - Delete module

### Assessment (Phase 3)
- `WS /api/assessment/ws/telemetry` - WebSocket for stealth assessment
- `GET /api/reviews/due` - Get spaced repetition reviews

### Chat (Phase 4)
- `POST /api/chat` - Chat with course content

## Key Features

### 1. Instructor Studio
Instructors can:
- Create and manage courses
- Upload videos and PDFs with drag-and-drop
- Organize modules with custom ordering
- Track processing status (transcription, parsing)
- Publish/unpublish courses

### 2. Stealth Assessment (Phase 3)
- Tracks dwell time normalized by reading speed
- Click pattern analysis for preference learning
- Bayesian updates to mastery probability
- Zero-interruption learning flow

### 3. FSRS Spaced Repetition (Phase 3)
- Implements FSRS algorithm (Free Spaced Repetition Scheduler)
- Tracks stability, difficulty, retrievability per concept
- Optimal review scheduling for 90% retention

### 4. Knowledge Graphs (Phase 2)
- Automatic concept extraction from content
- Association rule mining for prerequisite relationships
- Interactive graph visualization
- Manual instructor editing

### 5. Audio Overviews (Phase 2)
- AI-generated dialogue scripts (host + expert)
- Multi-speaker synthesis with ElevenLabs
- "Podcast mode" for on-the-go learning

## Database Schema

### PostgreSQL Tables
- `users` - User accounts
- `instructors` - Instructor profiles
- `courses` - Course metadata
- `modules` - Course modules (videos/PDFs)
- `enrollments` - User-course relationships
- `user_concept_mastery` - Stealth assessment data
- `spaced_repetition_cards` - FSRS review cards

### Neo4j Graph
- Nodes: Concepts
- Relationships: `PREREQUISITE`, `RELATED_TO`, `PART_OF`

### Qdrant Collections
- `course_chunks` - Semantic chunks with embeddings

## Contributing

This is an implementation of the research-backed architecture described in the project PDFs. Key papers:
- Stealth Assessment & Evidence-Centered Design
- FSRS Spaced Repetition Algorithm
- Knowledge Graph Construction (KnowEdu)
- Gamification (Octalysis Framework)

## License

MIT

## Roadmap

- **Q1 2026**: Complete Phase 2 (AI ingestion pipeline)
- **Q2 2026**: Complete Phase 3 (Adaptive engine)
- **Q3 2026**: Complete Phase 4 (Learning interface)
- **Q4 2026**: Production launch

---

Built with ❤️ using AI-native architecture and cognitive science principles.