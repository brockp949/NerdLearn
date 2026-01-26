# NerdLearn Bonus Features Implementation Plan

**Date:** 2026-01-21
**Branch:** claude/verify-plans-implementation-abJXU

This plan covers the "Known Limitations & Future Improvements" sections from all 3 phase summaries, excluding features that have already been implemented.

---

## Overview

| Category | Features | Priority |
|----------|----------|----------|
| Content Processing Enhancements | 10 features | High |
| Adaptive Learning Improvements | 5 features | High |
| Learning Interface Upgrades | 5 features | Medium |
| Infrastructure & DevOps | 4 features | Medium |

**Total: 24 bonus features to implement**

---

## Phase 1: Content Processing Enhancements

### 1.1 Named Entity Recognition (NER) for Concept Extraction
**Source:** PHASE_2_SUMMARY.md
**Priority:** High
**Effort:** Medium

**Current State:** Simple heuristic-based extraction (capitalized phrases + technical terms)

**Implementation:**
```
apps/worker/app/processors/
├── ner_processor.py          # New: SpaCy/HuggingFace NER
└── concept_extractor.py      # New: ML-based concept extraction
```

**Tasks:**
- [ ] Install SpaCy and download en_core_web_lg model
- [ ] Create NER processor with entity type filtering (ORG, PRODUCT, WORK_OF_ART, etc.)
- [ ] Implement custom NER model fine-tuning for educational content
- [ ] Add technical term detection using domain-specific dictionaries
- [ ] Integrate with existing graph_tasks.py
- [ ] Add confidence scores for extracted concepts
- [ ] Create evaluation script to compare vs current heuristic approach

**Expected Outcome:** 92.8% F1 score (per research guidelines) vs ~70% current

---

### 1.2 Explicit Prerequisite Declarations
**Source:** PHASE_2_SUMMARY.md
**Priority:** High
**Effort:** Low

**Current State:** Prerequisite detection based on module order only

**Implementation:**
```
apps/api/app/models/
└── course.py                 # Add prerequisite_module_ids field

apps/api/app/routers/
└── modules.py                # Add prerequisite management endpoints
```

**Tasks:**
- [ ] Add `prerequisite_module_ids` JSON field to Module model
- [ ] Create `POST /api/modules/{id}/prerequisites` endpoint
- [ ] Create `DELETE /api/modules/{id}/prerequisites/{prereq_id}` endpoint
- [ ] Update ZPD regulator to use explicit prerequisites
- [ ] Add UI in Studio for prerequisite drag-drop editing
- [ ] Implement prerequisite cycle detection (prevent circular dependencies)

---

### 1.3 Task Retry with Exponential Backoff
**Source:** PHASE_2_SUMMARY.md
**Priority:** High
**Effort:** Low

**Current State:** No retry logic for failed tasks

**Implementation:**
```python
# apps/worker/app/celery_app.py
@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,  # 10 minutes max
    retry_jitter=True,
    max_retries=5
)
def process_pdf(self, module_id: int):
    ...
```

**Tasks:**
- [ ] Add Celery retry decorators to all tasks
- [ ] Implement exponential backoff (2^n seconds)
- [ ] Add jitter to prevent thundering herd
- [ ] Create dead letter queue for permanently failed tasks
- [ ] Add retry count to processing status endpoint
- [ ] Send notification on final failure

---

### 1.4 Audio Overview Generation (ElevenLabs/OpenAI TTS)
**Source:** PHASE_2_SUMMARY.md
**Priority:** Medium
**Effort:** Medium

**Current State:** No audio summaries

**Implementation:**
```
apps/worker/app/processors/
└── audio_generator.py        # New: TTS audio overview generator

apps/worker/app/tasks/
└── audio_tasks.py            # New: Audio generation task
```

**Tasks:**
- [ ] Create audio generator service (ElevenLabs or OpenAI TTS)
- [ ] Implement summarization pipeline for audio scripts
- [ ] Generate module overviews (2-3 minute summaries)
- [ ] Create course-level audio introductions
- [ ] Store audio files in MinIO
- [ ] Add audio_url field to Module model
- [ ] Create audio player component in frontend

---

### 1.5 Additional File Format Support
**Source:** PHASE_2_SUMMARY.md
**Priority:** Medium
**Effort:** Medium

**Current State:** PDF and video only

**Implementation:**
```
apps/worker/app/processors/
├── docx_processor.py         # New: Word documents
├── pptx_processor.py         # New: PowerPoint
├── epub_processor.py         # New: E-books
└── markdown_processor.py     # New: Markdown files
```

**Tasks:**
- [ ] Install python-docx for DOCX processing
- [ ] Install python-pptx for PPTX processing (with slide image extraction)
- [ ] Install ebooklib for EPUB processing
- [ ] Create unified processor interface
- [ ] Add file type detection and routing
- [ ] Update Module model to support new types
- [ ] Add file type icons in frontend

---

### 1.6 Batch Processing for Courses
**Source:** PHASE_2_SUMMARY.md
**Priority:** Medium
**Effort:** Low

**Current State:** Single module processing only

**Implementation:**
```
apps/api/app/routers/
└── courses.py                # Add batch processing endpoint

apps/worker/app/tasks/
└── batch_tasks.py            # New: Course-level batch processing
```

**Tasks:**
- [ ] Create `POST /api/courses/{id}/process-all` endpoint
- [ ] Implement Celery group for parallel module processing
- [ ] Add course-level processing status
- [ ] Create progress aggregation endpoint
- [ ] Add "Process All" button in Studio UI

---

### 1.7 Embedding Cache
**Source:** PHASE_2_SUMMARY.md
**Priority:** Low
**Effort:** Low

**Current State:** Re-generates embeddings on every reindex

**Implementation:**
```
apps/worker/app/services/
└── embedding_cache.py        # New: Redis-based embedding cache
```

**Tasks:**
- [ ] Create Redis-based embedding cache (hash of text → embedding)
- [ ] Add cache lookup before OpenAI API call
- [ ] Implement cache invalidation on content change
- [ ] Add cache hit rate metrics
- [ ] Set TTL for embeddings (30 days)

---

### 1.8 Quality Metrics
**Source:** PHASE_2_SUMMARY.md
**Priority:** Low
**Effort:** Medium

**Current State:** No quality scoring for processed content

**Implementation:**
```
apps/worker/app/processors/
└── quality_analyzer.py       # New: Quality metrics calculator

apps/api/app/models/
└── course.py                 # Add quality_metrics JSON field
```

**Tasks:**
- [ ] Implement transcript confidence scoring (Whisper word-level confidence)
- [ ] Add chunk coherence scoring (semantic similarity within chunks)
- [ ] Calculate readability scores (Flesch-Kincaid, SMOG)
- [ ] Detect potential OCR errors in PDFs
- [ ] Add quality score to Module model
- [ ] Create quality dashboard in Studio
- [ ] Flag low-quality content for review

---

## Phase 2: Adaptive Learning Improvements

### 2.1 FSRS Parameter Optimization
**Source:** PHASE_3_SUMMARY.md
**Priority:** High
**Effort:** High

**Current State:** Default FSRS parameters for all users

**Note:** FSRS Parameter Learning is already implemented in `apps/api/app/adaptive/fsrs/parameter_learning.py`

**Remaining Tasks:**
- [ ] Create scheduled job to run parameter optimization weekly
- [ ] Add A/B test framework for retention targets (85% vs 90% vs 95%)
- [ ] Implement domain-specific parameter presets (math, language, programming)
- [ ] Create parameter analytics dashboard
- [ ] Add user preference for retention target

---

### 2.2 ML-Based Evidence Rules
**Source:** PHASE_3_SUMMARY.md
**Priority:** High
**Effort:** High

**Current State:** Heuristic evidence rules (DwellTime, VideoEngagement, ChatQuery)

**Implementation:**
```
apps/api/app/adaptive/stealth/
├── ml_evidence_model.py      # New: Neural evidence predictor
└── evidence_trainer.py       # New: Model training pipeline
```

**Tasks:**
- [ ] Collect training data from existing telemetry logs
- [ ] Train neural classifier on engagement patterns
- [ ] Implement per-user evidence weight personalization
- [ ] Create real-time inference service
- [ ] Add model versioning and rollback
- [ ] Implement online learning for continuous improvement
- [ ] Create A/B test comparing ML vs heuristic rules

---

### 2.3 Automated Prerequisite Detection
**Source:** PHASE_3_SUMMARY.md
**Priority:** Medium
**Effort:** High

**Current State:** Manual prerequisite definition

**Implementation:**
```
apps/api/app/adaptive/prerequisites/
├── prerequisite_detector.py  # New: ML-based prerequisite learning
└── graph_learner.py          # New: Curriculum graph algorithms
```

**Tasks:**
- [ ] Implement co-occurrence pattern analysis
- [ ] Use knowledge graph embeddings (TransE/RotatE)
- [ ] Detect implicit prerequisites from failure patterns
- [ ] Create prerequisite suggestion system
- [ ] Add confidence scores for detected prerequisites
- [ ] Implement manual override capability
- [ ] Create visualization for suggested vs confirmed prerequisites

---

### 2.4 Advanced ZPD (Multi-Dimensional Difficulty)
**Source:** PHASE_3_SUMMARY.md
**Priority:** Medium
**Effort:** Medium

**Current State:** Single difficulty dimension

**Implementation:**
```
apps/api/app/adaptive/zpd/
├── multi_dim_difficulty.py   # New: Multi-dimensional difficulty model
└── emotional_state.py        # New: Engagement/frustration detection
```

**Tasks:**
- [ ] Define difficulty dimensions (conceptual, procedural, factual, metacognitive)
- [ ] Create per-dimension mastery tracking
- [ ] Implement Bloom's taxonomy level detection for content
- [ ] Add frustration detection from telemetry patterns
- [ ] Implement engagement scoring
- [ ] Create difficulty adjustment per dimension
- [ ] Update recommendations to balance dimensions

---

### 2.5 Federated Learning for Privacy-Preserving Updates
**Source:** PHASE_3_SUMMARY.md
**Priority:** Low
**Effort:** Very High

**Current State:** Centralized model training

**Note:** Differential Privacy is already implemented in `apps/api/app/privacy/differential_privacy.py`

**Implementation:**
```
apps/api/app/adaptive/federated/
├── federated_client.py       # New: On-device model updates
├── federated_server.py       # New: Model aggregation server
└── secure_aggregation.py     # New: Secure multi-party computation
```

**Tasks:**
- [ ] Implement federated averaging (FedAvg) algorithm
- [ ] Create secure aggregation protocol
- [ ] Add differential privacy to gradient updates
- [ ] Implement model compression for bandwidth efficiency
- [ ] Create client SDK for mobile/web
- [ ] Add model versioning and sync
- [ ] Implement privacy budget management

---

## Phase 3: Learning Interface Upgrades

### 3.1 ML-Based Citation Extraction
**Source:** PHASE_4_SUMMARY.md
**Priority:** Medium
**Effort:** Medium

**Current State:** Regex-based citation extraction

**Implementation:**
```
apps/api/app/chat/
├── citation_extractor.py     # New: ML citation extraction
└── citation_models.py        # New: Citation data models
```

**Tasks:**
- [ ] Train NER model for citation entity detection
- [ ] Implement multi-hop citation linking
- [ ] Add visual highlighting in PDF viewer
- [ ] Create citation relationship graph
- [ ] Implement "cite the citation" feature
- [ ] Add citation confidence scores

---

### 3.2 Social Gamification
**Source:** PHASE_4_SUMMARY.md
**Priority:** Medium
**Effort:** High

**Current State:** Individual gamification only

**Implementation:**
```
apps/api/app/models/
├── social.py                 # New: Friend, Challenge, StudyGroup models

apps/api/app/routers/
└── social.py                 # New: Social API endpoints

apps/web/src/app/social/
└── page.tsx                  # New: Social features page
```

**Tasks:**
- [ ] Create Friend model with request/accept flow
- [ ] Implement friend leaderboards
- [ ] Create Challenge model (1v1, group challenges)
- [ ] Add StudyGroup model with shared XP bonuses
- [ ] Implement group chat/discussion
- [ ] Add challenge notifications
- [ ] Create social feed of friend activities
- [ ] Implement privacy controls

---

### 3.3 Personalization Settings
**Source:** PHASE_4_SUMMARY.md
**Priority:** Low
**Effort:** Low

**Current State:** Fixed gamification parameters

**Implementation:**
```
apps/api/app/models/
└── user.py                   # Add preferences JSON field

apps/web/src/app/settings/
└── page.tsx                  # New: User settings page
```

**Tasks:**
- [ ] Add user preferences model (notification settings, difficulty preferences)
- [ ] Create custom achievement goals (daily XP target, weekly modules)
- [ ] Implement learning style quiz
- [ ] Add preferred content formats (video, text, audio)
- [ ] Create personalized dashboard layout
- [ ] Implement theme customization

---

### 3.4 A/B Testing Framework for Gamification
**Source:** PHASE_4_SUMMARY.md
**Priority:** Low
**Effort:** Medium

**Current State:** No experimentation framework

**Implementation:**
```
apps/api/app/experiments/
├── ab_framework.py           # New: A/B test framework
├── feature_flags.py          # New: Feature flag system
└── metrics_collector.py      # New: Experiment metrics
```

**Tasks:**
- [ ] Implement feature flag system
- [ ] Create user segmentation for experiments
- [ ] Add experiment assignment and tracking
- [ ] Implement statistical significance calculator
- [ ] Create experiment dashboard
- [ ] Add XP reward experiments
- [ ] Test different achievement thresholds
- [ ] Implement multi-armed bandit for auto-optimization

---

### 3.5 Analytics Dashboard
**Source:** PHASE_4_SUMMARY.md
**Priority:** Medium
**Effort:** Medium

**Current State:** Basic stats in profile

**Implementation:**
```
apps/web/src/app/analytics/
├── page.tsx                  # New: Analytics dashboard
├── engagement.tsx            # New: Engagement metrics
├── retention.tsx             # New: Retention analysis
└── learning-curves.tsx       # New: Learning curve visualization
```

**Tasks:**
- [ ] Create engagement heatmap (hours × days)
- [ ] Implement retention cohort analysis
- [ ] Add learning curve visualization per concept
- [ ] Create mastery progression charts
- [ ] Implement prediction of engagement drop-off
- [ ] Add instructor analytics view
- [ ] Export analytics to CSV/PDF

---

## Phase 4: Infrastructure & DevOps

### 4.1 Webhook Callbacks for Processing Status
**Source:** PHASE_2_SUMMARY.md
**Priority:** Medium
**Effort:** Low

**Tasks:**
- [ ] Add webhook_url field to Course/Module models
- [ ] Implement webhook delivery service
- [ ] Add retry logic for failed webhooks
- [ ] Create webhook signature verification
- [ ] Document webhook payload formats

---

### 4.2 Task Progress Updates
**Source:** PHASE_2_SUMMARY.md
**Priority:** Low
**Effort:** Low

**Tasks:**
- [ ] Implement Celery task progress updates via Redis
- [ ] Add progress percentage to processing status endpoint
- [ ] Create real-time progress WebSocket
- [ ] Add progress bar in frontend

---

### 4.3 Distributed Telemetry Collector
**Source:** PHASE_3_SUMMARY.md
**Priority:** Low
**Effort:** Medium

**Tasks:**
- [ ] Migrate telemetry collector to Redis-backed storage
- [ ] Implement message queue for high-volume telemetry
- [ ] Add horizontal scaling support
- [ ] Create telemetry aggregation service
- [ ] Implement debouncing at collector level

---

### 4.4 CI/CD and Monitoring
**Source:** All phases
**Priority:** High
**Effort:** Medium

**Tasks:**
- [ ] Set up GitHub Actions CI pipeline
- [ ] Add automated testing (pytest, jest)
- [ ] Implement database migrations with Alembic
- [ ] Add Docker image builds and registry
- [ ] Set up staging environment
- [ ] Implement logging aggregation (ELK/Loki)
- [ ] Add metrics collection (Prometheus)
- [ ] Create monitoring dashboards (Grafana)
- [ ] Set up alerting for errors/performance

---

## Implementation Priority Matrix

| Feature | Business Value | Effort | Priority Score |
|---------|---------------|--------|----------------|
| CI/CD and Monitoring | High | Medium | **P0** |
| Task Retry with Backoff | High | Low | **P0** |
| NER Concept Extraction | High | Medium | **P1** |
| ML-Based Evidence Rules | High | High | **P1** |
| FSRS Parameter Optimization | High | High | **P1** |
| Explicit Prerequisites | High | Low | **P1** |
| Social Gamification | Medium | High | **P2** |
| Analytics Dashboard | Medium | Medium | **P2** |
| Audio Overview Generation | Medium | Medium | **P2** |
| Additional File Formats | Medium | Medium | **P2** |
| Automated Prerequisite Detection | Medium | High | **P2** |
| ML Citation Extraction | Medium | Medium | **P2** |
| Advanced ZPD | Medium | Medium | **P3** |
| Batch Processing | Medium | Low | **P3** |
| A/B Testing Framework | Low | Medium | **P3** |
| Personalization Settings | Low | Low | **P3** |
| Embedding Cache | Low | Low | **P3** |
| Quality Metrics | Low | Medium | **P3** |
| Webhook Callbacks | Medium | Low | **P3** |
| Task Progress Updates | Low | Low | **P3** |
| Distributed Telemetry | Low | Medium | **P4** |
| Federated Learning | Low | Very High | **P4** |

---

## Recommended Implementation Order

### Sprint 1: Foundation (P0)
1. CI/CD and Monitoring setup
2. Task retry with exponential backoff
3. Basic test coverage

### Sprint 2: Content Quality (P1)
4. NER-based concept extraction
5. Explicit prerequisite declarations

### Sprint 3: Adaptive Intelligence (P1)
6. FSRS parameter optimization scheduling
7. ML-based evidence rules (initial model)

### Sprint 4: Social & Analytics (P2)
8. Social gamification (friends, challenges)
9. Analytics dashboard

### Sprint 5: Content Expansion (P2)
10. Audio overview generation
11. Additional file format support
12. ML citation extraction

### Sprint 6: Advanced Features (P3)
13. Automated prerequisite detection
14. Advanced ZPD
15. A/B testing framework

### Sprint 7: Polish (P3-P4)
16. Remaining P3 features
17. Federated learning (if resources permit)

---

## Success Metrics

| Feature | Metric | Target |
|---------|--------|--------|
| NER Extraction | F1 Score | >90% |
| ML Evidence | AUC | >0.85 |
| FSRS Optimization | Retention Improvement | +5% |
| Social Features | DAU Increase | +20% |
| Audio Overviews | Completion Rate | >60% |
| Analytics | Instructor Adoption | >80% |

---

## Resource Requirements

- **Backend Engineer:** 1-2 FTE for adaptive/ML features
- **Frontend Engineer:** 1 FTE for UI/dashboard
- **ML Engineer:** 0.5-1 FTE for model training
- **DevOps:** 0.5 FTE for CI/CD and monitoring
- **GPU Resources:** For DKT training and inference

---

## Conclusion

This plan outlines 24 bonus features extracted from the Phase 2-4 summary documents. The features are prioritized based on business value and implementation effort. Following the recommended sprint order will deliver maximum value while building on existing infrastructure.

The platform already has a strong foundation with implemented features like DKT, ECD Framework, Cognitive Load Estimation, and the complete web frontend. These bonus features will elevate NerdLearn from a comprehensive learning platform to a world-class adaptive learning system.
