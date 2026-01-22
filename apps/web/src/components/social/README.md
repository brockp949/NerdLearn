# Phase 4: Social Learning Components

This directory should contain React components for the Phase 4 Innovation Mechanics features.

## API Endpoints Available

All endpoints are prefixed with `/api/social/`

### 1. Teachable Agent (Feynman Protocol)

Learn by teaching an AI student.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/teaching/start` | POST | Start a teaching session |
| `/teaching/explain` | POST | Submit an explanation |
| `/teaching/end/{session_id}` | POST | End session and get summary |
| `/teaching/session/{session_id}` | GET | Get session state |

**Request: Start Session**
```typescript
interface StartTeachingRequest {
  user_id: string;
  concept_id: string;
  concept_name: string;
  persona?: 'curious' | 'confused' | 'challenger' | 'visual' | 'practical';
  concept_description?: string;
}
```

**Request: Submit Explanation**
```typescript
interface TeachingExplanationRequest {
  session_id: string;
  explanation: string;
  concept_description?: string;
}
```

**Response: Teaching Response**
```typescript
interface TeachingResponse {
  response: string;
  comprehension: number; // 0-1
  comprehension_level: 'lost' | 'struggling' | 'emerging' | 'developing' | 'mastering';
  question_type?: string;
  knowledge_gaps: string[];
  concepts_understood: string[];
}
```

### 2. SimClass Debates

Multi-agent debates for perspective exploration.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/debate/start` | POST | Start a debate |
| `/debate/advance` | POST | Advance debate by one round |
| `/debate/summary/{session_id}` | POST | Get debate summary |
| `/debate/session/{session_id}` | GET | Get debate state |

**Request: Start Debate**
```typescript
interface StartDebateRequest {
  topic: string;
  format?: 'oxford' | 'socratic' | 'roundtable' | 'devils_advocate' | 'synthesis';
  panel_preset?: 'technical_pros_cons' | 'philosophical' | 'practical_application';
  learner_id?: string;
  max_rounds?: number;
}
```

### 3. Code Evaluator

AI-powered code review and feedback.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/code/challenge` | POST | Create a coding challenge |
| `/code/challenge/{id}` | GET | Get challenge details |
| `/code/submit` | POST | Submit code for evaluation |
| `/code/hint` | POST | Get a progressive hint |
| `/code/init-samples` | POST | Initialize sample challenges |
| `/code/challenges` | GET | List all challenges |

**Request: Submit Code**
```typescript
interface SubmitCodeRequest {
  challenge_id: string;
  user_id: string;
  code: string;
  dimensions?: ('correctness' | 'quality' | 'efficiency' | 'security' | 'documentation')[];
}
```

**Response: Evaluation Result**
```typescript
interface EvaluationResult {
  submission_id: string;
  passed: boolean;
  overall_score: number; // 0-100
  tests_passed: number;
  tests_total: number;
  dimension_scores: Record<string, DimensionScore>;
  feedback: FeedbackItem[];
  concepts_demonstrated: string[];
  concepts_to_review: string[];
  execution_time_ms?: number;
  runtime_errors: string[];
}
```

## Suggested Components

### TeachingSession.tsx
Interactive teaching session with:
- Concept selector (from course knowledge graph)
- Persona picker (curious, confused, challenger, visual, practical)
- Chat-like interface for explanations
- Comprehension meter with visual feedback
- Session summary with recommendations

### DebateViewer.tsx
Multi-agent debate interface with:
- Topic input and format selection
- Speaker cards showing role and stance
- Turn-based argument display
- Optional learner participation
- Summary view with key insights

### CodeChallenge.tsx
Coding challenge interface with:
- Challenge description and constraints
- Monaco editor for code input
- Test case display (visible ones)
- Multi-dimensional feedback display
- Progressive hint system

### CodeEvaluationFeedback.tsx
Detailed feedback display with:
- Overall score gauge
- Dimension breakdown (correctness, quality, efficiency, security)
- Line-by-line feedback markers
- Improvement suggestions
- Concept mastery indicators

## State Management

Consider using React Query or SWR for:
- Session state persistence
- Optimistic updates during teaching
- Debate round caching
- Evaluation result caching

## Example Hook

```typescript
// hooks/use-teaching.ts
import { useMutation, useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';

export function useTeachingSession() {
  const startMutation = useMutation({
    mutationFn: (params: StartTeachingRequest) =>
      api.post('/api/social/teaching/start', params),
  });

  const explainMutation = useMutation({
    mutationFn: (params: TeachingExplanationRequest) =>
      api.post('/api/social/teaching/explain', params),
  });

  return {
    startSession: startMutation.mutate,
    submitExplanation: explainMutation.mutate,
    isLoading: startMutation.isPending || explainMutation.isPending,
  };
}
```
