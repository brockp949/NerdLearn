/**
 * MSW (Mock Service Worker) Handlers
 *
 * These handlers intercept API requests during tests and return mock responses.
 * Organized by API domain (cognitive, social, curriculum, etc.)
 */

import { http, HttpResponse } from 'msw';

const API_BASE = '/api';

// =============================================================================
// COGNITIVE API HANDLERS
// =============================================================================

const cognitiveHandlers = [
  // Frustration Detection
  http.post(`${API_BASE}/cognitive/frustration/detect`, () => {
    return HttpResponse.json({
      score: 0.3,
      level: 'low',
      indicators: ['quick_responses'],
      timestamp: new Date().toISOString(),
      recommendation: 'Continue with current approach',
    });
  }),

  http.post(`${API_BASE}/cognitive/frustration/update-baseline`, () => {
    return HttpResponse.json({
      success: true,
      baseline_updated: true,
    });
  }),

  // Metacognition
  http.post(`${API_BASE}/cognitive/metacognition/prompt`, () => {
    return HttpResponse.json({
      prompt_id: 'prompt_123',
      prompt_type: 'reflection',
      prompt_text: 'How confident are you in understanding this concept?',
      should_display: true,
      timing: 'after',
    });
  }),

  http.get(`${API_BASE}/cognitive/metacognition/confidence-scale`, () => {
    return HttpResponse.json({
      scale_type: 'numeric',
      options: [
        { value: 1, label: 'Not at all confident' },
        { value: 2, label: 'Slightly confident' },
        { value: 3, label: 'Moderately confident' },
        { value: 4, label: 'Very confident' },
        { value: 5, label: 'Extremely confident' },
      ],
    });
  }),

  http.post(`${API_BASE}/cognitive/metacognition/record-confidence`, () => {
    return HttpResponse.json({
      success: true,
      confidence_id: 'conf_123',
    });
  }),

  http.post(`${API_BASE}/cognitive/metacognition/analyze-explanation`, () => {
    return HttpResponse.json({
      quality_score: 0.75,
      detected_concepts: ['binary_search', 'divide_conquer'],
      missing_concepts: ['time_complexity'],
      misconceptions: [],
      suggestions: ['Consider discussing time complexity'],
    });
  }),

  // Calibration
  http.post(`${API_BASE}/cognitive/calibration/calculate`, () => {
    return HttpResponse.json({
      bias: 0.1,
      overconfidence_index: 0.15,
      calibration_level: 'well_calibrated',
      data_points: 25,
    });
  }),

  http.post(`${API_BASE}/cognitive/calibration/feedback`, () => {
    return HttpResponse.json({
      feedback_type: 'positive',
      message: 'Your self-assessment is well-calibrated!',
      suggestions: [],
    });
  }),

  // Interventions
  http.post(`${API_BASE}/cognitive/intervention/decide`, () => {
    return HttpResponse.json({
      should_intervene: false,
      intervention_type: null,
      message: null,
      priority: 'low',
      cooldown_remaining: 0,
    });
  }),

  http.get(`${API_BASE}/cognitive/intervention/history/:userId`, () => {
    return HttpResponse.json({
      interventions: [],
      total_count: 0,
    });
  }),

  // Profile
  http.get(`${API_BASE}/cognitive/profile/:userId`, () => {
    return HttpResponse.json({
      user_id: 'user_123',
      frustration_baseline: { average: 0.2, std_dev: 0.1 },
      calibration_history: [],
      learning_preferences: { modality: 'visual' },
    });
  }),
];

// =============================================================================
// SOCIAL API HANDLERS
// =============================================================================

const socialHandlers = [
  // Teaching Sessions
  http.post(`${API_BASE}/social/teaching/start`, () => {
    return HttpResponse.json({
      session_id: 'session_123',
      concept_name: 'Binary Search',
      persona: 'curious',
      opening_question: "Hi! I'm curious about Binary Search. Can you explain it to me?",
      comprehension_level: 0.0,
    });
  }),

  http.post(`${API_BASE}/social/teaching/explain`, () => {
    return HttpResponse.json({
      session_id: 'session_123',
      student_response: "That's interesting! Can you give me an example?",
      question_type: 'example',
      comprehension_before: 0.0,
      comprehension_after: 0.3,
      identified_gaps: [],
      feedback: 'Good start!',
    });
  }),

  http.post(`${API_BASE}/social/teaching/end/:sessionId`, () => {
    return HttpResponse.json({
      session_id: 'session_123',
      total_exchanges: 5,
      final_comprehension: 0.8,
      teaching_effectiveness: 0.85,
      recommendations: ['Practice explaining with examples'],
    });
  }),

  http.get(`${API_BASE}/social/teaching/session/:sessionId`, () => {
    return HttpResponse.json({
      session_id: 'session_123',
      concept_name: 'Binary Search',
      exchanges: [],
      comprehension_level: 0.5,
      completed: false,
    });
  }),

  // Debates
  http.post(`${API_BASE}/social/debate/start`, () => {
    return HttpResponse.json({
      session_id: 'debate_123',
      topic: 'AI in Education',
      format: 'roundtable',
      agents: [
        { name: 'Alex', role: 'advocate' },
        { name: 'Jordan', role: 'skeptic' },
      ],
      current_round: 1,
    });
  }),

  http.post(`${API_BASE}/social/debate/advance`, () => {
    return HttpResponse.json({
      session_id: 'debate_123',
      contributions: [
        {
          agent: 'Alex',
          content: 'AI can personalize learning at scale.',
          role: 'advocate',
        },
      ],
      current_round: 2,
      completed: false,
    });
  }),

  // Code Challenges
  http.get(`${API_BASE}/social/code/challenges`, () => {
    return HttpResponse.json({
      challenges: [
        {
          challenge_id: 'two_sum',
          title: 'Two Sum',
          difficulty: 'beginner',
          concepts_tested: ['arrays', 'hash_maps'],
        },
      ],
    });
  }),

  http.post(`${API_BASE}/social/code/submit`, () => {
    return HttpResponse.json({
      passed: true,
      tests_passed: 3,
      tests_total: 3,
      execution_time_ms: 50,
      feedback: [
        { dimension: 'correctness', score: 1.0, feedback: 'All tests passed!' },
      ],
    });
  }),
];

// =============================================================================
// GRAPH API HANDLERS
// =============================================================================

const graphHandlers = [
  http.get(`${API_BASE}/graph/`, () => {
    return HttpResponse.json({
      nodes: [
        { id: 'binary_search', label: 'Binary Search', difficulty: 5 },
        { id: 'arrays', label: 'Arrays', difficulty: 3 },
      ],
      edges: [
        { source: 'arrays', target: 'binary_search', type: 'prerequisite' },
      ],
    });
  }),

  http.get(`${API_BASE}/graph/courses/:courseId`, () => {
    return HttpResponse.json({
      nodes: [],
      edges: [],
      meta: { course_id: 1, total_concepts: 0 },
    });
  }),

  http.post(`${API_BASE}/graph/courses/:courseId/learning-path`, () => {
    return HttpResponse.json({
      path: [
        { name: 'Arrays', difficulty: 3 },
        { name: 'Binary Search', difficulty: 5 },
      ],
    });
  }),
];

// =============================================================================
// ADAPTIVE API HANDLERS
// =============================================================================

const adaptiveHandlers = [
  http.post(`${API_BASE}/adaptive/cognitive-load/estimate`, () => {
    return HttpResponse.json({
      load_score: 0.6,
      load_level: 'moderate',
      recommendation: 'Current difficulty is appropriate',
    });
  }),
];

// =============================================================================
// CURRICULUM API HANDLERS
// =============================================================================

const curriculumHandlers = [
  http.post(`${API_BASE}/curriculum/generate`, () => {
    return HttpResponse.json({
      curriculum_id: 'curr_123',
      topic: 'Machine Learning',
      modules: [
        { id: 1, title: 'Introduction', order: 1 },
        { id: 2, title: 'Fundamentals', order: 2 },
      ],
    });
  }),

  http.get(`${API_BASE}/curriculum/jobs/:jobId`, () => {
    return HttpResponse.json({
      job_id: 'job_123',
      status: 'completed',
      progress: 100,
    });
  }),
];

// =============================================================================
// MULTIMODAL API HANDLERS
// =============================================================================

const multimodalHandlers = [
  http.post(`${API_BASE}/multimodal/podcast/generate`, () => {
    return HttpResponse.json({
      podcast_id: 'podcast_123',
      script: 'Welcome to the podcast...',
      duration_estimate: 600,
    });
  }),

  http.post(`${API_BASE}/multimodal/diagram/generate`, () => {
    return HttpResponse.json({
      diagram_id: 'diagram_123',
      mermaid_code: 'graph TD; A-->B;',
      diagram_type: 'flowchart',
    });
  }),
];

// =============================================================================
// CHAT API HANDLERS
// =============================================================================

const chatHandlers = [
  http.post(`${API_BASE}/chat/`, () => {
    return HttpResponse.json({
      response: 'This is a mock response from the RAG system.',
      sources: [],
      confidence: 0.9,
    });
  }),
];

// =============================================================================
// EXPORT ALL HANDLERS
// =============================================================================

export const handlers = [
  ...cognitiveHandlers,
  ...socialHandlers,
  ...graphHandlers,
  ...adaptiveHandlers,
  ...curriculumHandlers,
  ...multimodalHandlers,
  ...chatHandlers,
];
