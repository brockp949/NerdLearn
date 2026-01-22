/**
 * Phase 4: Agentic Social Layer Types
 *
 * Types for:
 * 1. Teachable Agent (Feynman Protocol)
 * 2. SimClass Debates
 * 3. Code Evaluator
 */

// =============================================================================
// TEACHABLE AGENT TYPES
// =============================================================================

export type StudentPersona = 'curious' | 'confused' | 'challenger' | 'visual' | 'practical';

export type ComprehensionLevel = 'lost' | 'struggling' | 'emerging' | 'developing' | 'mastering';

export type QuestionType =
  | 'clarification'
  | 'example'
  | 'connection'
  | 'application'
  | 'challenge'
  | 'elaboration'
  | 'confirmation';

export interface StartTeachingRequest {
  user_id: string;
  concept_id: string;
  concept_name: string;
  persona?: StudentPersona;
  concept_description?: string;
}

export interface TeachingExplanationRequest {
  session_id: string;
  explanation: string;
  concept_description?: string;
}

export interface TeachingSessionResponse {
  session_id: string;
  concept_name: string;
  persona: StudentPersona;
  student_name: string;
  opening_question: string;
  comprehension: number;
  comprehension_level: ComprehensionLevel;
  message: string;
}

export interface TeachingResponse {
  response: string;
  comprehension: number;
  comprehension_level: ComprehensionLevel;
  question_type: QuestionType | null;
  knowledge_gaps: string[];
  concepts_understood: string[];
}

export interface TeachingSessionSummary {
  session_id: string;
  concept: string;
  persona_used: StudentPersona;
  duration_minutes: number;
  total_exchanges: number;
  final_comprehension: number;
  comprehension_level: ComprehensionLevel;
  comprehension_progress: number[];
  improvement_per_exchange: number;
  knowledge_gaps_identified: string[];
  strong_explanations: string[];
  teaching_effectiveness: number;
  recommendations: string[];
}

export interface TeachingSessionState {
  session_id: string;
  concept_name: string;
  persona: StudentPersona;
  exchange_count: number;
  comprehension: number;
  comprehension_level: ComprehensionLevel;
  knowledge_gaps: string[];
  strong_points: string[];
  completed: boolean;
}

// =============================================================================
// SIMCLASS DEBATE TYPES
// =============================================================================

export type DebateRole =
  | 'advocate'
  | 'skeptic'
  | 'synthesizer'
  | 'historian'
  | 'futurist'
  | 'practitioner'
  | 'theorist'
  | 'contrarian';

export type DebateFormat =
  | 'oxford'
  | 'socratic'
  | 'roundtable'
  | 'devils_advocate'
  | 'synthesis';

export type PanelPreset = 'technical_pros_cons' | 'philosophical' | 'practical_application';

export interface StartDebateRequest {
  topic: string;
  format?: DebateFormat;
  panel_preset?: PanelPreset;
  learner_id?: string;
  max_rounds?: number;
}

export interface DebateContributionRequest {
  session_id: string;
  learner_contribution?: string;
}

export interface DebateParticipant {
  name: string;
  role: DebateRole;
  stance?: string;
}

export interface DebateArgument {
  speaker: string;
  role: DebateRole;
  content: string;
  argument_type: string;
  key_points: string[];
}

export interface DebateSessionResponse {
  session_id: string;
  topic: string;
  format: DebateFormat;
  participants: DebateParticipant[];
  current_round: number;
  max_rounds: number;
  opening_statements: DebateArgument[];
}

export interface DebateRoundResponse {
  session_id: string;
  current_round: number;
  completed: boolean;
  arguments: DebateArgument[];
}

export interface DebateSummary {
  session_id: string;
  topic: string;
  format: DebateFormat;
  total_rounds: number;
  total_arguments: number;
  participants: { name: string; role: DebateRole }[];
  executive_summary: string;
  key_insights: string[];
  consensus_points: string[];
  disagreement_points: string[];
  strongest_arguments: {
    speaker: string;
    argument: string;
    strength: 'compelling' | 'strong' | 'moderate';
  }[];
  learning_takeaways: string[];
  recommended_further_reading: string[];
  learner_contributions?: number;
}

export interface DebateSessionState {
  session_id: string;
  topic: string;
  format: DebateFormat;
  participants: DebateParticipant[];
  current_round: number;
  max_rounds: number;
  total_arguments: number;
  completed: boolean;
  key_insights: string[];
  consensus_points: string[];
  disagreement_points: string[];
}

// =============================================================================
// CODE EVALUATOR TYPES
// =============================================================================

export type DifficultyLevel = 'beginner' | 'intermediate' | 'advanced' | 'expert';

export type EvaluationDimension =
  | 'correctness'
  | 'quality'
  | 'efficiency'
  | 'security'
  | 'completeness'
  | 'documentation';

export type HintLevel = 'nudge' | 'guidance' | 'explanation' | 'partial' | 'solution';

export type FeedbackType = 'praise' | 'issue' | 'suggestion' | 'hint' | 'example';

export interface TestCase {
  input: any;
  expected: any;
  description: string;
}

export interface CodingChallenge {
  challenge_id: string;
  title: string;
  description: string;
  difficulty: DifficultyLevel;
  language: string;
  function_name: string;
  parameters: { name: string; type: string }[];
  return_type: string;
  constraints: string[];
  test_cases: TestCase[];
  concepts_tested: string[];
  estimated_minutes: number;
}

export interface CreateChallengeRequest {
  challenge_id: string;
  title: string;
  description: string;
  difficulty: DifficultyLevel;
  function_name: string;
  parameters: { name: string; type: string }[];
  return_type: string;
  test_cases: { input: any; expected: any; description?: string; hidden?: boolean; edge_case?: boolean }[];
  concepts_tested: string[];
  hints: string[];
  reference_solution: string;
  language?: string;
}

export interface SubmitCodeRequest {
  challenge_id: string;
  user_id: string;
  code: string;
  dimensions?: EvaluationDimension[];
}

export interface GetHintRequest {
  challenge_id: string;
  user_id: string;
  code: string;
  hint_level?: HintLevel;
}

export interface DimensionScore {
  score: number;
  strengths: string[];
  improvements: string[];
  feedback_count: number;
}

export interface FeedbackItem {
  type: FeedbackType;
  dimension: EvaluationDimension;
  message: string;
  line_number: number | null;
  suggestion: string | null;
  priority: number;
}

export interface EvaluationResult {
  submission_id: string;
  passed: boolean;
  overall_score: number;
  tests_passed: number;
  tests_total: number;
  dimension_scores: Record<EvaluationDimension, DimensionScore>;
  feedback: FeedbackItem[];
  concepts_demonstrated: string[];
  concepts_to_review: string[];
  execution_time_ms: number | null;
  runtime_errors: string[];
}

export interface HintResponse {
  hint_level: HintLevel;
  hint: string;
  hints_remaining: number;
}

export interface ChallengeSummary {
  challenge_id: string;
  title: string;
  difficulty: DifficultyLevel;
  concepts: string[];
  estimated_minutes: number;
}

export interface ChallengeListResponse {
  count: number;
  challenges: ChallengeSummary[];
}
