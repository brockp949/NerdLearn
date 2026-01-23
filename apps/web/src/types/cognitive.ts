/**
 * Phase 1: Cognitive Foundation Types
 *
 * Types for frustration detection, metacognition, calibration, and interventions
 */

// ============== Frustration Detection ==============

export type FrustrationLevel = 'none' | 'mild' | 'moderate' | 'high' | 'severe';
export type StruggleType = 'none' | 'productive' | 'unproductive';

export type BehavioralSignal =
  | 'rapid_guessing'
  | 'extended_pause'
  | 'consecutive_errors'
  | 'erratic_navigation'
  | 'help_seeking_spike'
  | 'content_abandonment'
  | 'repeated_attempts'
  | 'regression';

export interface InteractionEvent {
  timestamp?: string;
  event_type: 'answer' | 'answer_submitted' | 'click' | 'navigation' | 'hint' | 'pause';
  correct?: boolean;
  response_time_ms?: number;
  content_id?: string;
  hint_used?: boolean;
  attempts?: number;
  metadata?: Record<string, unknown>;
}

export interface FrustrationIndicators {
  response_time_variance: number;
  rapid_response_ratio: number;
  extended_pause_ratio: number;
  consecutive_error_count: number;
  hint_usage_rate: number;
  navigation_entropy: number;
}

export interface FrustrationDetectionRequest {
  user_id: string;
  events: InteractionEvent[];
  context?: Record<string, unknown>;
}

export interface FrustrationResponse {
  level: FrustrationLevel;
  score: number;
  struggle_type: StruggleType;
  active_signals: BehavioralSignal[];
  recommended_action: string;
  confidence: number;
  indicators: FrustrationIndicators;
}

// ============== Metacognition ==============

export type MetacognitionPromptType =
  | 'confidence_rating'
  | 'self_explanation'
  | 'prediction'
  | 'reflection'
  | 'strategy_selection'
  | 'error_analysis';

export type CalibrationLevel =
  | 'unknown'
  | 'well_calibrated'
  | 'overconfident'
  | 'underconfident'
  | 'variable';

export interface MetacognitionPromptRequest {
  user_id: string;
  concept_name: string;
  timing: 'before' | 'during' | 'after';
  context?: Record<string, unknown>;
  force?: boolean;
}

export interface MetacognitionPromptResponse {
  prompt_id: string | null;
  prompt_type: MetacognitionPromptType | null;
  prompt_text: string | null;
  timing: string;
  required: boolean;
  follow_up_prompts: string[];
  reason?: string;
}

export interface ConfidenceRatingInput {
  user_id: string;
  concept_id: string;
  content_id: string;
  confidence: number; // 0-1
  context?: string;
}

export interface SelfExplanationInput {
  explanation_text: string;
  concept_name: string;
  expected_concepts?: string[];
  common_misconceptions?: string[];
}

export interface SelfExplanationAnalysis {
  quality_score: number;
  concepts_mentioned: string[];
  misconceptions_detected: string[];
  feedback: string;
  improvement_suggestions: string[];
}

// ============== Calibration ==============

export interface CalibrationRequest {
  user_id: string;
  concept_id?: string;
  time_window_hours?: number;
}

export interface CalibrationResponse {
  user_id: string;
  calibration_level: CalibrationLevel;
  mean_confidence: number;
  mean_performance: number;
  calibration_error: number;
  overconfidence_rate: number;
  underconfidence_rate: number;
  data_points: number;
}

export interface CalibrationFeedback {
  level: CalibrationLevel;
  message: string;
  recommendations: string[];
  specific_concepts?: string[];
}

// ============== Interventions ==============

export type InterventionType =
  | 'encouragement'
  | 'break_suggestion'
  | 'progress_reminder'
  | 'growth_mindset'
  | 'hint'
  | 'worked_example'
  | 'simplify_content'
  | 'prerequisite_review'
  | 'scaffold'
  | 'reflection_prompt'
  | 'self_explanation'
  | 'strategy_suggestion'
  | 'calibration_feedback'
  | 'reduce_difficulty'
  | 'increase_difficulty'
  | 'change_modality'
  | 'practice_break'
  | 'none';

export type InterventionPriority = 'critical' | 'high' | 'medium' | 'low';

export interface LearnerStateInput {
  user_id: string;
  frustration_score?: number;
  frustration_level?: FrustrationLevel;
  cognitive_load_score?: number;
  cognitive_load_level?: string;
  calibration_level?: CalibrationLevel;
  consecutive_errors?: number;
  time_on_task_minutes?: number;
  session_duration_minutes?: number;
  concepts_mastered_today?: number;
}

export interface Intervention {
  intervention_id: string;
  type: InterventionType;
  priority: InterventionPriority;
  title: string;
  message: string;
  action?: string;
  action_data?: Record<string, unknown>;
  display_duration_seconds?: number;
  dismissible: boolean;
  follow_ups: string[];
}

export interface InterventionRequest {
  learner_state: LearnerStateInput;
  events?: InteractionEvent[];
  context?: Record<string, unknown>;
}

export interface InterventionDecision {
  should_intervene: boolean;
  intervention: Intervention | null;
  reason: string;
  cooldown_seconds: number;
}

export interface InterventionHistory {
  total_interventions: number;
  by_type: Record<string, number>;
  most_common: string | null;
}

// ============== Cognitive Profile ==============

export interface UserBaseline {
  avg_response_time_ms: number | null;
  response_time_std: number | null;
  baseline_accuracy: number | null;
}

export interface CognitiveProfile {
  user_id: string;
  baseline: UserBaseline;
  calibration: {
    level: CalibrationLevel;
    mean_confidence: number;
    mean_performance: number;
    error: number;
  };
  interventions: InterventionHistory;
}

// ============== Confidence Scale ==============

export interface ConfidenceScaleOption {
  value: number;
  label: string;
  description?: string;
  emoji?: string;
}

export interface ConfidenceScale {
  concept_name: string;
  scale_type: 'numeric' | 'verbal' | 'emoji';
  options: ConfidenceScaleOption[];
  prompt: string;
}
