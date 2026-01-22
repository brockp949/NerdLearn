/**
 * Phase 4: Social Learning Components
 *
 * Components for:
 * 1. Teachable Agent (Feynman Protocol) - Learn by teaching
 * 2. SimClass Debates - Multi-agent perspective exploration
 * 3. Code Evaluator - AI-powered code challenges
 */

export { TeachingSession } from "./TeachingSession";
export { DebateViewer } from "./DebateViewer";
export { default as CodeChallenge } from "./CodeChallenge";

// Re-export types for convenience
export type {
  // Teachable Agent
  StudentPersona,
  ComprehensionLevel,
  QuestionType,
  TeachingResponse,
  TeachingSessionResponse,
  TeachingSessionSummary,
  TeachingSessionState,
  // Debates
  DebateRole,
  DebateFormat,
  PanelPreset,
  DebateArgument,
  DebateSessionResponse,
  DebateRoundResponse,
  DebateSummary,
  DebateParticipant,
  // Code Evaluator
  DifficultyLevel,
  EvaluationDimension,
  HintLevel,
  FeedbackType,
  CodingChallenge,
  EvaluationResult,
  HintResponse,
} from "@/types/social";
