/**
 * PACER Learning Protocol Types
 *
 * P.A.C.E.R. Framework for information taxonomy:
 * - P: Procedural (Practice) - How-to steps and instructions
 * - A: Analogous (Critique) - Metaphors and comparisons
 * - C: Conceptual (Mapping) - Theories, principles, causality
 * - E: Evidence (Store & Rehearse) - Supporting facts/statistics
 * - R: Reference (Store & Rehearse) - Arbitrary details for SRS
 */

/** Five PACER information types */
export type PACERType =
  | "procedural"
  | "analogous"
  | "conceptual"
  | "evidence"
  | "reference";

/** Recommended cognitive action for each PACER type */
export type PACERAction = "practice" | "critique" | "map" | "link" | "recall";

/** Evidence relationship to concept */
export type EvidenceRelationshipType = "supports" | "contradicts" | "qualifies";

/** Breakdown point severity */
export type BreakdownSeverity = "minor" | "moderate" | "major";

/** PACER type metadata with display info */
export interface PACERTypeInfo {
  type: PACERType;
  label: string;
  action: PACERAction;
  actionLabel: string;
  description: string;
  color: string;
  icon: string;
}

/** PACER type configuration */
export const PACER_TYPES: Record<PACERType, PACERTypeInfo> = {
  procedural: {
    type: "procedural",
    label: "Procedural",
    action: "practice",
    actionLabel: "Practice",
    description: "Step-by-step instructions for HOW to do something",
    color: "blue",
    icon: "🔧",
  },
  analogous: {
    type: "analogous",
    label: "Analogous",
    action: "critique",
    actionLabel: "Critique",
    description: "Metaphors and comparisons to familiar concepts",
    color: "purple",
    icon: "🔄",
  },
  conceptual: {
    type: "conceptual",
    label: "Conceptual",
    action: "map",
    actionLabel: "Map",
    description: "Theories, principles, and causal relationships",
    color: "green",
    icon: "🧠",
  },
  evidence: {
    type: "evidence",
    label: "Evidence",
    action: "link",
    actionLabel: "Link",
    description: "Facts and statistics that support concepts",
    color: "amber",
    icon: "📊",
  },
  reference: {
    type: "reference",
    label: "Reference",
    action: "recall",
    actionLabel: "Recall",
    description: "Arbitrary details for rote memorization",
    color: "rose",
    icon: "📌",
  },
};

/** Classification result from API */
export interface ClassificationResult {
  pacerType: PACERType;
  confidence: number;
  reasoning: string;
  alternatives: Array<{
    type: PACERType;
    confidence: number;
  }>;
  recommendedAction: {
    action: string;
    description: string;
    tool: string;
  };
  contentHash: string;
}

/** Single step in the triage decision tree */
export interface TriageDecision {
  question: string;
  answer: boolean;
  leadsTo?: PACERType;
}

/** PACER-classified content item */
export interface PACERContentItem {
  id: number;
  courseId: number;
  moduleId?: number;
  pacerType: PACERType;
  title: string;
  content: string;
  classificationConfidence: number;
  metadata: Record<string, unknown>;
  createdAt: string;
}

/** Structural mapping between analogy domains */
export interface StructuralMapping {
  sourceElement: string;
  targetElement: string;
  relationship: string;
}

/** Breakdown point where analogy fails */
export interface BreakdownPoint {
  aspect: string;
  reason: string;
  severity: BreakdownSeverity;
  educationalNote: string;
}

/** Complete analogy for critique learning */
export interface Analogy {
  id: number;
  sourceDomain: string;
  targetDomain: string;
  summary: string;
  mappings: StructuralMapping[];
  validAspects: string[];
  breakdownPoints?: BreakdownPoint[]; // Hidden until after critique
  critiquePrompt: string;
}

/** User's critique submission */
export interface CritiqueSubmission {
  identifiedBreakdowns: string[];
  explanations: string[];
}

/** Critique evaluation result */
export interface CritiqueResult {
  score: number;
  precision: number;
  recall: number;
  correctlyIdentified: string[];
  missedBreakdowns: Array<{
    aspect: string;
    reason: string;
    note: string;
    severity?: BreakdownSeverity;
  }>;
  falsePositives: string[];
  feedback: string;
}

/** Evidence-concept link */
export interface EvidenceLink {
  evidenceId: number;
  conceptId: number;
  conceptName: string;
  relationshipType: EvidenceRelationshipType;
  strength: number;
  citation?: string;
}

/** Evidence item with linked concepts */
export interface LinkedEvidence {
  evidenceId: number;
  title: string;
  content: string;
  links: EvidenceLink[];
}

/** Procedural step completion result */
export interface StepResult {
  stepNumber: number;
  success: boolean;
  timeMs: number;
  errorCount: number;
  feedback: string;
}

/** Procedural progress status */
export interface ProceduralProgress {
  itemId: number;
  title: string;
  currentStep: number;
  totalSteps: number;
  completed: boolean;
  attempts: number;
  progressPercent: number;
  averageStepTimeMs?: number;
  totalErrors?: number;
}

/** User's PACER learning profile */
export interface UserPACERProfile {
  userId: number;
  proceduralProficiency: number;
  analogousProficiency: number;
  conceptualProficiency: number;
  evidenceProficiency: number;
  referenceProficiency: number;
  totalItemsProcessed: number;
  preferredTypes: PACERType[];
}

/** Request to classify content */
export interface ClassifyRequest {
  content: string;
  context?: Record<string, unknown>;
}

/** Request to create PACER content */
export interface CreateContentRequest {
  courseId: number;
  moduleId?: number;
  title: string;
  content: string;
  pacerType?: PACERType;
  metadata?: Record<string, unknown>;
}

/** Request to create analogy */
export interface CreateAnalogyRequest {
  courseId: number;
  moduleId?: number;
  title: string;
  content: string;
  sourceDomain: string;
  targetDomain: string;
  mappings: Array<{
    sourceElement: string;
    targetElement: string;
    relationship?: string;
  }>;
  breakdownPoints: Array<{
    aspect: string;
    reason: string;
    severity?: BreakdownSeverity;
    educationalNote?: string;
  }>;
}

/** Request to link evidence to concepts */
export interface LinkEvidenceRequest {
  evidenceItemId: number;
  conceptIds: number[];
  relationshipType?: EvidenceRelationshipType;
  strength?: number;
  citation?: string;
}

/** Request to complete procedural step */
export interface CompleteStepRequest {
  stepNumber: number;
  success: boolean;
  timeMs: number;
  errorCount?: number;
}

/** PACER colors for UI consistency */
export const PACER_COLORS: Record<PACERType, string> = {
  procedural: "#3B82F6", // blue-500
  analogous: "#A855F7", // purple-500
  conceptual: "#22C55E", // green-500
  evidence: "#F59E0B", // amber-500
  reference: "#F43F5E", // rose-500
};

/** PACER background colors (lighter) */
export const PACER_BG_COLORS: Record<PACERType, string> = {
  procedural: "#DBEAFE", // blue-100
  analogous: "#F3E8FF", // purple-100
  conceptual: "#DCFCE7", // green-100
  evidence: "#FEF3C7", // amber-100
  reference: "#FFE4E6", // rose-100
};

/** Helper to get PACER type info */
export function getPACERTypeInfo(type: PACERType): PACERTypeInfo {
  return PACER_TYPES[type];
}

/** Helper to get recommended action for type */
export function getRecommendedAction(type: PACERType): {
  action: PACERAction;
  label: string;
  description: string;
} {
  const info = PACER_TYPES[type];
  return {
    action: info.action,
    label: info.actionLabel,
    description: info.description,
  };
}
