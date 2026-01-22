/**
 * Types for Multi-Modal Content Transformation
 *
 * Supports text ↔ diagram ↔ podcast transformations
 * with persistent conceptual state tracking.
 */

// ============== Diagram Types ==============

export type DiagramType =
  | "flowchart"
  | "mindmap"
  | "concept_map"
  | "sequence"
  | "state";

export type NodeType =
  | "default"
  | "input"
  | "output"
  | "group"
  | "concept"
  | "process"
  | "decision"
  | "annotation";

export type EdgeType =
  | "default"
  | "straight"
  | "step"
  | "smoothstep"
  | "bezier"
  | "animated";

export interface DiagramNode {
  id: string;
  type: NodeType;
  position: { x: number; y: number };
  data: {
    label: string;
    description?: string;
    [key: string]: any;
  };
  style?: Record<string, any>;
  parentNode?: string;
  extent?: string;
}

export interface DiagramEdge {
  id: string;
  source: string;
  target: string;
  type: EdgeType;
  label?: string;
  animated?: boolean;
  style?: Record<string, any>;
  markerEnd?: { type: string };
}

export interface DiagramData {
  id: string;
  type: DiagramType;
  title: string;
  nodes: DiagramNode[];
  edges: DiagramEdge[];
  mermaidSource: string;
  metadata: {
    node_count: number;
    edge_count: number;
    focus_concepts?: string[];
    [key: string]: any;
  };
}

// ============== Podcast Types ==============

export type SpeakerRole = "host" | "guest" | "expert" | "skeptic";

export interface ScriptSegment {
  speaker: SpeakerRole;
  text: string;
  duration_seconds: number;
  emotion: string;
}

export interface PodcastScript {
  title: string;
  segments: ScriptSegment[];
  total_duration_seconds: number;
}

export interface PodcastEpisode {
  episode_id: string;
  title: string;
  script_segments: ScriptSegment[];
  total_duration_seconds: number;
  audio_url: string | null;
  concepts_covered: string[];
}

// ============== Content Morphing Types ==============

export type ContentModality = "text" | "diagram" | "podcast";

export type ConceptMasteryLevel =
  | "unknown"
  | "introduced"
  | "familiar"
  | "understood"
  | "mastered";

export interface ConceptState {
  concept_id: string;
  name: string;
  mastery_level: ConceptMasteryLevel;
  modalities_seen: ContentModality[];
  last_interaction: string | null;
  interaction_count: number;
  notes: string[];
}

export interface ConceptualState {
  user_id: string;
  content_id: string;
  concepts: Record<string, ConceptState>;
  current_modality: ContentModality;
  modality_history: Array<{
    from: ContentModality;
    to: ContentModality;
    timestamp: string;
  }>;
  session_start: string;
  total_time_seconds: number;
  metadata: Record<string, any>;
}

export interface MorphedContent {
  original_modality: ContentModality;
  target_modality: ContentModality;
  content: DiagramData | PodcastEpisode | string;
  concepts_extracted: string[];
  transformation_notes: string;
  state_updated: boolean;
}

// ============== API Request/Response Types ==============

export interface DiagramGenerationRequest {
  content: string;
  diagram_type: DiagramType;
  title?: string;
  focus_concepts?: string[];
}

export interface PodcastGenerationRequest {
  content: string;
  topic: string;
  duration_minutes: number;
  style: "educational" | "casual" | "debate" | "interview";
  include_expert?: boolean;
  use_debate_format?: boolean;
}

export interface ContentMorphRequest {
  content: string;
  source_modality: ContentModality;
  target_modality: ContentModality;
  user_id?: string;
  content_id?: string;
  options?: Record<string, any>;
}

export interface ModalityRecommendation {
  recommended_modality: ContentModality;
  reason: string;
  alternatives: ContentModality[];
  weak_concepts: string[];
  current_state: ContentModality;
}

export interface LearningSummary {
  user_id: string;
  content_id: string;
  total_concepts: number;
  progress_percent: number;
  mastery_distribution: Record<ConceptMasteryLevel, number>;
  modality_usage: Record<ContentModality, number>;
  session_duration_seconds: number;
  modality_switches: number;
  weak_concepts: string[];
  current_modality: ContentModality;
}
