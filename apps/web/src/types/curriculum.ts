/**
 * Phase 2: Curriculum Intelligence Types
 *
 * Types for curriculum generation, knowledge graphs, and GraphRAG
 */

// ============== Curriculum Generation ==============

export type DifficultyLevel = 'beginner' | 'intermediate' | 'advanced';
export type LearningStyle = 'visual' | 'text' | 'interactive' | 'balanced';
export type JobStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface CurriculumGenerationRequest {
  topic: string;
  course_id: number;
  duration_weeks?: number;
  difficulty_level?: DifficultyLevel;
  target_audience?: string;
  prerequisites?: string[];
  learning_style?: LearningStyle;
  max_modules?: number;
}

export interface ModuleData {
  week: number;
  title: string;
  concepts: string[];
  difficulty: number;
  prerequisites: string[];
  rationale: string;
  learning_objectives?: string[];
  estimated_hours?: number;
}

export interface Syllabus {
  modules: ModuleData[];
  overall_arc: string;
  generated_at?: string;
  topic?: string;
  duration_weeks?: number;
}

export interface CurriculumGenerationResponse {
  success: boolean;
  job_id?: string;
  syllabus?: Syllabus;
  quality_score?: number;
  errors: string[];
  warnings: string[];
  generation_time_seconds?: number;
}

export interface CurriculumJobStatus {
  job_id: string;
  status: JobStatus;
  progress: number;
  current_agent?: string;
  result?: CurriculumGenerationResponse;
  error?: string;
  created_at: string;
  updated_at: string;
}

export interface ArcOfLearningPreview {
  success: boolean;
  arc_of_learning?: Syllabus;
  errors: string[];
  message: string;
}

// ============== Knowledge Graph ==============

export interface GraphNode {
  id: string;
  label: string;
  module?: string;
  module_id?: number;
  module_order?: number;
  difficulty?: number;
  importance?: number;
  type?: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  type?: string;
  confidence?: number;
}

export interface CourseGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  meta: {
    course_id: number;
    total_concepts: number;
    total_relationships: number;
  };
}

export interface ConceptDetails {
  name: string;
  difficulty: number;
  importance: number;
  description?: string;
  module?: string;
  module_id?: number;
  prerequisites: Array<{ name: string; confidence: number }>;
  dependents: Array<{ name: string; confidence: number }>;
}

export interface LearningPathItem {
  name: string;
  difficulty: number;
  depth: number;
  module_order: number;
  weight: number;
}

export interface GraphStats {
  modules: number;
  concepts: number;
  prerequisites: number;
  avg_difficulty: number;
}

// ============== GraphRAG ==============

export interface ConceptCommunity {
  community_id: number;
  concepts: string[];
  central_concept: string;
  difficulty_range: [number, number];
  summary?: string;
  keywords: string[];
}

export interface GraphRAGResult {
  communities: ConceptCommunity[];
  global_summary: string;
  prerequisite_chains: string[][];
  concept_hierarchy: Record<string, string[]>;
  recommendations: string[];
}

// ============== Agent State ==============

export type AgentName = 'architect' | 'refiner' | 'verifier';

export interface AgentProgress {
  agent: AgentName;
  status: 'pending' | 'running' | 'completed' | 'failed';
  message?: string;
  timestamp: string;
}

// ============== Prerequisite Management ==============

export interface PrerequisiteRelation {
  prerequisite_name: string;
  concept_name: string;
  confidence: number;
  type: 'explicit' | 'sequential' | 'inferred';
}

export interface ConceptNode {
  course_id: number;
  module_id: number;
  name: string;
  difficulty?: number;
  importance?: number;
  description?: string;
}
