/**
 * Phase 2: Curriculum Intelligence Hooks
 *
 * React hooks for curriculum generation and knowledge graph operations
 */

import { useState, useCallback, useEffect, useRef } from "react";
import { curriculumApi, graphApi } from "@/lib/api";
import type {
  CurriculumGenerationResponse,
  CurriculumJobStatus,
  Syllabus,
  ArcOfLearningPreview,
  CourseGraph,
  ConceptDetails,
  LearningPathItem,
  GraphStats,
  CurriculumDifficulty as DifficultyLevel,
  LearningStyle,
} from "@/types/curriculum";

// =============================================================================
// CURRICULUM GENERATION HOOK
// =============================================================================

interface UseCurriculumGenerationOptions {
  onComplete?: (response: CurriculumGenerationResponse) => void;
  onError?: (error: string) => void;
  pollInterval?: number;
}

export function useCurriculumGeneration({
  onComplete,
  onError,
  pollInterval = 3000,
}: UseCurriculumGenerationOptions = {}) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<CurriculumJobStatus | null>(null);
  const [result, setResult] = useState<CurriculumGenerationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const clearPoll = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const pollJobStatus = useCallback(
    async (id: string) => {
      try {
        const status = await curriculumApi.getJobStatus(id);
        setJobStatus(status);

        if (status.status === "completed") {
          clearPoll();
          const result = await curriculumApi.getJobResult(id);
          setResult(result);
          setIsGenerating(false);
          onComplete?.(result);
        } else if (status.status === "failed") {
          clearPoll();
          setIsGenerating(false);
          const errorMsg = status.error || "Generation failed";
          setError(errorMsg);
          onError?.(errorMsg);
        }
      } catch (err: any) {
        clearPoll();
        setIsGenerating(false);
        const message = err.response?.data?.detail || "Failed to poll job status";
        setError(message);
        onError?.(message);
      }
    },
    [clearPoll, onComplete, onError]
  );

  const generate = useCallback(
    async (params: {
      topic: string;
      courseId: number;
      durationWeeks?: number;
      difficultyLevel?: DifficultyLevel;
      targetAudience?: string;
      prerequisites?: string[];
      learningStyle?: LearningStyle;
      maxModules?: number;
    }) => {
      setIsGenerating(true);
      setError(null);
      setResult(null);
      setJobStatus(null);

      try {
        const response = await curriculumApi.generateCurriculumAsync({
          topic: params.topic,
          course_id: params.courseId,
          duration_weeks: params.durationWeeks,
          difficulty_level: params.difficultyLevel,
          target_audience: params.targetAudience,
          prerequisites: params.prerequisites,
          learning_style: params.learningStyle,
          max_modules: params.maxModules,
        });

        setJobId(response.job_id);

        // Start polling
        pollRef.current = setInterval(() => {
          pollJobStatus(response.job_id);
        }, pollInterval);

        return response;
      } catch (err: any) {
        setIsGenerating(false);
        const message = err.response?.data?.detail || "Failed to start generation";
        setError(message);
        onError?.(message);
        return null;
      }
    },
    [pollInterval, pollJobStatus, onError]
  );

  const generateSync = useCallback(
    async (params: {
      topic: string;
      courseId: number;
      durationWeeks?: number;
      difficultyLevel?: DifficultyLevel;
      targetAudience?: string;
      prerequisites?: string[];
      learningStyle?: LearningStyle;
      maxModules?: number;
    }) => {
      setIsGenerating(true);
      setError(null);
      setResult(null);

      try {
        const response = await curriculumApi.generateCurriculum({
          topic: params.topic,
          course_id: params.courseId,
          duration_weeks: params.durationWeeks,
          difficulty_level: params.difficultyLevel,
          target_audience: params.targetAudience,
          prerequisites: params.prerequisites,
          learning_style: params.learningStyle,
          max_modules: params.maxModules,
        });

        setResult(response);
        setIsGenerating(false);
        onComplete?.(response);
        return response;
      } catch (err: any) {
        setIsGenerating(false);
        const message = err.response?.data?.detail || "Failed to generate curriculum";
        setError(message);
        onError?.(message);
        return null;
      }
    },
    [onComplete, onError]
  );

  const cancel = useCallback(() => {
    clearPoll();
    setIsGenerating(false);
  }, [clearPoll]);

  const reset = useCallback(() => {
    clearPoll();
    setIsGenerating(false);
    setJobId(null);
    setJobStatus(null);
    setResult(null);
    setError(null);
  }, [clearPoll]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearPoll();
    };
  }, [clearPoll]);

  return {
    isGenerating,
    jobId,
    jobStatus,
    result,
    error,
    progress: jobStatus?.progress ?? 0,
    currentAgent: jobStatus?.current_agent,
    generate,
    generateSync,
    cancel,
    reset,
  };
}

// =============================================================================
// ARC OF LEARNING PREVIEW HOOK
// =============================================================================

export function useArcPreview() {
  const [preview, setPreview] = useState<ArcOfLearningPreview | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const getPreview = useCallback(
    async (
      topic: string,
      durationWeeks?: number,
      difficultyLevel?: DifficultyLevel
    ) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await curriculumApi.previewArcOfLearning({
          topic,
          duration_weeks: durationWeeks,
          difficulty_level: difficultyLevel,
        });
        setPreview(response);
        return response;
      } catch (err: any) {
        const message = err.response?.data?.detail || "Failed to get preview";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  return {
    preview,
    isLoading,
    error,
    getPreview,
  };
}

// =============================================================================
// KNOWLEDGE GRAPH HOOK
// =============================================================================

interface UseKnowledgeGraphOptions {
  courseId: number;
  autoLoad?: boolean;
}

export function useKnowledgeGraph({ courseId, autoLoad = false }: UseKnowledgeGraphOptions) {
  const [graph, setGraph] = useState<CourseGraph | null>(null);
  const [stats, setStats] = useState<GraphStats | null>(null);
  const [selectedConcept, setSelectedConcept] = useState<ConceptDetails | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadGraph = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await graphApi.getCourseGraph(courseId);
      setGraph(response);

      // Extract stats from response
      if (response.meta) {
        setStats({
          modules: 0, // Not available in this endpoint
          concepts: response.meta.total_concepts,
          prerequisites: response.meta.total_relationships,
          avg_difficulty: 0, // Calculate from nodes if needed
        });
      }

      return response;
    } catch (err: any) {
      const message = err.response?.data?.detail || "Failed to load graph";
      setError(message);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [courseId]);

  const selectConcept = useCallback(
    async (conceptName: string) => {
      setIsLoading(true);
      setError(null);

      try {
        // Note: This would need a separate API endpoint in production
        // For now, we find the concept in the loaded graph
        if (graph) {
          const node = graph.nodes.find((n) => n.id === conceptName || n.label === conceptName);
          if (node) {
            const incomingEdges = graph.edges.filter((e) => e.target === node.id);
            const outgoingEdges = graph.edges.filter((e) => e.source === node.id);

            setSelectedConcept({
              name: node.label,
              difficulty: node.difficulty ?? 5,
              importance: node.importance ?? 0.5,
              description: undefined,
              module: node.module,
              module_id: node.module_id,
              prerequisites: incomingEdges.map((e) => ({
                name: e.source,
                confidence: e.confidence ?? 0.5,
              })),
              dependents: outgoingEdges.map((e) => ({
                name: e.target,
                confidence: e.confidence ?? 0.5,
              })),
            });
          }
        }
        return selectedConcept;
      } catch (err: any) {
        const message = err.response?.data?.detail || "Failed to get concept details";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [graph, selectedConcept]
  );

  const clearSelection = useCallback(() => {
    setSelectedConcept(null);
  }, []);

  useEffect(() => {
    if (autoLoad) {
      loadGraph();
    }
  }, [autoLoad, loadGraph]);

  return {
    graph,
    stats,
    selectedConcept,
    isLoading,
    error,
    loadGraph,
    selectConcept,
    clearSelection,
    refresh: loadGraph,
  };
}

// =============================================================================
// LEARNING PATH HOOK
// =============================================================================

interface UseLearningPathOptions {
  courseId: number;
  userId?: string;
}

export function useLearningPath({ courseId }: UseLearningPathOptions) {
  const [path, setPath] = useState<LearningPathItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generatePath = useCallback(
    async (targetConcepts: string[], masteredConcepts?: string[]) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await graphApi.getLearningPath(
          courseId,
          targetConcepts,
          masteredConcepts
        );
        setPath(response);
        return response;
      } catch (err: any) {
        const message = err.response?.data?.detail || "Failed to generate path";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [courseId]
  );

  const clearPath = useCallback(() => {
    setPath([]);
  }, []);

  return {
    path,
    isLoading,
    error,
    generatePath,
    clearPath,
  };
}

// =============================================================================
// COMBINED CURRICULUM HOOK
// =============================================================================

interface UseCurriculumOptions {
  courseId: number;
}

export function useCurriculum({ courseId }: UseCurriculumOptions) {
  const generationHook = useCurriculumGeneration();
  const previewHook = useArcPreview();
  const graphHook = useKnowledgeGraph({ courseId, autoLoad: true });
  const pathHook = useLearningPath({ courseId });

  return {
    generation: generationHook,
    preview: previewHook,
    graph: graphHook,
    path: pathHook,
  };
}
