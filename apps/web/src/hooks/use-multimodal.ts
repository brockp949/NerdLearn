import { useState, useCallback, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { multimodalApi } from "@/lib/api";
import type {
  ContentModality,
  DiagramData,
  PodcastEpisode,
  ConceptualState,
  ModalityRecommendation,
  LearningSummary,
  DiagramType,
} from "@/types/multimodal";

/**
 * Hook for managing multi-modal content transformations
 *
 * Provides:
 * - Content morphing between modalities
 * - State persistence across transformations
 * - AI-powered modality recommendations
 */
export function useMultimodal(userId?: string, contentId?: string) {
  const queryClient = useQueryClient();
  const [currentModality, setCurrentModality] = useState<ContentModality>("text");
  const [currentContent, setCurrentContent] = useState<string | DiagramData | PodcastEpisode | null>(null);

  // Fetch conceptual state
  const {
    data: conceptualState,
    isLoading: isLoadingState,
    refetch: refetchState,
  } = useQuery({
    queryKey: ["conceptualState", userId, contentId],
    queryFn: () => multimodalApi.getConceptualState(userId!, contentId!),
    enabled: !!userId && !!contentId,
  });

  // Fetch learning summary
  const {
    data: learningSummary,
    isLoading: isLoadingSummary,
    refetch: refetchSummary,
  } = useQuery({
    queryKey: ["learningSummary", userId, contentId],
    queryFn: () => multimodalApi.getLearningSummary(userId!, contentId!),
    enabled: !!userId && !!contentId,
  });

  // Fetch modality recommendation
  const {
    data: recommendation,
    refetch: refetchRecommendation,
  } = useQuery({
    queryKey: ["modalityRecommendation", userId, contentId],
    queryFn: () =>
      multimodalApi.getModalityRecommendation({
        user_id: userId!,
        content_id: contentId!,
      }),
    enabled: !!userId && !!contentId,
  });

  // Morph content mutation
  const morphMutation = useMutation({
    mutationFn: (params: {
      content: string;
      sourceModality: ContentModality;
      targetModality: ContentModality;
      options?: Record<string, any>;
    }) =>
      multimodalApi.morphContent({
        content: params.content,
        source_modality: params.sourceModality,
        target_modality: params.targetModality,
        user_id: userId,
        content_id: contentId,
        options: params.options,
      }),
    onSuccess: (data) => {
      setCurrentModality(data.target_modality);
      setCurrentContent(data.content);
      // Invalidate queries to refresh state
      if (userId && contentId) {
        queryClient.invalidateQueries({ queryKey: ["conceptualState", userId, contentId] });
        queryClient.invalidateQueries({ queryKey: ["learningSummary", userId, contentId] });
        queryClient.invalidateQueries({ queryKey: ["modalityRecommendation", userId, contentId] });
      }
    },
  });

  // Generate podcast mutation
  const podcastMutation = useMutation({
    mutationFn: (params: {
      content: string;
      topic: string;
      durationMinutes?: number;
      style?: "educational" | "casual" | "debate" | "interview";
    }) =>
      multimodalApi.generatePodcast({
        content: params.content,
        topic: params.topic,
        duration_minutes: params.durationMinutes,
        style: params.style,
      }),
    onSuccess: (data) => {
      setCurrentModality("podcast");
      setCurrentContent(data);
    },
  });

  // Generate diagram mutation
  const diagramMutation = useMutation({
    mutationFn: (params: {
      content: string;
      diagramType?: DiagramType;
      title?: string;
      focusConcepts?: string[];
    }) =>
      multimodalApi.generateDiagram({
        content: params.content,
        diagram_type: params.diagramType,
        title: params.title,
        focus_concepts: params.focusConcepts,
      }),
    onSuccess: (data) => {
      setCurrentModality("diagram");
      setCurrentContent(data);
    },
  });

  // Reset state mutation
  const resetMutation = useMutation({
    mutationFn: () => multimodalApi.resetConceptualState(userId!, contentId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["conceptualState", userId, contentId] });
      queryClient.invalidateQueries({ queryKey: ["learningSummary", userId, contentId] });
    },
  });

  // Convenience function to switch modality
  const switchModality = useCallback(
    async (
      targetModality: ContentModality,
      content: string,
      options?: Record<string, any>
    ) => {
      return morphMutation.mutateAsync({
        content,
        sourceModality: currentModality,
        targetModality,
        options,
      });
    },
    [currentModality, morphMutation]
  );

  // Quick generation functions
  const generatePodcast = useCallback(
    async (content: string, topic: string, options?: { durationMinutes?: number; style?: "educational" | "casual" | "debate" | "interview" }) => {
      return podcastMutation.mutateAsync({
        content,
        topic,
        ...options,
      });
    },
    [podcastMutation]
  );

  const generateDiagram = useCallback(
    async (content: string, options?: { diagramType?: DiagramType; title?: string; focusConcepts?: string[] }) => {
      return diagramMutation.mutateAsync({
        content,
        ...options,
      });
    },
    [diagramMutation]
  );

  const resetState = useCallback(() => {
    if (userId && contentId) {
      return resetMutation.mutateAsync();
    }
  }, [userId, contentId, resetMutation]);

  return {
    // Current state
    currentModality,
    currentContent,
    setCurrentModality,
    setCurrentContent,

    // Server state
    conceptualState: conceptualState as ConceptualState | undefined,
    learningSummary: learningSummary as LearningSummary | undefined,
    recommendation: recommendation as ModalityRecommendation | undefined,

    // Loading states
    isLoading: morphMutation.isPending || podcastMutation.isPending || diagramMutation.isPending,
    isLoadingState,
    isLoadingSummary,
    isMorphing: morphMutation.isPending,
    isGeneratingPodcast: podcastMutation.isPending,
    isGeneratingDiagram: diagramMutation.isPending,

    // Errors
    morphError: morphMutation.error,
    podcastError: podcastMutation.error,
    diagramError: diagramMutation.error,

    // Actions
    switchModality,
    generatePodcast,
    generateDiagram,
    resetState,

    // Refetch functions
    refetchState,
    refetchSummary,
    refetchRecommendation,
  };
}

/**
 * Hook for generating diagrams
 */
export function useDiagramGenerator() {
  const [diagram, setDiagram] = useState<DiagramData | null>(null);

  const generateMutation = useMutation({
    mutationFn: multimodalApi.generateDiagram,
    onSuccess: (data) => setDiagram(data),
  });

  const generateFromConceptsMutation = useMutation({
    mutationFn: multimodalApi.generateDiagramFromConcepts,
    onSuccess: (data) => setDiagram(data),
  });

  return {
    diagram,
    isGenerating: generateMutation.isPending || generateFromConceptsMutation.isPending,
    error: generateMutation.error || generateFromConceptsMutation.error,
    generate: generateMutation.mutateAsync,
    generateFromConcepts: generateFromConceptsMutation.mutateAsync,
    clear: () => setDiagram(null),
  };
}

/**
 * Hook for generating podcasts
 */
export function usePodcastGenerator() {
  const [episode, setEpisode] = useState<PodcastEpisode | null>(null);

  const generateMutation = useMutation({
    mutationFn: multimodalApi.generatePodcast,
    onSuccess: (data) => setEpisode(data),
  });

  return {
    episode,
    isGenerating: generateMutation.isPending,
    error: generateMutation.error,
    generate: generateMutation.mutateAsync,
    clear: () => setEpisode(null),
  };
}

/**
 * Hook for fetching supported modalities and diagram types
 */
export function useMultimodalMetadata() {
  const modalitiesQuery = useQuery({
    queryKey: ["supportedModalities"],
    queryFn: multimodalApi.getSupportedModalities,
    staleTime: Infinity, // Rarely changes
  });

  const diagramTypesQuery = useQuery({
    queryKey: ["diagramTypes"],
    queryFn: multimodalApi.getDiagramTypes,
    staleTime: Infinity,
  });

  return {
    modalities: modalitiesQuery.data?.modalities || [],
    diagramTypes: diagramTypesQuery.data?.diagram_types || [],
    isLoading: modalitiesQuery.isLoading || diagramTypesQuery.isLoading,
  };
}
