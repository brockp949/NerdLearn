/**
 * PACER Learning Protocol Hooks
 *
 * React hooks for content classification, analogy critique,
 * evidence linking, and procedural progress tracking.
 */

import { useState, useCallback, useEffect } from "react";
import { pacerApi } from "@/lib/api";
import type {
  PACERType,
  ClassificationResult,
  TriageDecision,
  PACERContentItem,
  Analogy,
  CritiqueResult,
  CritiqueSubmission,
  ProceduralProgress,
  StepResult,
  UserPACERProfile,
  EvidenceLink,
} from "@/types/pacer";

// =============================================================================
// CONTENT CLASSIFICATION HOOK
// =============================================================================

interface UseClassifierOptions {
  autoClassifyOnChange?: boolean;
  debounceMs?: number;
}

export function useClassifier(options: UseClassifierOptions = {}) {
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [triageDecisions, setTriageDecisions] = useState<TriageDecision[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const classify = useCallback(
    async (content: string, context?: Record<string, unknown>) => {
      if (content.length < 10) {
        setError("Content must be at least 10 characters");
        return null;
      }

      setIsLoading(true);
      setError(null);

      try {
        const response = await pacerApi.classify({ content, context });
        setResult(response);
        return response;
      } catch (err: unknown) {
        const message =
          err instanceof Error ? err.message : "Classification failed";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const runTriageTree = useCallback(
    async (content: string, context?: Record<string, unknown>) => {
      if (content.length < 10) {
        setError("Content must be at least 10 characters");
        return null;
      }

      setIsLoading(true);
      setError(null);

      try {
        const response = await pacerApi.classifyTriage({ content, context });
        setTriageDecisions(response);
        return response;
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : "Triage failed";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const classifyBatch = useCallback(
    async (contents: string[], context?: Record<string, unknown>) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await pacerApi.classifyBatch(contents, context);
        return response;
      } catch (err: unknown) {
        const message =
          err instanceof Error ? err.message : "Batch classification failed";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const reset = useCallback(() => {
    setResult(null);
    setTriageDecisions([]);
    setError(null);
  }, []);

  return {
    result,
    triageDecisions,
    isLoading,
    error,
    classify,
    runTriageTree,
    classifyBatch,
    reset,
  };
}

// =============================================================================
// ANALOGY CRITIQUE HOOK
// =============================================================================

interface UseAnalogyCritiqueOptions {
  userId: number;
}

export function useAnalogyCritique({ userId }: UseAnalogyCritiqueOptions) {
  const [analogy, setAnalogy] = useState<Analogy | null>(null);
  const [critiqueResult, setCritiqueResult] = useState<CritiqueResult | null>(
    null
  );
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showBreakdowns, setShowBreakdowns] = useState(false);

  const loadAnalogy = useCallback(
    async (analogyId: number, includeBreakdowns = false) => {
      setIsLoading(true);
      setError(null);
      setCritiqueResult(null);
      setShowBreakdowns(false);

      try {
        const response = await pacerApi.getAnalogy(analogyId, includeBreakdowns);
        setAnalogy(response);
        return response;
      } catch (err: unknown) {
        const message =
          err instanceof Error ? err.message : "Failed to load analogy";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const submitCritique = useCallback(
    async (submission: CritiqueSubmission) => {
      if (!analogy) {
        setError("No analogy loaded");
        return null;
      }

      setIsLoading(true);
      setError(null);

      try {
        const response = await pacerApi.submitCritique(
          analogy.id,
          submission,
          userId
        );
        setCritiqueResult(response);
        setShowBreakdowns(true);
        return response;
      } catch (err: unknown) {
        const message =
          err instanceof Error ? err.message : "Failed to submit critique";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [analogy, userId]
  );

  const revealBreakdowns = useCallback(() => {
    setShowBreakdowns(true);
  }, []);

  const reset = useCallback(() => {
    setAnalogy(null);
    setCritiqueResult(null);
    setShowBreakdowns(false);
    setError(null);
  }, []);

  return {
    analogy,
    critiqueResult,
    isLoading,
    error,
    showBreakdowns,
    loadAnalogy,
    submitCritique,
    revealBreakdowns,
    reset,
  };
}

// =============================================================================
// EVIDENCE LINKING HOOK
// =============================================================================

interface UseEvidenceLinkingOptions {
  conceptId?: number;
}

export function useEvidenceLinking(options: UseEvidenceLinkingOptions = {}) {
  const [evidence, setEvidence] = useState<
    Array<{
      evidenceId: number;
      title: string;
      content: string;
      relationshipType: string;
      strength: number;
    }>
  >([]);
  const [suggestions, setSuggestions] = useState<
    Array<{
      conceptId: number;
      conceptName: string;
      relevance: number;
      suggestedRelationship: string;
    }>
  >([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadEvidenceForConcept = useCallback(
    async (conceptId: number, relationshipType?: string) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await pacerApi.getEvidenceForConcept(
          conceptId,
          relationshipType
        );
        setEvidence(response.evidence || []);
        return response;
      } catch (err: unknown) {
        const message =
          err instanceof Error ? err.message : "Failed to load evidence";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const autoLinkEvidence = useCallback(
    async (content: string, courseId: number, minRelevance = 0.3) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await pacerApi.autoLinkEvidence(
          content,
          courseId,
          minRelevance
        );
        setSuggestions(response.suggestions || []);
        return response;
      } catch (err: unknown) {
        const message =
          err instanceof Error ? err.message : "Failed to get suggestions";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const createLinks = useCallback(
    async (
      evidenceItemId: number,
      conceptIds: number[],
      relationshipType = "supports",
      strength = 0.7,
      citation?: string
    ) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await pacerApi.linkEvidence({
          evidenceItemId,
          conceptIds,
          relationshipType,
          strength,
          citation,
        });
        return response;
      } catch (err: unknown) {
        const message =
          err instanceof Error ? err.message : "Failed to create links";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  // Auto-load evidence if conceptId is provided
  useEffect(() => {
    if (options.conceptId) {
      loadEvidenceForConcept(options.conceptId);
    }
  }, [options.conceptId, loadEvidenceForConcept]);

  return {
    evidence,
    suggestions,
    isLoading,
    error,
    loadEvidenceForConcept,
    autoLinkEvidence,
    createLinks,
  };
}

// =============================================================================
// PROCEDURAL PROGRESS HOOK
// =============================================================================

interface UseProceduralProgressOptions {
  userId: number;
  itemId?: number;
}

export function useProceduralProgress({
  userId,
  itemId,
}: UseProceduralProgressOptions) {
  const [progress, setProgress] = useState<ProceduralProgress | null>(null);
  const [stepResult, setStepResult] = useState<StepResult | null>(null);
  const [activeProcedures, setActiveProcedures] = useState<
    ProceduralProgress[]
  >([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startProcedure = useCallback(
    async (procedureItemId: number) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await pacerApi.startProcedure(procedureItemId, userId);
        // Load the status after starting
        const status = await pacerApi.getProceduralStatus(
          procedureItemId,
          userId
        );
        setProgress(status);
        return response;
      } catch (err: unknown) {
        const message =
          err instanceof Error ? err.message : "Failed to start procedure";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [userId]
  );

  const completeStep = useCallback(
    async (
      procedureItemId: number,
      stepNumber: number,
      success: boolean,
      timeMs: number,
      errorCount = 0
    ) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await pacerApi.completeStep(
          procedureItemId,
          { stepNumber, success, timeMs, errorCount },
          userId
        );
        setStepResult(response);

        // Refresh progress status
        const status = await pacerApi.getProceduralStatus(
          procedureItemId,
          userId
        );
        setProgress(status);

        return response;
      } catch (err: unknown) {
        const message =
          err instanceof Error ? err.message : "Failed to complete step";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [userId]
  );

  const loadStatus = useCallback(
    async (procedureItemId: number) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await pacerApi.getProceduralStatus(
          procedureItemId,
          userId
        );
        setProgress(response);
        return response;
      } catch (err: unknown) {
        const message =
          err instanceof Error ? err.message : "Failed to load status";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [userId]
  );

  const loadActiveProcedures = useCallback(
    async (includeCompleted = false) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await pacerApi.getActiveProcedures(
          userId,
          includeCompleted
        );
        setActiveProcedures(response.procedures || []);
        return response;
      } catch (err: unknown) {
        const message =
          err instanceof Error ? err.message : "Failed to load procedures";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [userId]
  );

  // Auto-load status if itemId is provided
  useEffect(() => {
    if (itemId) {
      loadStatus(itemId);
    }
  }, [itemId, loadStatus]);

  return {
    progress,
    stepResult,
    activeProcedures,
    isLoading,
    error,
    startProcedure,
    completeStep,
    loadStatus,
    loadActiveProcedures,
  };
}

// =============================================================================
// USER PACER PROFILE HOOK
// =============================================================================

interface UsePACERProfileOptions {
  userId: number;
  autoLoad?: boolean;
}

export function usePACERProfile({
  userId,
  autoLoad = false,
}: UsePACERProfileOptions) {
  const [profile, setProfile] = useState<UserPACERProfile | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadProfile = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await pacerApi.getProfile(userId);
      setProfile(response);
      return response;
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Failed to load profile";
      setError(message);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    if (autoLoad) {
      loadProfile();
    }
  }, [autoLoad, loadProfile]);

  const getProficiencyByType = useCallback(
    (type: PACERType): number => {
      if (!profile) return 0.5;
      const key = `${type}Proficiency` as keyof UserPACERProfile;
      return (profile[key] as number) || 0.5;
    },
    [profile]
  );

  const getStrongestType = useCallback((): PACERType | null => {
    if (!profile) return null;

    const proficiencies: Array<{ type: PACERType; value: number }> = [
      { type: "procedural", value: profile.proceduralProficiency },
      { type: "analogous", value: profile.analogousProficiency },
      { type: "conceptual", value: profile.conceptualProficiency },
      { type: "evidence", value: profile.evidenceProficiency },
      { type: "reference", value: profile.referenceProficiency },
    ];

    proficiencies.sort((a, b) => b.value - a.value);
    return proficiencies[0].type;
  }, [profile]);

  const getWeakestType = useCallback((): PACERType | null => {
    if (!profile) return null;

    const proficiencies: Array<{ type: PACERType; value: number }> = [
      { type: "procedural", value: profile.proceduralProficiency },
      { type: "analogous", value: profile.analogousProficiency },
      { type: "conceptual", value: profile.conceptualProficiency },
      { type: "evidence", value: profile.evidenceProficiency },
      { type: "reference", value: profile.referenceProficiency },
    ];

    proficiencies.sort((a, b) => a.value - b.value);
    return proficiencies[0].type;
  }, [profile]);

  return {
    profile,
    isLoading,
    error,
    loadProfile,
    refresh: loadProfile,
    getProficiencyByType,
    getStrongestType,
    getWeakestType,
  };
}

// =============================================================================
// COMBINED PACER HOOK
// =============================================================================

interface UsePACEROptions {
  userId: number;
  autoLoadProfile?: boolean;
}

export function usePACER({ userId, autoLoadProfile = true }: UsePACEROptions) {
  const classifier = useClassifier();
  const analogyCritique = useAnalogyCritique({ userId });
  const evidenceLinking = useEvidenceLinking();
  const proceduralProgress = useProceduralProgress({ userId });
  const profile = usePACERProfile({ userId, autoLoad: autoLoadProfile });

  return {
    classifier,
    analogyCritique,
    evidenceLinking,
    proceduralProgress,
    profile,
  };
}

export default usePACER;
