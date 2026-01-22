/**
 * Phase 1: Cognitive Foundation Hooks
 *
 * React hooks for frustration detection, metacognition, calibration, and interventions
 */

import { useState, useCallback, useRef, useEffect } from "react";
import { cognitiveApi } from "@/lib/api";
import type {
  FrustrationResponse,
  InteractionEvent,
  MetacognitionPromptResponse,
  CalibrationResponse,
  InterventionDecision,
  CognitiveProfile,
  SelfExplanationAnalysis,
  CalibrationFeedback,
  InterventionHistory,
} from "@/types/cognitive";

// =============================================================================
// FRUSTRATION DETECTION HOOK
// =============================================================================

interface UseFrustrationOptions {
  userId: string;
  autoDetect?: boolean;
  detectionInterval?: number; // ms
  minEventsForDetection?: number;
}

export function useFrustration({
  userId,
  autoDetect = false,
  detectionInterval = 30000,
  minEventsForDetection = 5,
}: UseFrustrationOptions) {
  const [frustration, setFrustration] = useState<FrustrationResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [eventCount, setEventCount] = useState(0);
  const eventsRef = useRef<InteractionEvent[]>([]);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const addEvent = useCallback((event: Omit<InteractionEvent, "timestamp">) => {
    eventsRef.current.push({
      ...event,
      timestamp: new Date().toISOString(),
    });
    // Keep only last 50 events
    if (eventsRef.current.length > 50) {
      eventsRef.current = eventsRef.current.slice(-50);
    }
    setEventCount(eventsRef.current.length);
  }, []);

  const detectFrustration = useCallback(
    async (events?: InteractionEvent[], context?: Record<string, unknown>) => {
      const eventsToAnalyze = events || eventsRef.current;
      if (eventsToAnalyze.length < minEventsForDetection) {
        return null;
      }

      setIsLoading(true);
      setError(null);

      try {
        const response = await cognitiveApi.detectFrustration({
          user_id: userId,
          events: eventsToAnalyze,
          context,
        });
        setFrustration(response);
        return response;
      } catch (err: any) {
        const message = err.response?.data?.detail || "Failed to detect frustration";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [userId, minEventsForDetection]
  );

  const updateBaseline = useCallback(
    async (events: InteractionEvent[]) => {
      try {
        return await cognitiveApi.updateBaseline(userId, events);
      } catch (err: any) {
        setError(err.response?.data?.detail || "Failed to update baseline");
        return null;
      }
    },
    [userId]
  );

  const clearEvents = useCallback(() => {
    eventsRef.current = [];
    setEventCount(0);
  }, []);

  // Auto-detection interval
  useEffect(() => {
    if (autoDetect) {
      intervalRef.current = setInterval(() => {
        if (eventsRef.current.length >= minEventsForDetection) {
          detectFrustration();
        }
      }, detectionInterval);

      return () => {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      };
    }
  }, [autoDetect, detectionInterval, detectFrustration, minEventsForDetection]);

  return {
    frustration,
    isLoading,
    error,
    addEvent,
    detectFrustration,
    updateBaseline,
    clearEvents,
    eventCount,
  };
}

// =============================================================================
// METACOGNITION HOOK
// =============================================================================

interface UseMetacognitionOptions {
  userId: string;
}

export function useMetacognition({ userId }: UseMetacognitionOptions) {
  const [prompt, setPrompt] = useState<MetacognitionPromptResponse | null>(null);
  const [analysis, setAnalysis] = useState<SelfExplanationAnalysis | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const getPrompt = useCallback(
    async (
      conceptName: string,
      timing: "before" | "during" | "after",
      context?: Record<string, unknown>,
      force?: boolean
    ) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await cognitiveApi.getMetacognitionPrompt({
          user_id: userId,
          concept_name: conceptName,
          timing,
          context,
          force,
        });
        setPrompt(response);
        return response;
      } catch (err: any) {
        const message = err.response?.data?.detail || "Failed to get prompt";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [userId]
  );

  const getConfidenceScale = useCallback(
    async (conceptName: string, scaleType: "numeric" | "verbal" | "emoji" = "numeric") => {
      try {
        return await cognitiveApi.getConfidenceScale(conceptName, scaleType);
      } catch (err: any) {
        setError(err.response?.data?.detail || "Failed to get scale");
        return null;
      }
    },
    []
  );

  const recordConfidence = useCallback(
    async (conceptId: string, contentId: string, confidence: number, context?: string) => {
      try {
        return await cognitiveApi.recordConfidence({
          user_id: userId,
          concept_id: conceptId,
          content_id: contentId,
          confidence,
          context,
        });
      } catch (err: any) {
        setError(err.response?.data?.detail || "Failed to record confidence");
        return null;
      }
    },
    [userId]
  );

  const analyzeExplanation = useCallback(
    async (
      explanationText: string,
      conceptName: string,
      expectedConcepts?: string[],
      commonMisconceptions?: string[]
    ) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await cognitiveApi.analyzeExplanation({
          explanation_text: explanationText,
          concept_name: conceptName,
          expected_concepts: expectedConcepts,
          common_misconceptions: commonMisconceptions,
        });
        setAnalysis(response);
        return response;
      } catch (err: any) {
        const message = err.response?.data?.detail || "Failed to analyze explanation";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  return {
    prompt,
    analysis,
    isLoading,
    error,
    getPrompt,
    getConfidenceScale,
    recordConfidence,
    analyzeExplanation,
  };
}

// =============================================================================
// CALIBRATION HOOK
// =============================================================================

interface UseCalibrationOptions {
  userId: string;
}

export function useCalibration({ userId }: UseCalibrationOptions) {
  const [calibration, setCalibration] = useState<CalibrationResponse | null>(null);
  const [feedback, setFeedback] = useState<CalibrationFeedback | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const calculateCalibration = useCallback(
    async (conceptId?: string, timeWindowHours?: number) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await cognitiveApi.calculateCalibration({
          user_id: userId,
          concept_id: conceptId,
          time_window_hours: timeWindowHours,
        });
        setCalibration(response);
        return response;
      } catch (err: any) {
        const message = err.response?.data?.detail || "Failed to calculate calibration";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [userId]
  );

  const getCalibrationFeedback = useCallback(
    async (conceptId?: string, timeWindowHours?: number) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await cognitiveApi.getCalibrationFeedback({
          user_id: userId,
          concept_id: conceptId,
          time_window_hours: timeWindowHours,
        });
        setFeedback(response);
        return response;
      } catch (err: any) {
        const message = err.response?.data?.detail || "Failed to get feedback";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [userId]
  );

  const updatePerformance = useCallback(
    async (conceptId: string, actualPerformance: number) => {
      try {
        return await cognitiveApi.updatePerformance(userId, conceptId, actualPerformance);
      } catch (err: any) {
        setError(err.response?.data?.detail || "Failed to update performance");
        return null;
      }
    },
    [userId]
  );

  return {
    calibration,
    feedback,
    isLoading,
    error,
    calculateCalibration,
    getCalibrationFeedback,
    updatePerformance,
  };
}

// =============================================================================
// INTERVENTION HOOK
// =============================================================================

interface UseInterventionOptions {
  userId: string;
  onIntervention?: (intervention: InterventionDecision) => void;
}

export function useIntervention({ userId, onIntervention }: UseInterventionOptions) {
  const [decision, setDecision] = useState<InterventionDecision | null>(null);
  const [history, setHistory] = useState<InterventionHistory | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const lastInterventionRef = useRef<Date | null>(null);

  const checkIntervention = useCallback(
    async (
      learnerState: {
        frustration_score?: number;
        frustration_level?: string;
        cognitive_load_score?: number;
        cognitive_load_level?: string;
        calibration_level?: string;
        consecutive_errors?: number;
        time_on_task_minutes?: number;
        session_duration_minutes?: number;
        concepts_mastered_today?: number;
      },
      events?: InteractionEvent[],
      context?: Record<string, unknown>
    ) => {
      // Check cooldown
      if (lastInterventionRef.current) {
        const elapsed = Date.now() - lastInterventionRef.current.getTime();
        if (elapsed < 30000) {
          // 30 second minimum cooldown
          return null;
        }
      }

      setIsLoading(true);
      setError(null);

      try {
        const response = await cognitiveApi.decideIntervention({
          learner_state: {
            user_id: userId,
            ...learnerState,
          },
          events,
          context,
        });

        setDecision(response);

        if (response.should_intervene) {
          lastInterventionRef.current = new Date();
          onIntervention?.(response);
        }

        return response;
      } catch (err: any) {
        const message = err.response?.data?.detail || "Failed to check intervention";
        setError(message);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [userId, onIntervention]
  );

  const getHistory = useCallback(async () => {
    try {
      const response = await cognitiveApi.getInterventionHistory(userId);
      setHistory(response);
      return response;
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to get history");
      return null;
    }
  }, [userId]);

  const dismissIntervention = useCallback(() => {
    setDecision(null);
  }, []);

  return {
    decision,
    history,
    isLoading,
    error,
    checkIntervention,
    getHistory,
    dismissIntervention,
  };
}

// =============================================================================
// COGNITIVE PROFILE HOOK
// =============================================================================

interface UseCognitiveProfileOptions {
  userId: string;
  autoLoad?: boolean;
}

export function useCognitiveProfile({ userId, autoLoad = false }: UseCognitiveProfileOptions) {
  const [profile, setProfile] = useState<CognitiveProfile | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadProfile = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await cognitiveApi.getCognitiveProfile(userId);
      setProfile(response);
      return response;
    } catch (err: any) {
      const message = err.response?.data?.detail || "Failed to load profile";
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

  return {
    profile,
    isLoading,
    error,
    loadProfile,
    refresh: loadProfile,
  };
}

// =============================================================================
// COMBINED COGNITIVE STATE HOOK
// =============================================================================

interface UseCognitiveStateOptions {
  userId: string;
  autoDetect?: boolean;
}

export function useCognitiveState({ userId, autoDetect = true }: UseCognitiveStateOptions) {
  const frustrationHook = useFrustration({ userId, autoDetect });
  const calibrationHook = useCalibration({ userId });
  const interventionHook = useIntervention({ userId });
  const profileHook = useCognitiveProfile({ userId, autoLoad: true });

  const getLearnerState = useCallback(() => {
    return {
      frustration_score: frustrationHook.frustration?.score ?? 0,
      frustration_level: frustrationHook.frustration?.level ?? "none",
      calibration_level: calibrationHook.calibration?.calibration_level ?? "unknown",
    };
  }, [frustrationHook.frustration, calibrationHook.calibration]);

  return {
    frustration: frustrationHook,
    calibration: calibrationHook,
    intervention: interventionHook,
    profile: profileHook,
    getLearnerState,
  };
}
