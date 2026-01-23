/**
 * Phase 4: Social Learning Hooks
 *
 * Custom React hooks for social learning features:
 * 1. useTeaching - Teachable Agent (Feynman Protocol)
 * 2. useDebate - SimClass Debates
 * 3. useCodeChallenge - Code Evaluator
 */

import { useState, useCallback } from "react";
import { socialApi } from "@/lib/api";
import type {
  StudentPersona,
  TeachingSessionResponse,
  TeachingResponse,
  TeachingSessionSummary,
  DebateFormat,
  PanelPreset,
  DebateSessionResponse,
  DebateRoundResponse,
  DebateSummary,
  CodingDifficultyLevel as DifficultyLevel,
  HintLevel,
  CodingChallenge,
  EvaluationResult,
  HintResponse,
} from "@/types/social";

// =============================================================================
// TEACHABLE AGENT HOOK
// =============================================================================

interface UseTeachingOptions {
  userId: string;
  onSessionEnd?: (summary: TeachingSessionSummary) => void;
}

export function useTeaching({ userId, onSessionEnd }: UseTeachingOptions) {
  const [session, setSession] = useState<TeachingSessionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startSession = useCallback(
    async (conceptId: string, conceptName: string, persona?: StudentPersona) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await socialApi.startTeachingSession({
          user_id: userId,
          concept_id: conceptId,
          concept_name: conceptName,
          persona,
        });
        setSession(response);
        return response;
      } catch (err: any) {
        const message = err.response?.data?.detail || "Failed to start session";
        setError(message);
        throw new Error(message);
      } finally {
        setIsLoading(false);
      }
    },
    [userId]
  );

  const submitExplanation = useCallback(
    async (explanation: string): Promise<TeachingResponse> => {
      if (!session) throw new Error("No active session");

      setIsLoading(true);
      setError(null);

      try {
        const response = await socialApi.submitExplanation({
          session_id: session.session_id,
          explanation,
        });
        return response;
      } catch (err: any) {
        const message = err.response?.data?.detail || "Failed to submit explanation";
        setError(message);
        throw new Error(message);
      } finally {
        setIsLoading(false);
      }
    },
    [session]
  );

  const endSession = useCallback(async (): Promise<TeachingSessionSummary> => {
    if (!session) throw new Error("No active session");

    setIsLoading(true);
    setError(null);

    try {
      const summary = await socialApi.endTeachingSession(session.session_id);
      onSessionEnd?.(summary);
      setSession(null);
      return summary;
    } catch (err: any) {
      const message = err.response?.data?.detail || "Failed to end session";
      setError(message);
      throw new Error(message);
    } finally {
      setIsLoading(false);
    }
  }, [session, onSessionEnd]);

  return {
    session,
    isLoading,
    error,
    startSession,
    submitExplanation,
    endSession,
  };
}

// =============================================================================
// DEBATE HOOK
// =============================================================================

interface UseDebateOptions {
  learnerId?: string;
  onComplete?: (summary: DebateSummary) => void;
}

export function useDebate({ learnerId, onComplete }: UseDebateOptions = {}) {
  const [session, setSession] = useState<DebateSessionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startDebate = useCallback(
    async (topic: string, format?: DebateFormat, preset?: PanelPreset) => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await socialApi.startDebate({
          topic,
          format,
          panel_preset: preset,
          learner_id: learnerId,
        });
        setSession(response);
        return response;
      } catch (err: any) {
        const message = err.response?.data?.detail || "Failed to start debate";
        setError(message);
        throw new Error(message);
      } finally {
        setIsLoading(false);
      }
    },
    [learnerId]
  );

  const advanceDebate = useCallback(
    async (contribution?: string): Promise<DebateRoundResponse> => {
      if (!session) throw new Error("No active session");

      setIsLoading(true);
      setError(null);

      try {
        const response = await socialApi.advanceDebate({
          session_id: session.session_id,
          learner_contribution: contribution,
        });

        if (response.completed) {
          const summary = await socialApi.getDebateSummary(session.session_id);
          onComplete?.(summary);
        }

        return response;
      } catch (err: any) {
        const message = err.response?.data?.detail || "Failed to advance debate";
        setError(message);
        throw new Error(message);
      } finally {
        setIsLoading(false);
      }
    },
    [session, onComplete]
  );

  const getSummary = useCallback(async (): Promise<DebateSummary> => {
    if (!session) throw new Error("No active session");

    try {
      return await socialApi.getDebateSummary(session.session_id);
    } catch (err: any) {
      const message = err.response?.data?.detail || "Failed to get summary";
      setError(message);
      throw new Error(message);
    }
  }, [session]);

  return {
    session,
    isLoading,
    error,
    startDebate,
    advanceDebate,
    getSummary,
  };
}

// =============================================================================
// CODE CHALLENGE HOOK
// =============================================================================

interface UseCodeChallengeOptions {
  userId: string;
  onComplete?: (result: EvaluationResult) => void;
}

export function useCodeChallenge({ userId, onComplete }: UseCodeChallengeOptions) {
  const [challenge, setChallenge] = useState<CodingChallenge | null>(null);
  const [result, setResult] = useState<EvaluationResult | null>(null);
  const [hints, setHints] = useState<HintResponse[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadChallenge = useCallback(async (challengeId: string) => {
    setIsLoading(true);
    setError(null);
    setResult(null);
    setHints([]);

    try {
      const response = await socialApi.getChallenge(challengeId);
      setChallenge(response);
      return response;
    } catch (err: any) {
      const message = err.response?.data?.detail || "Failed to load challenge";
      setError(message);
      throw new Error(message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const submitCode = useCallback(
    async (code: string): Promise<EvaluationResult> => {
      if (!challenge) throw new Error("No challenge loaded");

      setIsLoading(true);
      setError(null);

      try {
        const response = await socialApi.submitCode({
          challenge_id: challenge.challenge_id,
          user_id: userId,
          code,
        });
        setResult(response);
        onComplete?.(response);
        return response;
      } catch (err: any) {
        const message = err.response?.data?.detail || "Failed to submit code";
        setError(message);
        throw new Error(message);
      } finally {
        setIsLoading(false);
      }
    },
    [challenge, userId, onComplete]
  );

  const getHint = useCallback(
    async (code: string, level?: HintLevel): Promise<HintResponse> => {
      if (!challenge) throw new Error("No challenge loaded");

      setIsLoading(true);
      setError(null);

      try {
        const response = await socialApi.getHint({
          challenge_id: challenge.challenge_id,
          user_id: userId,
          code,
          hint_level: level,
        });
        setHints((prev) => [...prev, response]);
        return response;
      } catch (err: any) {
        const message = err.response?.data?.detail || "Failed to get hint";
        setError(message);
        throw new Error(message);
      } finally {
        setIsLoading(false);
      }
    },
    [challenge, userId]
  );

  const reset = useCallback(() => {
    setChallenge(null);
    setResult(null);
    setHints([]);
    setError(null);
  }, []);

  return {
    challenge,
    result,
    hints,
    isLoading,
    error,
    loadChallenge,
    submitCode,
    getHint,
    reset,
  };
}

// =============================================================================
// UTILITY HOOKS
// =============================================================================

export function useChallengeList() {
  const [challenges, setChallenges] = useState<Array<{
    challenge_id: string;
    title: string;
    difficulty: DifficultyLevel;
    concepts: string[];
    estimated_minutes: number;
  }>>([]);
  const [isLoading, setIsLoading] = useState(false);

  const loadChallenges = useCallback(async () => {
    setIsLoading(true);
    try {
      await socialApi.initSampleChallenges();
      const response = await socialApi.listChallenges();
      setChallenges(response.challenges || []);
      return response.challenges;
    } catch (err) {
      console.error("Failed to load challenges:", err);
      return [];
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { challenges, isLoading, loadChallenges };
}
