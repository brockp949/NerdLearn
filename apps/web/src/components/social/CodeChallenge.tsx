"use client";

import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Code2,
  Play,
  Loader2,
  CheckCircle,
  XCircle,
  AlertCircle,
  Lightbulb,
  ChevronDown,
  ChevronUp,
  Clock,
  Target,
  Zap,
  Shield,
  FileText,
  HelpCircle,
  RefreshCw,
  Trophy,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { socialApi } from "@/lib/api";
import type {
  CodingChallenge,
  CodingDifficultyLevel,
  EvaluationResult,
  HintLevel,
  HintResponse,
  EvaluationDimension,
} from "@/types/social";

// =============================================================================
// CONFIGURATION
// =============================================================================

const difficultyConfig: Record<CodingDifficultyLevel, {
  label: string;
  color: string;
  bgColor: string;
}> = {
  beginner: { label: "Beginner", color: "text-green-400", bgColor: "bg-green-500/20" },
  intermediate: { label: "Intermediate", color: "text-blue-400", bgColor: "bg-blue-500/20" },
  advanced: { label: "Advanced", color: "text-orange-400", bgColor: "bg-orange-500/20" },
  expert: { label: "Expert", color: "text-red-400", bgColor: "bg-red-500/20" },
};

const dimensionConfig: Record<EvaluationDimension, {
  label: string;
  icon: React.ElementType;
  color: string;
}> = {
  correctness: { label: "Correctness", icon: CheckCircle, color: "text-green-400" },
  quality: { label: "Quality", icon: Code2, color: "text-blue-400" },
  efficiency: { label: "Efficiency", icon: Zap, color: "text-yellow-400" },
  security: { label: "Security", icon: Shield, color: "text-red-400" },
  completeness: { label: "Completeness", icon: Target, color: "text-purple-400" },
  documentation: { label: "Documentation", icon: FileText, color: "text-cyan-400" },
};

const hintLevelConfig: Record<HintLevel, { label: string; cost: number }> = {
  nudge: { label: "Nudge", cost: 5 },
  guidance: { label: "Guidance", cost: 15 },
  explanation: { label: "Explanation", cost: 30 },
  partial: { label: "Partial Solution", cost: 50 },
  solution: { label: "Full Solution", cost: 100 },
};

// =============================================================================
// TYPES
// =============================================================================

interface CodeChallengeProps {
  challengeId?: string;
  userId: string;
  onComplete?: (result: EvaluationResult) => void;
  className?: string;
}

// =============================================================================
// COMPONENT
// =============================================================================

export function CodeChallengeComponent({
  challengeId: initialChallengeId,
  userId,
  onComplete,
  className = "",
}: CodeChallengeProps) {
  // State
  const [availableChallenges, setAvailableChallenges] = useState<Array<{
    challenge_id: string;
    title: string;
    difficulty: CodingDifficultyLevel;
    concepts: string[];
    estimated_minutes: number;
  }>>([]);
  const [selectedChallengeId, setSelectedChallengeId] = useState<string | null>(initialChallengeId || null);
  const [challenge, setChallenge] = useState<CodingChallenge | null>(null);
  const [code, setCode] = useState("");
  const [result, setResult] = useState<EvaluationResult | null>(null);
  const [hints, setHints] = useState<HintResponse[]>([]);
  const [currentHintLevel, setCurrentHintLevel] = useState<HintLevel>("nudge");
  const [showHints, setShowHints] = useState(false);
  const [showTestCases, setShowTestCases] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [isGettingHint, setIsGettingHint] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load challenges
  useEffect(() => {
    const loadChallenges = async () => {
      try {
        // Initialize sample challenges first
        await socialApi.initSampleChallenges();
        const response = await socialApi.listChallenges();
        setAvailableChallenges(response.challenges || []);
      } catch (err) {
        console.error("Failed to load challenges:", err);
      }
    };
    loadChallenges();
  }, []);

  // Load challenge details
  useEffect(() => {
    if (!selectedChallengeId) return;

    const loadChallenge = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const challengeData = await socialApi.getChallenge(selectedChallengeId);
        setChallenge(challengeData);

        // Set initial code template
        const params = challengeData.parameters
          .map((p: { name: string; type: string }) => p.name)
          .join(", ");
        setCode(`def ${challengeData.function_name}(${params}):\n    # Your code here\n    pass\n`);
      } catch (err: any) {
        setError(err.response?.data?.detail || "Failed to load challenge");
      } finally {
        setIsLoading(false);
      }
    };

    loadChallenge();
  }, [selectedChallengeId]);

  // Submit code
  const submitCode = useCallback(async () => {
    if (!challenge || !code.trim() || isEvaluating) return;

    setIsEvaluating(true);
    setError(null);
    setResult(null);

    try {
      const evaluationResult = await socialApi.submitCode({
        challenge_id: challenge.challenge_id,
        user_id: userId,
        code,
      });

      setResult(evaluationResult);
      onComplete?.(evaluationResult);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to evaluate code");
    } finally {
      setIsEvaluating(false);
    }
  }, [challenge, code, userId, isEvaluating, onComplete]);

  // Get hint
  const getHint = useCallback(async (level: HintLevel) => {
    if (!challenge || isGettingHint) return;

    setIsGettingHint(true);
    setError(null);

    try {
      const hint = await socialApi.getHint({
        challenge_id: challenge.challenge_id,
        user_id: userId,
        code,
        hint_level: level,
      });

      setHints((prev) => [...prev, hint]);
      setCurrentHintLevel(level);
      setShowHints(true);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to get hint");
    } finally {
      setIsGettingHint(false);
    }
  }, [challenge, code, userId, isGettingHint]);

  // Reset
  const resetChallenge = () => {
    setSelectedChallengeId(null);
    setChallenge(null);
    setCode("");
    setResult(null);
    setHints([]);
    setCurrentHintLevel("nudge");
    setError(null);
  };

  // ==========================================================================
  // RENDER: Challenge Selection
  // ==========================================================================
  if (!selectedChallengeId || !challenge) {
    return (
      <div className={`rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl overflow-hidden ${className}`}>
        <div className="p-6 border-b border-white/10">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-emerald-500 to-cyan-500 flex items-center justify-center">
              <Code2 className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Coding Challenges</h2>
              <p className="text-white/60 text-sm">Test your skills with AI-powered evaluation</p>
            </div>
          </div>
        </div>

        <div className="p-6 space-y-4">
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-white/60" />
            </div>
          ) : availableChallenges.length === 0 ? (
            <div className="text-center py-8 text-white/50">
              No challenges available. Loading sample challenges...
            </div>
          ) : (
            availableChallenges.map((ch) => {
              const diffConfig = difficultyConfig[ch.difficulty];
              return (
                <button
                  key={ch.challenge_id}
                  onClick={() => setSelectedChallengeId(ch.challenge_id)}
                  className="w-full p-4 rounded-xl border border-white/10 hover:border-white/30 hover:bg-white/5 transition-all text-left"
                >
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold text-white">{ch.title}</h3>
                    <span className={`px-2 py-0.5 rounded-full text-xs ${diffConfig.bgColor} ${diffConfig.color}`}>
                      {diffConfig.label}
                    </span>
                  </div>
                  <div className="flex items-center gap-4 text-xs text-white/50">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      ~{ch.estimated_minutes} min
                    </span>
                    <span className="flex items-center gap-1">
                      <Target className="w-3 h-3" />
                      {ch.concepts.join(", ")}
                    </span>
                  </div>
                </button>
              );
            })
          )}
        </div>
      </div>
    );
  }

  // ==========================================================================
  // RENDER: Active Challenge
  // ==========================================================================
  const diffConfig = difficultyConfig[challenge.difficulty];

  return (
    <div className={`rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl overflow-hidden ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-white/10 bg-white/5">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="font-semibold text-white">{challenge.title}</h2>
              <span className={`px-2 py-0.5 rounded-full text-xs ${diffConfig.bgColor} ${diffConfig.color}`}>
                {diffConfig.label}
              </span>
            </div>
            <div className="flex items-center gap-3 text-xs text-white/50 mt-1">
              <span>~{challenge.estimated_minutes} min</span>
              <span>•</span>
              <span>{challenge.language}</span>
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={resetChallenge}>
            <X className="w-4 h-4" />
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x divide-white/10">
        {/* Left: Problem */}
        <div className="p-4 space-y-4 max-h-[600px] overflow-y-auto">
          {/* Description */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-2">Problem</h3>
            <p className="text-sm text-white/70 whitespace-pre-wrap">{challenge.description}</p>
          </div>

          {/* Function Signature */}
          <div className="p-3 rounded-lg bg-white/5 border border-white/10">
            <code className="text-xs text-green-400">
              def {challenge.function_name}(
              {challenge.parameters.map((p) => `${p.name}: ${p.type}`).join(", ")}
              ) → {challenge.return_type}
            </code>
          </div>

          {/* Test Cases */}
          <div>
            <button
              onClick={() => setShowTestCases(!showTestCases)}
              className="flex items-center gap-2 text-sm font-semibold text-white mb-2"
            >
              Test Cases
              {showTestCases ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            <AnimatePresence>
              {showTestCases && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="space-y-2"
                >
                  {challenge.test_cases.map((tc, i) => (
                    <div key={i} className="p-3 rounded-lg bg-white/5 border border-white/10 text-xs">
                      <div className="text-white/50 mb-1">{tc.description}</div>
                      <div className="font-mono">
                        <span className="text-blue-400">Input:</span>{" "}
                        <span className="text-white/80">{JSON.stringify(tc.input)}</span>
                      </div>
                      <div className="font-mono">
                        <span className="text-green-400">Expected:</span>{" "}
                        <span className="text-white/80">{JSON.stringify(tc.expected)}</span>
                      </div>
                    </div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Hints Section */}
          <div>
            <button
              onClick={() => setShowHints(!showHints)}
              className="flex items-center gap-2 text-sm font-semibold text-amber-400 mb-2"
            >
              <Lightbulb className="w-4 h-4" />
              Hints ({hints.length})
              {showHints ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            <AnimatePresence>
              {showHints && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="space-y-2"
                >
                  {hints.map((hint, i) => (
                    <div key={i} className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-sm text-white/80">
                      <span className="text-xs text-amber-400 font-medium">
                        {hintLevelConfig[hint.hint_level].label}:
                      </span>
                      <p className="mt-1">{hint.hint}</p>
                    </div>
                  ))}

                  {/* Get More Hints */}
                  <div className="flex flex-wrap gap-2">
                    {(Object.keys(hintLevelConfig) as HintLevel[]).map((level) => {
                      const config = hintLevelConfig[level];
                      const alreadyHave = hints.some((h) => h.hint_level === level);
                      return (
                        <Button
                          key={level}
                          variant="outline"
                          size="sm"
                          disabled={alreadyHave || isGettingHint}
                          onClick={() => getHint(level)}
                          className="text-xs"
                        >
                          {isGettingHint ? (
                            <Loader2 className="w-3 h-3 animate-spin mr-1" />
                          ) : (
                            <HelpCircle className="w-3 h-3 mr-1" />
                          )}
                          {config.label}
                        </Button>
                      );
                    })}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Right: Code Editor & Results */}
        <div className="p-4 space-y-4">
          {/* Code Editor */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-2">Your Solution</h3>
            <textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              className="w-full h-64 px-4 py-3 rounded-xl bg-black/50 border border-white/10 text-white font-mono text-sm resize-none focus:outline-none focus:border-white/30"
              spellCheck={false}
            />
          </div>

          {/* Actions */}
          <div className="flex gap-2">
            <Button
              onClick={submitCode}
              disabled={!code.trim() || isEvaluating}
              className="flex-1"
            >
              {isEvaluating ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin mr-2" />
                  Evaluating...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Run & Evaluate
                </>
              )}
            </Button>
            <Button
              variant="outline"
              onClick={() => {
                const params = challenge.parameters.map((p) => p.name).join(", ");
                setCode(`def ${challenge.function_name}(${params}):\n    # Your code here\n    pass\n`);
                setResult(null);
              }}
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>

          {error && (
            <div className="p-3 rounded-lg bg-red-500/20 border border-red-500/30 text-red-300 text-sm flex items-center gap-2">
              <AlertCircle className="w-4 h-4" />
              {error}
            </div>
          )}

          {/* Results */}
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-4"
            >
              {/* Overall Score */}
              <div className={`p-4 rounded-xl border ${result.passed ? "bg-green-500/10 border-green-500/30" : "bg-red-500/10 border-red-500/30"}`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {result.passed ? (
                      <Trophy className="w-8 h-8 text-green-400" />
                    ) : (
                      <XCircle className="w-8 h-8 text-red-400" />
                    )}
                    <div>
                      <div className={`text-2xl font-bold ${result.passed ? "text-green-400" : "text-red-400"}`}>
                        {Math.round(result.overall_score)}%
                      </div>
                      <div className="text-xs text-white/50">
                        {result.tests_passed}/{result.tests_total} tests passed
                      </div>
                    </div>
                  </div>
                  {result.execution_time_ms && (
                    <div className="text-right">
                      <div className="text-sm text-white/80">{result.execution_time_ms.toFixed(2)}ms</div>
                      <div className="text-xs text-white/50">execution time</div>
                    </div>
                  )}
                </div>
              </div>

              {/* Dimension Scores */}
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(result.dimension_scores).map(([dim, score]) => {
                  const config = dimensionConfig[dim as EvaluationDimension];
                  if (!config) return null;
                  const Icon = config.icon;
                  return (
                    <div key={dim} className="p-3 rounded-lg bg-white/5 border border-white/10">
                      <div className="flex items-center gap-2 mb-1">
                        <Icon className={`w-4 h-4 ${config.color}`} />
                        <span className="text-xs font-medium text-white/80">{config.label}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                          <div
                            className={`h-full ${score.score >= 70 ? "bg-green-500" : score.score >= 40 ? "bg-yellow-500" : "bg-red-500"}`}
                            style={{ width: `${score.score}%` }}
                          />
                        </div>
                        <span className="text-xs text-white/60">{Math.round(score.score)}%</span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Feedback */}
              {result.feedback.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-white">Feedback</h4>
                  {result.feedback.slice(0, 5).map((fb, i) => (
                    <div key={i} className="p-2 rounded-lg bg-white/5 border border-white/10 text-xs">
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`px-1.5 py-0.5 rounded ${fb.type === "praise" ? "bg-green-500/20 text-green-400" :
                          fb.type === "issue" ? "bg-red-500/20 text-red-400" :
                            "bg-blue-500/20 text-blue-400"
                          }`}>
                          {fb.type}
                        </span>
                        {fb.line_number && (
                          <span className="text-white/40">Line {fb.line_number}</span>
                        )}
                      </div>
                      <p className="text-white/70">{fb.message}</p>
                      {fb.suggestion && (
                        <p className="text-white/50 mt-1">Suggestion: {fb.suggestion}</p>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Runtime Errors */}
              {result.runtime_errors.length > 0 && (
                <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20">
                  <h4 className="text-sm font-semibold text-red-400 mb-2">Runtime Errors</h4>
                  {result.runtime_errors.map((err, i) => (
                    <pre key={i} className="text-xs text-red-300 font-mono whitespace-pre-wrap">{err}</pre>
                  ))}
                </div>
              )}

              {/* Concepts */}
              {(result.concepts_demonstrated.length > 0 || result.concepts_to_review.length > 0) && (
                <div className="flex flex-wrap gap-2">
                  {result.concepts_demonstrated.map((c, i) => (
                    <span key={`demo-${i}`} className="px-2 py-1 rounded-full text-xs bg-green-500/20 text-green-400">
                      ✓ {c}
                    </span>
                  ))}
                  {result.concepts_to_review.map((c, i) => (
                    <span key={`review-${i}`} className="px-2 py-1 rounded-full text-xs bg-amber-500/20 text-amber-400">
                      Review: {c}
                    </span>
                  ))}
                </div>
              )}
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}

export default CodeChallengeComponent;
