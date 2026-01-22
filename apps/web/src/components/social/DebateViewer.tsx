"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Users,
  MessageSquare,
  Send,
  Loader2,
  Play,
  Pause,
  SkipForward,
  FileText,
  AlertCircle,
  CheckCircle,
  Lightbulb,
  Target,
  Zap,
  Scale,
  BookOpen,
  Rocket,
  Wrench,
  GraduationCap,
  RefreshCw,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { socialApi } from "@/lib/api";
import type {
  DebateRole,
  DebateFormat,
  PanelPreset,
  DebateArgument,
  DebateSessionResponse,
  DebateRoundResponse,
  DebateSummary,
  DebateParticipant,
} from "@/types/social";

// =============================================================================
// CONFIGURATION
// =============================================================================

const roleConfig: Record<DebateRole, {
  label: string;
  icon: React.ElementType;
  color: string;
  bgColor: string;
}> = {
  advocate: { label: "Advocate", icon: Target, color: "text-green-400", bgColor: "bg-green-500/20" },
  skeptic: { label: "Skeptic", icon: AlertCircle, color: "text-red-400", bgColor: "bg-red-500/20" },
  synthesizer: { label: "Synthesizer", icon: Scale, color: "text-purple-400", bgColor: "bg-purple-500/20" },
  historian: { label: "Historian", icon: BookOpen, color: "text-amber-400", bgColor: "bg-amber-500/20" },
  futurist: { label: "Futurist", icon: Rocket, color: "text-cyan-400", bgColor: "bg-cyan-500/20" },
  practitioner: { label: "Practitioner", icon: Wrench, color: "text-orange-400", bgColor: "bg-orange-500/20" },
  theorist: { label: "Theorist", icon: GraduationCap, color: "text-blue-400", bgColor: "bg-blue-500/20" },
  contrarian: { label: "Contrarian", icon: RefreshCw, color: "text-pink-400", bgColor: "bg-pink-500/20" },
};

const formatConfig: Record<DebateFormat, { label: string; description: string }> = {
  oxford: { label: "Oxford", description: "Formal pro/con debate structure" },
  socratic: { label: "Socratic", description: "Question-driven exploration" },
  roundtable: { label: "Roundtable", description: "Open discussion among equals" },
  devils_advocate: { label: "Devil's Advocate", description: "Challenge the consensus" },
  synthesis: { label: "Synthesis", description: "Find common ground" },
};

const presetConfig: Record<PanelPreset, { label: string; description: string }> = {
  technical_pros_cons: { label: "Technical Pros/Cons", description: "Advocate, Skeptic, Synthesizer" },
  philosophical: { label: "Philosophical", description: "Theorist, Historian, Futurist, Contrarian" },
  practical_application: { label: "Practical Application", description: "Practitioner, Theorist, Advocate" },
};

// =============================================================================
// TYPES
// =============================================================================

interface DebateViewerProps {
  learnerId?: string;
  onComplete?: (summary: DebateSummary) => void;
  className?: string;
}

// =============================================================================
// COMPONENT
// =============================================================================

export function DebateViewer({
  learnerId,
  onComplete,
  className = "",
}: DebateViewerProps) {
  // State
  const [topic, setTopic] = useState("");
  const [selectedFormat, setSelectedFormat] = useState<DebateFormat>("roundtable");
  const [selectedPreset, setSelectedPreset] = useState<PanelPreset>("technical_pros_cons");
  const [session, setSession] = useState<DebateSessionResponse | null>(null);
  const [arguments_, setArguments] = useState<DebateArgument[]>([]);
  const [currentRound, setCurrentRound] = useState(1);
  const [maxRounds, setMaxRounds] = useState(5);
  const [isCompleted, setIsCompleted] = useState(false);
  const [summary, setSummary] = useState<DebateSummary | null>(null);
  const [learnerInput, setLearnerInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isAdvancing, setIsAdvancing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const argumentsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll
  useEffect(() => {
    argumentsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [arguments_]);

  // Start debate
  const startDebate = useCallback(async () => {
    if (!topic.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await socialApi.startDebate({
        topic: topic.trim(),
        format: selectedFormat,
        panel_preset: selectedPreset,
        learner_id: learnerId,
        max_rounds: 5,
      });

      setSession(response);
      setArguments(response.opening_statements);
      setCurrentRound(response.current_round);
      setMaxRounds(response.max_rounds);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to start debate");
    } finally {
      setIsLoading(false);
    }
  }, [topic, selectedFormat, selectedPreset, learnerId]);

  // Advance debate
  const advanceDebate = useCallback(async (contribution?: string) => {
    if (!session || isAdvancing) return;

    setIsAdvancing(true);
    setError(null);

    try {
      const response: DebateRoundResponse = await socialApi.advanceDebate({
        session_id: session.session_id,
        learner_contribution: contribution,
      });

      setArguments((prev) => [...prev, ...response.arguments]);
      setCurrentRound(response.current_round);
      setIsCompleted(response.completed);
      setLearnerInput("");

      if (response.completed) {
        // Fetch summary
        const summaryResponse = await socialApi.getDebateSummary(session.session_id);
        setSummary(summaryResponse);
        onComplete?.(summaryResponse);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to advance debate");
    } finally {
      setIsAdvancing(false);
    }
  }, [session, isAdvancing, onComplete]);

  // Reset
  const resetDebate = () => {
    setTopic("");
    setSession(null);
    setArguments([]);
    setCurrentRound(1);
    setIsCompleted(false);
    setSummary(null);
    setLearnerInput("");
    setError(null);
  };

  // ==========================================================================
  // RENDER: Summary View
  // ==========================================================================
  if (summary) {
    return (
      <div className={`rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl overflow-hidden ${className}`}>
        <div className="p-6 border-b border-white/10 bg-gradient-to-r from-purple-500/20 to-blue-500/20">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-purple-500/20 flex items-center justify-center">
                <FileText className="w-6 h-6 text-purple-400" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">Debate Summary</h2>
                <p className="text-white/60 text-sm">{summary.topic}</p>
              </div>
            </div>
            <Button variant="ghost" size="icon" onClick={resetDebate}>
              <X className="w-5 h-5" />
            </Button>
          </div>
        </div>

        <div className="p-6 space-y-6 max-h-[600px] overflow-y-auto">
          {/* Executive Summary */}
          <div className="p-4 rounded-xl bg-white/5 border border-white/10">
            <p className="text-white/80">{summary.executive_summary}</p>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-white/5 border border-white/10 text-center">
              <div className="text-2xl font-bold text-blue-400">{summary.total_rounds}</div>
              <div className="text-xs text-white/50">Rounds</div>
            </div>
            <div className="p-4 rounded-xl bg-white/5 border border-white/10 text-center">
              <div className="text-2xl font-bold text-purple-400">{summary.total_arguments}</div>
              <div className="text-xs text-white/50">Arguments</div>
            </div>
            <div className="p-4 rounded-xl bg-white/5 border border-white/10 text-center">
              <div className="text-2xl font-bold text-green-400">{summary.participants.length}</div>
              <div className="text-xs text-white/50">Participants</div>
            </div>
          </div>

          {/* Key Insights */}
          {summary.key_insights.length > 0 && (
            <div className="p-4 rounded-xl bg-blue-500/10 border border-blue-500/20">
              <div className="flex items-center gap-2 mb-3">
                <Lightbulb className="w-5 h-5 text-blue-400" />
                <h3 className="font-semibold text-blue-400">Key Insights</h3>
              </div>
              <ul className="space-y-2">
                {summary.key_insights.map((insight, i) => (
                  <li key={i} className="text-sm text-white/70 flex items-start gap-2">
                    <span className="text-blue-400 mt-1">•</span>
                    {insight}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Consensus & Disagreement */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {summary.consensus_points.length > 0 && (
              <div className="p-4 rounded-xl bg-green-500/10 border border-green-500/20">
                <div className="flex items-center gap-2 mb-3">
                  <CheckCircle className="w-5 h-5 text-green-400" />
                  <h3 className="font-semibold text-green-400">Consensus</h3>
                </div>
                <ul className="space-y-1">
                  {summary.consensus_points.map((point, i) => (
                    <li key={i} className="text-xs text-white/70">• {point}</li>
                  ))}
                </ul>
              </div>
            )}

            {summary.disagreement_points.length > 0 && (
              <div className="p-4 rounded-xl bg-orange-500/10 border border-orange-500/20">
                <div className="flex items-center gap-2 mb-3">
                  <Zap className="w-5 h-5 text-orange-400" />
                  <h3 className="font-semibold text-orange-400">Disagreements</h3>
                </div>
                <ul className="space-y-1">
                  {summary.disagreement_points.map((point, i) => (
                    <li key={i} className="text-xs text-white/70">• {point}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Learning Takeaways */}
          {summary.learning_takeaways.length > 0 && (
            <div className="p-4 rounded-xl bg-purple-500/10 border border-purple-500/20">
              <div className="flex items-center gap-2 mb-3">
                <GraduationCap className="w-5 h-5 text-purple-400" />
                <h3 className="font-semibold text-purple-400">Learning Takeaways</h3>
              </div>
              <ul className="space-y-2">
                {summary.learning_takeaways.map((takeaway, i) => (
                  <li key={i} className="text-sm text-white/70 flex items-start gap-2">
                    <span className="text-purple-400 mt-1">{i + 1}.</span>
                    {takeaway}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <Button onClick={resetDebate} className="w-full" variant="outline">
            Start New Debate
          </Button>
        </div>
      </div>
    );
  }

  // ==========================================================================
  // RENDER: Topic Selection
  // ==========================================================================
  if (!session) {
    return (
      <div className={`rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl overflow-hidden ${className}`}>
        <div className="p-6 border-b border-white/10">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center">
              <Users className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">SimClass Debate</h2>
              <p className="text-white/60 text-sm">Multi-agent perspective exploration</p>
            </div>
          </div>
        </div>

        <div className="p-6 space-y-6">
          {error && (
            <div className="p-3 rounded-lg bg-red-500/20 border border-red-500/30 text-red-300 text-sm flex items-center gap-2">
              <AlertCircle className="w-4 h-4" />
              {error}
            </div>
          )}

          {/* Topic Input */}
          <div>
            <label className="block text-sm font-medium text-white/80 mb-2">
              Debate Topic
            </label>
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g., Is AI dangerous? or Should we colonize Mars?"
              className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/30 focus:outline-none focus:border-white/30"
            />
          </div>

          {/* Format Selection */}
          <div>
            <label className="block text-sm font-medium text-white/80 mb-2">
              Debate Format
            </label>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              {(Object.entries(formatConfig) as [DebateFormat, typeof formatConfig[DebateFormat]][]).map(
                ([format, config]) => (
                  <button
                    key={format}
                    onClick={() => setSelectedFormat(format)}
                    className={`
                      p-3 rounded-xl border text-left transition-all
                      ${selectedFormat === format
                        ? "border-purple-500 bg-purple-500/20"
                        : "border-white/10 hover:border-white/30 hover:bg-white/5"
                      }
                    `}
                  >
                    <div className="font-medium text-sm text-white">{config.label}</div>
                    <div className="text-xs text-white/50 mt-0.5">{config.description}</div>
                  </button>
                )
              )}
            </div>
          </div>

          {/* Panel Preset */}
          <div>
            <label className="block text-sm font-medium text-white/80 mb-2">
              Panel Preset
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              {(Object.entries(presetConfig) as [PanelPreset, typeof presetConfig[PanelPreset]][]).map(
                ([preset, config]) => (
                  <button
                    key={preset}
                    onClick={() => setSelectedPreset(preset)}
                    className={`
                      p-3 rounded-xl border text-left transition-all
                      ${selectedPreset === preset
                        ? "border-blue-500 bg-blue-500/20"
                        : "border-white/10 hover:border-white/30 hover:bg-white/5"
                      }
                    `}
                  >
                    <div className="font-medium text-sm text-white">{config.label}</div>
                    <div className="text-xs text-white/50 mt-0.5">{config.description}</div>
                  </button>
                )
              )}
            </div>
          </div>

          <Button
            onClick={startDebate}
            disabled={!topic.trim() || isLoading}
            className="w-full"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin mr-2" />
                Starting Debate...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Start Debate
              </>
            )}
          </Button>
        </div>
      </div>
    );
  }

  // ==========================================================================
  // RENDER: Active Debate
  // ==========================================================================
  return (
    <div className={`rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl overflow-hidden flex flex-col ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-white/10 bg-white/5">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="font-semibold text-white">{session.topic}</h2>
            <div className="flex items-center gap-3 text-xs text-white/50 mt-1">
              <span>Format: {formatConfig[session.format].label}</span>
              <span>•</span>
              <span>Round {currentRound} / {maxRounds}</span>
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={resetDebate}>
            <X className="w-4 h-4" />
          </Button>
        </div>

        {/* Participants */}
        <div className="flex gap-2 mt-3 overflow-x-auto pb-1">
          {session.participants.map((p, i) => {
            const config = roleConfig[p.role];
            const Icon = config.icon;
            return (
              <div
                key={i}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${config.bgColor} flex-shrink-0`}
              >
                <Icon className={`w-3 h-3 ${config.color}`} />
                <span className={`text-xs font-medium ${config.color}`}>{p.name}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Arguments */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-[300px] max-h-[500px]">
        <AnimatePresence>
          {arguments_.map((arg, i) => {
            const config = roleConfig[arg.role];
            const Icon = config.icon;

            return (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex gap-3"
              >
                <div className={`w-10 h-10 rounded-xl ${config.bgColor} flex items-center justify-center flex-shrink-0`}>
                  <Icon className={`w-5 h-5 ${config.color}`} />
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`font-semibold text-sm ${config.color}`}>{arg.speaker}</span>
                    <span className="text-xs px-2 py-0.5 rounded-full bg-white/10 text-white/50">
                      {config.label}
                    </span>
                    {arg.argument_type !== "statement" && (
                      <span className="text-xs text-white/40">{arg.argument_type}</span>
                    )}
                  </div>
                  <div className="p-3 rounded-xl bg-white/5 border border-white/10">
                    <p className="text-sm text-white/80 whitespace-pre-wrap">{arg.content}</p>
                    {arg.key_points.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {arg.key_points.map((point, j) => (
                          <span
                            key={j}
                            className="text-xs px-2 py-0.5 rounded bg-white/10 text-white/50"
                          >
                            {point}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>

        {isAdvancing && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center justify-center py-4"
          >
            <Loader2 className="w-6 h-6 animate-spin text-purple-400" />
            <span className="ml-2 text-white/60">Agents are debating...</span>
          </motion.div>
        )}

        <div ref={argumentsEndRef} />
      </div>

      {/* Controls */}
      <div className="p-4 border-t border-white/10 space-y-3">
        {error && (
          <div className="p-2 rounded-lg bg-red-500/20 text-red-300 text-xs flex items-center gap-2">
            <AlertCircle className="w-3 h-3" />
            {error}
          </div>
        )}

        {!isCompleted && (
          <>
            {/* Learner contribution */}
            {learnerId && (
              <div className="flex gap-2">
                <input
                  type="text"
                  value={learnerInput}
                  onChange={(e) => setLearnerInput(e.target.value)}
                  placeholder="Add your perspective (optional)..."
                  className="flex-1 px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/30 text-sm focus:outline-none focus:border-white/30"
                />
              </div>
            )}

            <div className="flex gap-2">
              <Button
                onClick={() => advanceDebate(learnerInput || undefined)}
                disabled={isAdvancing}
                className="flex-1"
              >
                {isAdvancing ? (
                  <Loader2 className="w-4 h-4 animate-spin mr-2" />
                ) : (
                  <SkipForward className="w-4 h-4 mr-2" />
                )}
                {learnerInput ? "Submit & Continue" : "Next Round"}
              </Button>
            </div>
          </>
        )}

        {isCompleted && (
          <Button onClick={() => socialApi.getDebateSummary(session.session_id).then(setSummary)} className="w-full">
            <FileText className="w-4 h-4 mr-2" />
            View Summary
          </Button>
        )}
      </div>
    </div>
  );
}

export default DebateViewer;
