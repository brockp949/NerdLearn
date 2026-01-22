"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  GraduationCap,
  Send,
  Loader2,
  User,
  Bot,
  Brain,
  Target,
  AlertCircle,
  CheckCircle,
  Lightbulb,
  BarChart3,
  X,
  Play,
  Square,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { socialApi } from "@/lib/api";
import type {
  StudentPersona,
  ComprehensionLevel,
  TeachingResponse,
  TeachingSessionResponse,
  TeachingSessionSummary,
} from "@/types/social";

// =============================================================================
// CONFIGURATION
// =============================================================================

const personaConfig: Record<StudentPersona, {
  label: string;
  description: string;
  icon: string;
  color: string;
}> = {
  curious: {
    label: "Curious",
    description: "Asks lots of 'why' and 'how' questions",
    icon: "🤔",
    color: "from-blue-500 to-cyan-500",
  },
  confused: {
    label: "Confused",
    description: "Needs things explained multiple ways",
    icon: "😕",
    color: "from-purple-500 to-pink-500",
  },
  challenger: {
    label: "Challenger",
    description: "Questions assumptions and edge cases",
    icon: "🧐",
    color: "from-orange-500 to-red-500",
  },
  visual: {
    label: "Visual",
    description: "Asks for examples and analogies",
    icon: "👁️",
    color: "from-green-500 to-emerald-500",
  },
  practical: {
    label: "Practical",
    description: "Wants real-world applications",
    icon: "🔧",
    color: "from-amber-500 to-yellow-500",
  },
};

const comprehensionConfig: Record<ComprehensionLevel, {
  label: string;
  color: string;
  bgColor: string;
  percentage: number;
}> = {
  lost: { label: "Lost", color: "text-red-400", bgColor: "bg-red-500", percentage: 10 },
  struggling: { label: "Struggling", color: "text-orange-400", bgColor: "bg-orange-500", percentage: 30 },
  emerging: { label: "Emerging", color: "text-yellow-400", bgColor: "bg-yellow-500", percentage: 50 },
  developing: { label: "Developing", color: "text-blue-400", bgColor: "bg-blue-500", percentage: 70 },
  mastering: { label: "Mastering", color: "text-green-400", bgColor: "bg-green-500", percentage: 90 },
};

// =============================================================================
// TYPES
// =============================================================================

interface Message {
  id: string;
  role: "user" | "student";
  content: string;
  timestamp: Date;
  questionType?: string;
}

interface TeachingSessionProps {
  userId: string;
  conceptId: string;
  conceptName: string;
  conceptDescription?: string;
  onSessionEnd?: (summary: TeachingSessionSummary) => void;
  className?: string;
}

// =============================================================================
// COMPONENT
// =============================================================================

export function TeachingSession({
  userId,
  conceptId,
  conceptName,
  conceptDescription,
  onSessionEnd,
  className = "",
}: TeachingSessionProps) {
  // State
  const [selectedPersona, setSelectedPersona] = useState<StudentPersona | null>(null);
  const [session, setSession] = useState<TeachingSessionResponse | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [comprehension, setComprehension] = useState(0);
  const [comprehensionLevel, setComprehensionLevel] = useState<ComprehensionLevel>("lost");
  const [knowledgeGaps, setKnowledgeGaps] = useState<string[]>([]);
  const [conceptsUnderstood, setConceptsUnderstood] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isEnding, setIsEnding] = useState(false);
  const [summary, setSummary] = useState<TeachingSessionSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Start session
  const startSession = useCallback(async (persona: StudentPersona) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await socialApi.startTeachingSession({
        user_id: userId,
        concept_id: conceptId,
        concept_name: conceptName,
        persona,
        concept_description: conceptDescription,
      });

      setSession(response);
      setSelectedPersona(persona);
      setComprehension(response.comprehension);
      setComprehensionLevel(response.comprehension_level);

      // Add opening message
      setMessages([
        {
          id: "opening",
          role: "student",
          content: response.opening_question,
          timestamp: new Date(),
        },
      ]);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to start session");
    } finally {
      setIsLoading(false);
    }
  }, [userId, conceptId, conceptName, conceptDescription]);

  // Submit explanation
  const submitExplanation = useCallback(async () => {
    if (!inputValue.trim() || !session || isLoading) return;

    const explanation = inputValue.trim();
    setInputValue("");
    setIsLoading(true);
    setError(null);

    // Add user message
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: explanation,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);

    try {
      const response: TeachingResponse = await socialApi.submitExplanation({
        session_id: session.session_id,
        explanation,
        concept_description: conceptDescription,
      });

      // Add student response
      const studentMessage: Message = {
        id: `student-${Date.now()}`,
        role: "student",
        content: response.response,
        timestamp: new Date(),
        questionType: response.question_type || undefined,
      };
      setMessages((prev) => [...prev, studentMessage]);

      // Update state
      setComprehension(response.comprehension);
      setComprehensionLevel(response.comprehension_level);

      if (response.knowledge_gaps.length > 0) {
        setKnowledgeGaps((prev) => [...new Set([...prev, ...response.knowledge_gaps])]);
      }
      if (response.concepts_understood.length > 0) {
        setConceptsUnderstood((prev) => [...new Set([...prev, ...response.concepts_understood])]);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to process explanation");
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }, [inputValue, session, isLoading, conceptDescription]);

  // End session
  const endSession = useCallback(async () => {
    if (!session || isEnding) return;

    setIsEnding(true);
    setError(null);

    try {
      const summaryResponse = await socialApi.endTeachingSession(session.session_id);
      setSummary(summaryResponse);
      onSessionEnd?.(summaryResponse);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to end session");
    } finally {
      setIsEnding(false);
    }
  }, [session, isEnding, onSessionEnd]);

  // Handle key press
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitExplanation();
    }
  };

  // Reset session
  const resetSession = () => {
    setSelectedPersona(null);
    setSession(null);
    setMessages([]);
    setComprehension(0);
    setComprehensionLevel("lost");
    setKnowledgeGaps([]);
    setConceptsUnderstood([]);
    setSummary(null);
    setError(null);
  };

  // ==========================================================================
  // RENDER: Summary View
  // ==========================================================================
  if (summary) {
    return (
      <div className={`rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl overflow-hidden ${className}`}>
        <div className="p-6 border-b border-white/10 bg-gradient-to-r from-green-500/20 to-emerald-500/20">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-green-500/20 flex items-center justify-center">
                <CheckCircle className="w-6 h-6 text-green-400" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">Session Complete!</h2>
                <p className="text-white/60 text-sm">You taught: {summary.concept}</p>
              </div>
            </div>
            <Button variant="ghost" size="icon" onClick={resetSession}>
              <X className="w-5 h-5" />
            </Button>
          </div>
        </div>

        <div className="p-6 space-y-6">
          {/* Score Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl bg-white/5 border border-white/10 text-center">
              <div className="text-3xl font-bold text-green-400">
                {Math.round(summary.teaching_effectiveness * 100)}%
              </div>
              <div className="text-xs text-white/50 mt-1">Teaching Score</div>
            </div>
            <div className="p-4 rounded-xl bg-white/5 border border-white/10 text-center">
              <div className="text-3xl font-bold text-blue-400">
                {Math.round(summary.final_comprehension * 100)}%
              </div>
              <div className="text-xs text-white/50 mt-1">Student Understood</div>
            </div>
            <div className="p-4 rounded-xl bg-white/5 border border-white/10 text-center">
              <div className="text-3xl font-bold text-purple-400">
                {summary.total_exchanges}
              </div>
              <div className="text-xs text-white/50 mt-1">Exchanges</div>
            </div>
            <div className="p-4 rounded-xl bg-white/5 border border-white/10 text-center">
              <div className="text-3xl font-bold text-amber-400">
                {summary.duration_minutes.toFixed(1)}
              </div>
              <div className="text-xs text-white/50 mt-1">Minutes</div>
            </div>
          </div>

          {/* Recommendations */}
          {summary.recommendations.length > 0 && (
            <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/20">
              <div className="flex items-center gap-2 mb-3">
                <Lightbulb className="w-5 h-5 text-amber-400" />
                <h3 className="font-semibold text-amber-400">Recommendations</h3>
              </div>
              <ul className="space-y-2">
                {summary.recommendations.map((rec, i) => (
                  <li key={i} className="text-sm text-white/70 flex items-start gap-2">
                    <span className="text-amber-400 mt-1">•</span>
                    {rec}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Knowledge Gaps */}
          {summary.knowledge_gaps_identified.length > 0 && (
            <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20">
              <div className="flex items-center gap-2 mb-3">
                <AlertCircle className="w-5 h-5 text-red-400" />
                <h3 className="font-semibold text-red-400">Areas to Review</h3>
              </div>
              <div className="flex flex-wrap gap-2">
                {summary.knowledge_gaps_identified.map((gap, i) => (
                  <span
                    key={i}
                    className="px-2 py-1 rounded-lg bg-red-500/20 text-red-300 text-xs"
                  >
                    {gap}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Strong Points */}
          {summary.strong_explanations.length > 0 && (
            <div className="p-4 rounded-xl bg-green-500/10 border border-green-500/20">
              <div className="flex items-center gap-2 mb-3">
                <CheckCircle className="w-5 h-5 text-green-400" />
                <h3 className="font-semibold text-green-400">Well Explained</h3>
              </div>
              <div className="flex flex-wrap gap-2">
                {summary.strong_explanations.map((point, i) => (
                  <span
                    key={i}
                    className="px-2 py-1 rounded-lg bg-green-500/20 text-green-300 text-xs"
                  >
                    {point}
                  </span>
                ))}
              </div>
            </div>
          )}

          <Button onClick={resetSession} className="w-full" variant="outline">
            Start New Session
          </Button>
        </div>
      </div>
    );
  }

  // ==========================================================================
  // RENDER: Persona Selection
  // ==========================================================================
  if (!selectedPersona || !session) {
    return (
      <div className={`rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl overflow-hidden ${className}`}>
        <div className="p-6 border-b border-white/10">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
              <GraduationCap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Teach: {conceptName}</h2>
              <p className="text-white/60 text-sm">Choose a student persona to teach</p>
            </div>
          </div>
        </div>

        <div className="p-6">
          <p className="text-white/70 mb-4">
            The best way to learn is to teach. Explain this concept to a simulated student
            and solidify your understanding through the Feynman Technique.
          </p>

          {error && (
            <div className="mb-4 p-3 rounded-lg bg-red-500/20 border border-red-500/30 text-red-300 text-sm flex items-center gap-2">
              <AlertCircle className="w-4 h-4" />
              {error}
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {(Object.entries(personaConfig) as [StudentPersona, typeof personaConfig[StudentPersona]][]).map(
              ([persona, config]) => (
                <button
                  key={persona}
                  onClick={() => startSession(persona)}
                  disabled={isLoading}
                  className={`
                    p-4 rounded-xl border border-white/10 text-left
                    hover:border-white/30 hover:bg-white/5 transition-all
                    ${isLoading ? "opacity-50 cursor-not-allowed" : ""}
                  `}
                >
                  <div className="flex items-center gap-3 mb-2">
                    <span className="text-2xl">{config.icon}</span>
                    <span className="font-semibold text-white">{config.label}</span>
                  </div>
                  <p className="text-xs text-white/50">{config.description}</p>
                </button>
              )
            )}
          </div>

          {isLoading && (
            <div className="mt-4 flex items-center justify-center gap-2 text-white/60">
              <Loader2 className="w-4 h-4 animate-spin" />
              Starting session...
            </div>
          )}
        </div>
      </div>
    );
  }

  // ==========================================================================
  // RENDER: Active Session
  // ==========================================================================
  const currentPersonaConfig = personaConfig[selectedPersona];
  const currentCompConfig = comprehensionConfig[comprehensionLevel];

  return (
    <div className={`rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl overflow-hidden flex flex-col ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-white/10 bg-white/5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${currentPersonaConfig.color} flex items-center justify-center`}>
              <span className="text-lg">{currentPersonaConfig.icon}</span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-white">Alex</span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-white/10 text-white/60">
                  {currentPersonaConfig.label} Student
                </span>
              </div>
              <p className="text-xs text-white/50">Learning: {conceptName}</p>
            </div>
          </div>

          <Button
            variant="ghost"
            size="sm"
            onClick={endSession}
            disabled={isEnding}
            className="text-white/60 hover:text-white"
          >
            {isEnding ? (
              <Loader2 className="w-4 h-4 animate-spin mr-2" />
            ) : (
              <Square className="w-4 h-4 mr-2" />
            )}
            End Session
          </Button>
        </div>

        {/* Comprehension Bar */}
        <div className="mt-4">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-white/60">Student Comprehension</span>
            <span className={currentCompConfig.color}>{currentCompConfig.label}</span>
          </div>
          <div className="h-2 bg-white/10 rounded-full overflow-hidden">
            <motion.div
              className={`h-full ${currentCompConfig.bgColor}`}
              initial={{ width: 0 }}
              animate={{ width: `${comprehension * 100}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-[300px] max-h-[500px]">
        <AnimatePresence>
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex gap-3 ${message.role === "user" ? "flex-row-reverse" : ""}`}
            >
              <div
                className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${
                  message.role === "user"
                    ? "bg-blue-500/20 text-blue-400"
                    : `bg-gradient-to-br ${currentPersonaConfig.color} text-white`
                }`}
              >
                {message.role === "user" ? (
                  <User className="w-4 h-4" />
                ) : (
                  <span className="text-sm">{currentPersonaConfig.icon}</span>
                )}
              </div>
              <div
                className={`max-w-[80%] p-3 rounded-xl ${
                  message.role === "user"
                    ? "bg-blue-500/20 border border-blue-500/30 text-white"
                    : "bg-white/5 border border-white/10 text-white/90"
                }`}
              >
                <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                {message.questionType && (
                  <span className="inline-block mt-2 text-xs px-2 py-0.5 rounded bg-white/10 text-white/50">
                    {message.questionType}
                  </span>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex gap-3"
          >
            <div className={`w-8 h-8 rounded-lg flex items-center justify-center bg-gradient-to-br ${currentPersonaConfig.color}`}>
              <span className="text-sm">{currentPersonaConfig.icon}</span>
            </div>
            <div className="bg-white/5 border border-white/10 p-3 rounded-xl">
              <div className="flex items-center gap-2 text-white/50 text-sm">
                <Loader2 className="w-4 h-4 animate-spin" />
                Alex is thinking...
              </div>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Sidebar Info */}
      {(knowledgeGaps.length > 0 || conceptsUnderstood.length > 0) && (
        <div className="px-4 py-2 border-t border-white/10 bg-white/5 flex gap-4 overflow-x-auto">
          {knowledgeGaps.length > 0 && (
            <div className="flex items-center gap-2 text-xs">
              <AlertCircle className="w-3 h-3 text-amber-400 flex-shrink-0" />
              <span className="text-white/50">Gaps:</span>
              {knowledgeGaps.slice(0, 3).map((gap, i) => (
                <span key={i} className="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300">
                  {gap}
                </span>
              ))}
            </div>
          )}
          {conceptsUnderstood.length > 0 && (
            <div className="flex items-center gap-2 text-xs">
              <CheckCircle className="w-3 h-3 text-green-400 flex-shrink-0" />
              <span className="text-white/50">Got:</span>
              {conceptsUnderstood.slice(0, 3).map((concept, i) => (
                <span key={i} className="px-1.5 py-0.5 rounded bg-green-500/20 text-green-300">
                  {concept}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Input */}
      <div className="p-4 border-t border-white/10">
        {error && (
          <div className="mb-3 p-2 rounded-lg bg-red-500/20 text-red-300 text-xs flex items-center gap-2">
            <AlertCircle className="w-3 h-3" />
            {error}
          </div>
        )}
        <div className="flex gap-2">
          <textarea
            ref={inputRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Explain the concept to Alex..."
            disabled={isLoading}
            rows={2}
            className="flex-1 px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white placeholder-white/30 text-sm resize-none focus:outline-none focus:border-white/30"
          />
          <Button
            onClick={submitExplanation}
            disabled={!inputValue.trim() || isLoading}
            className="self-end"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </Button>
        </div>
        <p className="text-xs text-white/30 mt-2">
          Press Enter to send, Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}

export default TeachingSession;
