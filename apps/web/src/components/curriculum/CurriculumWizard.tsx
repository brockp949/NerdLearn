"use client";

/**
 * Curriculum Generation Wizard
 *
 * Multi-step wizard for generating AI-powered curricula:
 * 1. Topic Selection - What to learn
 * 2. Configuration - Duration, difficulty, style
 * 3. Generation - AI agents at work
 * 4. Review - Preview and confirm syllabus
 */

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Sparkles,
  ArrowRight,
  ArrowLeft,
  Clock,
  Target,
  BookOpen,
  Loader2,
  CheckCircle2,
  XCircle,
  ChevronDown,
  ChevronUp,
  GraduationCap,
  Lightbulb,
  Layers,
  Play,
  RotateCcw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useCurriculumGeneration } from "@/hooks/use-curriculum";
import type { CurriculumDifficulty as DifficultyLevel, LearningStyle, ModuleData, Syllabus } from "@/types/curriculum";

interface CurriculumWizardProps {
  courseId?: number;
  onComplete?: (syllabus: Syllabus) => void;
}

type WizardStep = "topic" | "configure" | "generating" | "review";

const difficultyOptions: { value: DifficultyLevel; label: string; description: string }[] = [
  { value: "beginner", label: "Beginner", description: "No prior knowledge required" },
  { value: "intermediate", label: "Intermediate", description: "Some background helpful" },
  { value: "advanced", label: "Advanced", description: "Strong foundation needed" },
];

const learningStyleOptions: { value: LearningStyle; label: string; description: string; icon: any }[] = [
  { value: "visual", label: "Visual", description: "Diagrams, charts, videos", icon: "🎨" },
  { value: "text", label: "Reading", description: "Articles, documentation", icon: "📚" },
  { value: "interactive", label: "Interactive", description: "Exercises, coding", icon: "⚡" },
  { value: "balanced", label: "Balanced", description: "Mix of all styles", icon: "🎯" },
];

export function CurriculumWizard({ courseId = 1, onComplete }: CurriculumWizardProps) {
  const [step, setStep] = useState<WizardStep>("topic");
  const [topic, setTopic] = useState("");
  const [durationWeeks, setDurationWeeks] = useState(4);
  const [difficulty, setDifficulty] = useState<DifficultyLevel>("intermediate");
  const [learningStyle, setLearningStyle] = useState<LearningStyle>("balanced");
  const [targetAudience, setTargetAudience] = useState("");
  const [prerequisites, setPrerequisites] = useState<string[]>([]);
  const [newPrereq, setNewPrereq] = useState("");
  const [expandedModule, setExpandedModule] = useState<number | null>(null);

  const {
    isGenerating,
    result,
    error,
    progress,
    currentAgent,
    generateSync,
    reset,
  } = useCurriculumGeneration({
    onComplete: (res) => {
      if (res.success && res.syllabus) {
        setStep("review");
      }
    },
  });

  const handleStartGeneration = useCallback(async () => {
    setStep("generating");
    await generateSync({
      topic,
      courseId,
      durationWeeks,
      difficultyLevel: difficulty,
      targetAudience: targetAudience || undefined,
      prerequisites: prerequisites.length > 0 ? prerequisites : undefined,
      learningStyle,
    });
  }, [topic, courseId, durationWeeks, difficulty, targetAudience, prerequisites, learningStyle, generateSync]);

  const handleAddPrereq = () => {
    if (newPrereq.trim() && !prerequisites.includes(newPrereq.trim())) {
      setPrerequisites([...prerequisites, newPrereq.trim()]);
      setNewPrereq("");
    }
  };

  const handleRemovePrereq = (prereq: string) => {
    setPrerequisites(prerequisites.filter((p) => p !== prereq));
  };

  const handleFinish = () => {
    if (result?.syllabus) {
      onComplete?.(result.syllabus);
    }
  };

  const handleRestart = () => {
    reset();
    setStep("topic");
    setTopic("");
    setExpandedModule(null);
  };

  const renderTopicStep = () => (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className="space-y-6"
    >
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-purple-500 to-pink-500 mb-4">
          <Sparkles className="w-8 h-8 text-white" />
        </div>
        <h2 className="text-2xl font-bold text-white mb-2">What do you want to learn?</h2>
        <p className="text-white/60">Our AI will craft a personalized curriculum just for you</p>
      </div>

      <div className="relative">
        <input
          type="text"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="e.g., Machine Learning Fundamentals, React Development, Quantum Computing..."
          className="w-full px-4 py-4 bg-white/5 border border-white/10 rounded-xl text-white placeholder-white/40 focus:outline-none focus:border-purple-500/50 focus:ring-2 focus:ring-purple-500/20 text-lg"
        />
        <Lightbulb className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/30" />
      </div>

      <div className="flex flex-wrap gap-2">
        {["Machine Learning", "Web Development", "Data Science", "Blockchain", "Cybersecurity"].map((suggestion) => (
          <button
            key={suggestion}
            onClick={() => setTopic(suggestion)}
            className="px-3 py-1.5 rounded-full bg-white/5 hover:bg-white/10 text-white/60 hover:text-white text-sm transition-colors"
          >
            {suggestion}
          </button>
        ))}
      </div>

      <div className="flex justify-end pt-4">
        <Button
          onClick={() => setStep("configure")}
          disabled={!topic.trim()}
          className="gap-2"
        >
          Continue
          <ArrowRight className="w-4 h-4" />
        </Button>
      </div>
    </motion.div>
  );

  const renderConfigureStep = () => (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className="space-y-6"
    >
      <div className="mb-6">
        <h2 className="text-xl font-bold text-white mb-1">Customize Your Curriculum</h2>
        <p className="text-white/60 text-sm">Topic: {topic}</p>
      </div>

      {/* Duration */}
      <div className="p-4 rounded-xl border border-white/10 bg-white/5">
        <div className="flex items-center gap-2 mb-3">
          <Clock className="w-4 h-4 text-purple-400" />
          <span className="font-medium text-white">Duration</span>
        </div>
        <div className="flex items-center gap-4">
          <input
            type="range"
            min="1"
            max="12"
            value={durationWeeks}
            onChange={(e) => setDurationWeeks(parseInt(e.target.value))}
            className="flex-1 accent-purple-500"
          />
          <span className="text-white font-mono w-20 text-right">
            {durationWeeks} {durationWeeks === 1 ? "week" : "weeks"}
          </span>
        </div>
      </div>

      {/* Difficulty */}
      <div className="p-4 rounded-xl border border-white/10 bg-white/5">
        <div className="flex items-center gap-2 mb-3">
          <Target className="w-4 h-4 text-purple-400" />
          <span className="font-medium text-white">Difficulty Level</span>
        </div>
        <div className="grid grid-cols-3 gap-2">
          {difficultyOptions.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setDifficulty(opt.value)}
              className={`p-3 rounded-lg border transition-all text-left ${difficulty === opt.value
                  ? "border-purple-500 bg-purple-500/20"
                  : "border-white/10 hover:border-white/20"
                }`}
            >
              <div className="font-medium text-white text-sm">{opt.label}</div>
              <div className="text-xs text-white/50">{opt.description}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Learning Style */}
      <div className="p-4 rounded-xl border border-white/10 bg-white/5">
        <div className="flex items-center gap-2 mb-3">
          <BookOpen className="w-4 h-4 text-purple-400" />
          <span className="font-medium text-white">Learning Style</span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {learningStyleOptions.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setLearningStyle(opt.value)}
              className={`p-3 rounded-lg border transition-all text-center ${learningStyle === opt.value
                  ? "border-purple-500 bg-purple-500/20"
                  : "border-white/10 hover:border-white/20"
                }`}
            >
              <div className="text-2xl mb-1">{opt.icon}</div>
              <div className="font-medium text-white text-sm">{opt.label}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Prerequisites */}
      <div className="p-4 rounded-xl border border-white/10 bg-white/5">
        <div className="flex items-center gap-2 mb-3">
          <Layers className="w-4 h-4 text-purple-400" />
          <span className="font-medium text-white">Prerequisites (Optional)</span>
        </div>
        <div className="flex gap-2 mb-2">
          <input
            type="text"
            value={newPrereq}
            onChange={(e) => setNewPrereq(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAddPrereq()}
            placeholder="Add a skill you already have..."
            className="flex-1 px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder-white/40 text-sm focus:outline-none focus:border-purple-500/50"
          />
          <Button variant="secondary" size="sm" onClick={handleAddPrereq}>
            Add
          </Button>
        </div>
        {prerequisites.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {prerequisites.map((prereq) => (
              <span
                key={prereq}
                className="px-2 py-1 rounded-full bg-purple-500/20 text-purple-300 text-xs flex items-center gap-1"
              >
                {prereq}
                <button
                  onClick={() => handleRemovePrereq(prereq)}
                  className="hover:text-white"
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="flex justify-between pt-4">
        <Button variant="ghost" onClick={() => setStep("topic")} className="gap-2">
          <ArrowLeft className="w-4 h-4" />
          Back
        </Button>
        <Button onClick={handleStartGeneration} className="gap-2">
          <Play className="w-4 h-4" />
          Generate Curriculum
        </Button>
      </div>
    </motion.div>
  );

  const renderGeneratingStep = () => (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="text-center py-12"
    >
      <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gradient-to-br from-purple-500 to-pink-500 mb-6">
        <Loader2 className="w-10 h-10 text-white animate-spin" />
      </div>
      <h2 className="text-2xl font-bold text-white mb-2">Generating Your Curriculum</h2>
      <p className="text-white/60 mb-8">Our AI agents are crafting the perfect learning path...</p>

      {/* Agent Progress */}
      <div className="max-w-md mx-auto space-y-3">
        {["Architect", "Refiner", "Verifier"].map((agent, idx) => {
          const isActive = currentAgent?.toLowerCase() === agent.toLowerCase();
          const isComplete =
            (agent === "Architect" && (currentAgent === "refiner" || currentAgent === "verifier" || result)) ||
            (agent === "Refiner" && (currentAgent === "verifier" || result)) ||
            (agent === "Verifier" && result);

          return (
            <div
              key={agent}
              className={`flex items-center gap-3 p-3 rounded-lg transition-all ${isActive
                  ? "bg-purple-500/20 border border-purple-500/30"
                  : isComplete
                    ? "bg-emerald-500/10 border border-emerald-500/20"
                    : "bg-white/5 border border-white/10"
                }`}
            >
              {isActive ? (
                <Loader2 className="w-5 h-5 text-purple-400 animate-spin" />
              ) : isComplete ? (
                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              ) : (
                <div className="w-5 h-5 rounded-full border-2 border-white/20" />
              )}
              <span className={`font-medium ${isActive ? "text-purple-300" : isComplete ? "text-emerald-300" : "text-white/50"}`}>
                {agent} Agent
              </span>
              <span className="ml-auto text-xs text-white/40">
                {agent === "Architect" && "Planning structure"}
                {agent === "Refiner" && "Adding detail"}
                {agent === "Verifier" && "Quality check"}
              </span>
            </div>
          );
        })}
      </div>

      {error && (
        <div className="mt-8 p-4 rounded-lg bg-red-500/20 border border-red-500/30 max-w-md mx-auto">
          <div className="flex items-center gap-2 text-red-400 mb-2">
            <XCircle className="w-5 h-5" />
            <span className="font-medium">Generation Failed</span>
          </div>
          <p className="text-sm text-white/70">{error}</p>
          <Button variant="ghost" size="sm" onClick={handleRestart} className="mt-4">
            <RotateCcw className="w-4 h-4 mr-2" />
            Try Again
          </Button>
        </div>
      )}
    </motion.div>
  );

  const renderReviewStep = () => {
    const syllabus = result?.syllabus;
    if (!syllabus) return null;

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white mb-1">Your Curriculum is Ready!</h2>
            <p className="text-white/60 text-sm">{syllabus.overall_arc}</p>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/20 text-emerald-400 text-sm">
            <CheckCircle2 className="w-4 h-4" />
            {result?.quality_score ? `${(result.quality_score * 100).toFixed(0)}% Quality` : "Complete"}
          </div>
        </div>

        {/* Modules */}
        <div className="space-y-3">
          {syllabus.modules.map((module: ModuleData, idx: number) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="rounded-xl border border-white/10 bg-white/5 overflow-hidden"
            >
              <button
                onClick={() => setExpandedModule(expandedModule === idx ? null : idx)}
                className="w-full flex items-center gap-4 p-4 text-left hover:bg-white/5 transition-colors"
              >
                <div className="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center font-bold text-purple-400">
                  {module.week}
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-white">{module.title}</h3>
                  <div className="flex items-center gap-3 text-xs text-white/50">
                    <span>{module.concepts.length} concepts</span>
                    <span>Difficulty: {module.difficulty}/10</span>
                  </div>
                </div>
                {expandedModule === idx ? (
                  <ChevronUp className="w-5 h-5 text-white/50" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-white/50" />
                )}
              </button>

              <AnimatePresence>
                {expandedModule === idx && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="border-t border-white/10"
                  >
                    <div className="p-4 space-y-3">
                      <div>
                        <div className="text-xs text-white/50 mb-2">Concepts</div>
                        <div className="flex flex-wrap gap-2">
                          {module.concepts.map((concept, i) => (
                            <span
                              key={i}
                              className="px-2 py-1 rounded-full bg-purple-500/20 text-purple-300 text-xs"
                            >
                              {concept}
                            </span>
                          ))}
                        </div>
                      </div>
                      {module.prerequisites.length > 0 && (
                        <div>
                          <div className="text-xs text-white/50 mb-2">Prerequisites</div>
                          <div className="flex flex-wrap gap-2">
                            {module.prerequisites.map((prereq, i) => (
                              <span
                                key={i}
                                className="px-2 py-1 rounded-full bg-white/10 text-white/60 text-xs"
                              >
                                {prereq}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      <div className="p-3 rounded-lg bg-white/5">
                        <div className="text-xs text-white/50 mb-1">Rationale</div>
                        <p className="text-sm text-white/70">{module.rationale}</p>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          ))}
        </div>

        <div className="flex justify-between pt-4">
          <Button variant="ghost" onClick={handleRestart} className="gap-2">
            <RotateCcw className="w-4 h-4" />
            Start Over
          </Button>
          <Button onClick={handleFinish} className="gap-2">
            <GraduationCap className="w-4 h-4" />
            Start Learning
          </Button>
        </div>
      </motion.div>
    );
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Progress Indicator */}
      <div className="flex items-center gap-2 mb-8">
        {[
          { id: "topic", label: "Topic" },
          { id: "configure", label: "Configure" },
          { id: "generating", label: "Generate" },
          { id: "review", label: "Review" },
        ].map((s, idx) => {
          const steps: WizardStep[] = ["topic", "configure", "generating", "review"];
          const currentIdx = steps.indexOf(step);
          const isComplete = idx < currentIdx;
          const isCurrent = step === s.id;

          return (
            <div key={s.id} className="flex items-center flex-1">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-colors ${isComplete
                    ? "bg-purple-500 text-white"
                    : isCurrent
                      ? "bg-purple-500/30 text-purple-300 ring-2 ring-purple-500"
                      : "bg-white/10 text-white/40"
                  }`}
              >
                {isComplete ? <CheckCircle2 className="w-4 h-4" /> : idx + 1}
              </div>
              {idx < 3 && (
                <div
                  className={`flex-1 h-0.5 mx-2 ${isComplete ? "bg-purple-500" : "bg-white/10"
                    }`}
                />
              )}
            </div>
          );
        })}
      </div>

      {/* Step Content */}
      <AnimatePresence mode="wait">
        {step === "topic" && renderTopicStep()}
        {step === "configure" && renderConfigureStep()}
        {step === "generating" && renderGeneratingStep()}
        {step === "review" && renderReviewStep()}
      </AnimatePresence>
    </div>
  );
}

export default CurriculumWizard;
