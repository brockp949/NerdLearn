"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  GraduationCap,
  Users,
  Code2,
  Sparkles,
  ArrowRight,
  Brain,
  MessageSquare,
  Trophy,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { TeachingSession } from "@/components/social/TeachingSession";
import { DebateViewer } from "@/components/social/DebateViewer";
import CodeChallenge from "@/components/social/CodeChallenge";

type SocialMode = "home" | "teaching" | "debate" | "code";

// Demo concepts for teaching
const demoConcepts = [
  { id: "binary_search", name: "Binary Search", description: "Efficient search algorithm for sorted arrays" },
  { id: "recursion", name: "Recursion", description: "Functions that call themselves" },
  { id: "big_o", name: "Big O Notation", description: "Algorithm complexity analysis" },
  { id: "linked_lists", name: "Linked Lists", description: "Dynamic data structure with nodes" },
  { id: "hash_tables", name: "Hash Tables", description: "Key-value storage with O(1) access" },
];

export default function SocialPage() {
  const [mode, setMode] = useState<SocialMode>("home");
  const [selectedConcept, setSelectedConcept] = useState<typeof demoConcepts[0] | null>(null);

  // Mock user ID - in production this would come from auth
  const userId = "demo_user_123";

  const renderHome = () => (
    <div className="max-w-6xl mx-auto px-6 py-12">
      {/* Hero Section */}
      <div className="text-center mb-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm mb-6"
        >
          <Sparkles className="w-4 h-4" />
          Phase 4: Innovation Mechanics
        </motion.div>
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-4xl md:text-5xl font-bold text-white mb-4"
        >
          Social Learning Lab
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="text-lg text-white/60 max-w-2xl mx-auto"
        >
          Deepen your understanding through teaching, debate, and hands-on coding challenges
          powered by AI agents.
        </motion.p>
      </div>

      {/* Feature Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Teachable Agent */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="group relative rounded-2xl border border-white/10 bg-gradient-to-br from-purple-500/10 to-pink-500/10 p-6 hover:border-purple-500/30 transition-all cursor-pointer"
          onClick={() => setMode("teaching")}
        >
          <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 to-pink-500/5 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="relative">
            <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center mb-4">
              <GraduationCap className="w-7 h-7 text-white" />
            </div>
            <h3 className="text-xl font-bold text-white mb-2">Learn by Teaching</h3>
            <p className="text-white/60 text-sm mb-4">
              Teach concepts to an AI student using the Feynman Technique.
              The best way to learn is to explain it to someone else.
            </p>
            <div className="flex items-center gap-2 text-purple-400 text-sm font-medium">
              Start Teaching <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>
        </motion.div>

        {/* SimClass Debates */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="group relative rounded-2xl border border-white/10 bg-gradient-to-br from-indigo-500/10 to-blue-500/10 p-6 hover:border-indigo-500/30 transition-all cursor-pointer"
          onClick={() => setMode("debate")}
        >
          <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-blue-500/5 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="relative">
            <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-indigo-500 to-blue-500 flex items-center justify-center mb-4">
              <Users className="w-7 h-7 text-white" />
            </div>
            <h3 className="text-xl font-bold text-white mb-2">AI Debates</h3>
            <p className="text-white/60 text-sm mb-4">
              Watch AI agents debate topics from multiple perspectives.
              Explore ideas through structured discourse.
            </p>
            <div className="flex items-center gap-2 text-indigo-400 text-sm font-medium">
              Start Debate <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>
        </motion.div>

        {/* Code Challenges */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="group relative rounded-2xl border border-white/10 bg-gradient-to-br from-emerald-500/10 to-cyan-500/10 p-6 hover:border-emerald-500/30 transition-all cursor-pointer"
          onClick={() => setMode("code")}
        >
          <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-cyan-500/5 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity" />
          <div className="relative">
            <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-emerald-500 to-cyan-500 flex items-center justify-center mb-4">
              <Code2 className="w-7 h-7 text-white" />
            </div>
            <h3 className="text-xl font-bold text-white mb-2">Code Challenges</h3>
            <p className="text-white/60 text-sm mb-4">
              Solve coding problems with AI-powered evaluation.
              Get feedback on correctness, quality, and efficiency.
            </p>
            <div className="flex items-center gap-2 text-emerald-400 text-sm font-medium">
              Start Coding <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>
        </motion.div>
      </div>

      {/* Stats Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="mt-12 grid grid-cols-2 md:grid-cols-4 gap-4"
      >
        <div className="text-center p-6 rounded-xl bg-white/5 border border-white/10">
          <Brain className="w-6 h-6 text-purple-400 mx-auto mb-2" />
          <div className="text-2xl font-bold text-white">5</div>
          <div className="text-xs text-white/50">Student Personas</div>
        </div>
        <div className="text-center p-6 rounded-xl bg-white/5 border border-white/10">
          <MessageSquare className="w-6 h-6 text-indigo-400 mx-auto mb-2" />
          <div className="text-2xl font-bold text-white">8</div>
          <div className="text-xs text-white/50">Debate Roles</div>
        </div>
        <div className="text-center p-6 rounded-xl bg-white/5 border border-white/10">
          <Code2 className="w-6 h-6 text-emerald-400 mx-auto mb-2" />
          <div className="text-2xl font-bold text-white">6</div>
          <div className="text-xs text-white/50">Eval Dimensions</div>
        </div>
        <div className="text-center p-6 rounded-xl bg-white/5 border border-white/10">
          <Trophy className="w-6 h-6 text-amber-400 mx-auto mb-2" />
          <div className="text-2xl font-bold text-white">5</div>
          <div className="text-xs text-white/50">Hint Levels</div>
        </div>
      </motion.div>
    </div>
  );

  const renderTeaching = () => (
    <div className="max-w-4xl mx-auto px-6 py-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white">Learn by Teaching</h2>
          <p className="text-white/60 text-sm">Select a concept to teach</p>
        </div>
        <Button variant="ghost" onClick={() => { setMode("home"); setSelectedConcept(null); }}>
          Back to Lab
        </Button>
      </div>

      {!selectedConcept ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {demoConcepts.map((concept) => (
            <button
              key={concept.id}
              onClick={() => setSelectedConcept(concept)}
              className="p-4 rounded-xl border border-white/10 hover:border-purple-500/30 hover:bg-white/5 transition-all text-left"
            >
              <h3 className="font-semibold text-white mb-1">{concept.name}</h3>
              <p className="text-sm text-white/50">{concept.description}</p>
            </button>
          ))}
        </div>
      ) : (
        <TeachingSession
          userId={userId}
          conceptId={selectedConcept.id}
          conceptName={selectedConcept.name}
          conceptDescription={selectedConcept.description}
          onSessionEnd={(summary) => {
            console.log("Teaching session ended:", summary);
          }}
        />
      )}
    </div>
  );

  const renderDebate = () => (
    <div className="max-w-4xl mx-auto px-6 py-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white">AI Debates</h2>
          <p className="text-white/60 text-sm">Explore topics through multi-agent discussion</p>
        </div>
        <Button variant="ghost" onClick={() => setMode("home")}>
          Back to Lab
        </Button>
      </div>

      <DebateViewer
        learnerId={userId}
        onComplete={(summary) => {
          console.log("Debate completed:", summary);
        }}
      />
    </div>
  );

  const renderCode = () => (
    <div className="max-w-6xl mx-auto px-6 py-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white">Code Challenges</h2>
          <p className="text-white/60 text-sm">Solve problems with AI evaluation</p>
        </div>
        <Button variant="ghost" onClick={() => setMode("home")}>
          Back to Lab
        </Button>
      </div>

      <CodeChallenge
        userId={userId}
        onComplete={(result) => {
          console.log("Challenge completed:", result);
        }}
      />
    </div>
  );

  return (
    <div className="min-h-[calc(100vh-64px)] bg-gradient-to-b from-background to-background/50">
      <AnimatePresence mode="wait">
        {mode === "home" && (
          <motion.div
            key="home"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {renderHome()}
          </motion.div>
        )}
        {mode === "teaching" && (
          <motion.div
            key="teaching"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
          >
            {renderTeaching()}
          </motion.div>
        )}
        {mode === "debate" && (
          <motion.div
            key="debate"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
          >
            {renderDebate()}
          </motion.div>
        )}
        {mode === "code" && (
          <motion.div
            key="code"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
          >
            {renderCode()}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
