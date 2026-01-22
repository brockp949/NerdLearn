"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Map,
  Plus,
  Sparkles,
  ChevronRight,
  Clock,
  BookOpen,
  Target,
  Flame,
  Trophy,
  Filter,
  Search,
  LayoutGrid,
  List,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { CurriculumWizard } from "@/components/curriculum";
import type { Syllabus } from "@/types/curriculum";

type ViewMode = "grid" | "list";
type QuestStatus = "active" | "completed" | "not_started";

interface Quest {
  id: string;
  title: string;
  topic: string;
  description: string;
  progress: number;
  totalModules: number;
  completedModules: number;
  status: QuestStatus;
  difficulty: "beginner" | "intermediate" | "advanced";
  estimatedHours: number;
  streak: number;
  lastActivity?: string;
  syllabus?: Syllabus;
}

// Mock data - in production this would come from API
const mockQuests: Quest[] = [
  {
    id: "1",
    title: "Machine Learning Fundamentals",
    topic: "Machine Learning",
    description: "Master the foundations of ML, from linear regression to neural networks.",
    progress: 65,
    totalModules: 8,
    completedModules: 5,
    status: "active",
    difficulty: "intermediate",
    estimatedHours: 40,
    streak: 7,
    lastActivity: "2 hours ago",
  },
  {
    id: "2",
    title: "React Advanced Patterns",
    topic: "React",
    description: "Learn compound components, render props, and custom hooks.",
    progress: 100,
    totalModules: 6,
    completedModules: 6,
    status: "completed",
    difficulty: "advanced",
    estimatedHours: 25,
    streak: 0,
    lastActivity: "1 week ago",
  },
  {
    id: "3",
    title: "Data Structures & Algorithms",
    topic: "DSA",
    description: "Build a solid foundation in classic CS fundamentals.",
    progress: 0,
    totalModules: 12,
    completedModules: 0,
    status: "not_started",
    difficulty: "beginner",
    estimatedHours: 60,
    streak: 0,
  },
];

export default function QuestsPage() {
  const [showWizard, setShowWizard] = useState(false);
  const [quests, setQuests] = useState<Quest[]>(mockQuests);
  const [viewMode, setViewMode] = useState<ViewMode>("grid");
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<QuestStatus | "all">("all");

  const handleCurriculumComplete = (syllabus: Syllabus) => {
    // Create a new quest from the generated syllabus
    const newQuest: Quest = {
      id: `quest_${Date.now()}`,
      title: syllabus.topic || "New Learning Quest",
      topic: syllabus.topic || "Custom",
      description: syllabus.overall_arc,
      progress: 0,
      totalModules: syllabus.modules.length,
      completedModules: 0,
      status: "not_started",
      difficulty: "intermediate",
      estimatedHours: (syllabus.duration_weeks || 4) * 10,
      streak: 0,
      syllabus,
    };
    setQuests((prev) => [newQuest, ...prev]);
    setShowWizard(false);
  };

  const filteredQuests = quests.filter((quest) => {
    const matchesSearch =
      quest.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      quest.topic.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === "all" || quest.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const getStatusColor = (status: QuestStatus) => {
    switch (status) {
      case "active":
        return "text-emerald-400 bg-emerald-500/20";
      case "completed":
        return "text-purple-400 bg-purple-500/20";
      case "not_started":
        return "text-white/50 bg-white/10";
    }
  };

  const getDifficultyColor = (difficulty: Quest["difficulty"]) => {
    switch (difficulty) {
      case "beginner":
        return "text-green-400";
      case "intermediate":
        return "text-amber-400";
      case "advanced":
        return "text-red-400";
    }
  };

  const renderQuestCard = (quest: Quest) => (
    <motion.div
      key={quest.id}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      whileHover={{ scale: 1.02 }}
      className="group relative rounded-xl border border-white/10 bg-white/5 backdrop-blur-md overflow-hidden cursor-pointer transition-all hover:border-primary/30 hover:shadow-[0_0_30px_rgba(139,92,246,0.1)]"
    >
      {/* Progress Bar at Top */}
      <div className="h-1 bg-white/10">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${quest.progress}%` }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className={`h-full ${
            quest.status === "completed"
              ? "bg-gradient-to-r from-purple-500 to-pink-500"
              : "bg-gradient-to-r from-primary to-cyan-500"
          }`}
        />
      </div>

      <div className="p-5">
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <span className={`text-xs px-2 py-0.5 rounded-full ${getStatusColor(quest.status)}`}>
                {quest.status.replace("_", " ")}
              </span>
              <span className={`text-xs ${getDifficultyColor(quest.difficulty)}`}>
                {quest.difficulty}
              </span>
            </div>
            <h3 className="font-bold text-white group-hover:text-primary transition-colors">
              {quest.title}
            </h3>
          </div>
          {quest.streak > 0 && (
            <div className="flex items-center gap-1 text-amber-400">
              <Flame className="w-4 h-4" />
              <span className="text-sm font-bold">{quest.streak}</span>
            </div>
          )}
        </div>

        {/* Description */}
        <p className="text-sm text-white/60 mb-4 line-clamp-2">{quest.description}</p>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="flex items-center gap-1 text-white/50">
            <BookOpen className="w-3 h-3" />
            <span>
              {quest.completedModules}/{quest.totalModules}
            </span>
          </div>
          <div className="flex items-center gap-1 text-white/50">
            <Clock className="w-3 h-3" />
            <span>{quest.estimatedHours}h</span>
          </div>
          <div className="flex items-center gap-1 text-white/50">
            <Target className="w-3 h-3" />
            <span>{quest.progress}%</span>
          </div>
        </div>

        {/* Footer */}
        {quest.lastActivity && (
          <div className="mt-4 pt-3 border-t border-white/10 text-xs text-white/40">
            Last activity: {quest.lastActivity}
          </div>
        )}

        {/* Hover Action */}
        <div className="absolute bottom-0 left-0 right-0 h-0 group-hover:h-12 bg-gradient-to-t from-primary/20 to-transparent transition-all duration-300 flex items-center justify-center opacity-0 group-hover:opacity-100">
          <span className="text-sm font-medium text-primary flex items-center gap-1">
            Continue Quest <ChevronRight className="w-4 h-4" />
          </span>
        </div>
      </div>
    </motion.div>
  );

  const renderQuestListItem = (quest: Quest) => (
    <motion.div
      key={quest.id}
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      className="flex items-center gap-4 p-4 rounded-xl border border-white/10 bg-white/5 backdrop-blur-md hover:border-primary/30 transition-all cursor-pointer"
    >
      {/* Progress Circle */}
      <div className="relative w-12 h-12 flex-shrink-0">
        <svg className="w-12 h-12 -rotate-90">
          <circle
            cx="24"
            cy="24"
            r="20"
            className="fill-none stroke-white/10"
            strokeWidth="4"
          />
          <circle
            cx="24"
            cy="24"
            r="20"
            className={`fill-none ${
              quest.status === "completed" ? "stroke-purple-500" : "stroke-primary"
            }`}
            strokeWidth="4"
            strokeDasharray={`${quest.progress * 1.26} 126`}
            strokeLinecap="round"
          />
        </svg>
        <span className="absolute inset-0 flex items-center justify-center text-xs font-bold">
          {quest.progress}%
        </span>
      </div>

      {/* Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <h3 className="font-bold text-white truncate">{quest.title}</h3>
          <span className={`text-xs px-2 py-0.5 rounded-full ${getStatusColor(quest.status)}`}>
            {quest.status.replace("_", " ")}
          </span>
        </div>
        <p className="text-sm text-white/50 truncate">{quest.description}</p>
      </div>

      {/* Stats */}
      <div className="hidden sm:flex items-center gap-6 text-sm text-white/50">
        <div className="flex items-center gap-1">
          <BookOpen className="w-4 h-4" />
          {quest.completedModules}/{quest.totalModules}
        </div>
        <div className="flex items-center gap-1">
          <Clock className="w-4 h-4" />
          {quest.estimatedHours}h
        </div>
        {quest.streak > 0 && (
          <div className="flex items-center gap-1 text-amber-400">
            <Flame className="w-4 h-4" />
            {quest.streak}
          </div>
        )}
      </div>

      <ChevronRight className="w-5 h-5 text-white/30" />
    </motion.div>
  );

  return (
    <div className="min-h-[calc(100vh-64px)] bg-gradient-to-b from-background to-background/50">
      <AnimatePresence mode="wait">
        {showWizard ? (
          <motion.div
            key="wizard"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="container mx-auto px-4 py-8"
          >
            {/* Wizard Header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-purple-600 flex items-center justify-center">
                  <Sparkles className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-white">Create Learning Quest</h1>
                  <p className="text-sm text-white/50">AI-powered curriculum generation</p>
                </div>
              </div>
              <Button
                variant="ghost"
                onClick={() => setShowWizard(false)}
              >
                Cancel
              </Button>
            </div>

            {/* Wizard Component */}
            <CurriculumWizard
              courseId="new_course"
              onComplete={handleCurriculumComplete}
            />
          </motion.div>
        ) : (
          <motion.div
            key="list"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="container mx-auto px-4 py-8"
          >
            {/* Page Header */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-8">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center">
                  <Map className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-white">Learning Quests</h1>
                  <p className="text-sm text-white/50">Your personalized learning journeys</p>
                </div>
              </div>
              <Button onClick={() => setShowWizard(true)} className="gap-2">
                <Plus className="w-4 h-4" />
                New Quest
              </Button>
            </div>

            {/* Stats Banner */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
              <div className="p-4 rounded-xl border border-white/10 bg-white/5">
                <div className="flex items-center gap-2 mb-2">
                  <BookOpen className="w-4 h-4 text-primary" />
                  <span className="text-xs text-white/50">Total Quests</span>
                </div>
                <span className="text-2xl font-bold text-white">{quests.length}</span>
              </div>
              <div className="p-4 rounded-xl border border-white/10 bg-white/5">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="w-4 h-4 text-emerald-400" />
                  <span className="text-xs text-white/50">Active</span>
                </div>
                <span className="text-2xl font-bold text-white">
                  {quests.filter((q) => q.status === "active").length}
                </span>
              </div>
              <div className="p-4 rounded-xl border border-white/10 bg-white/5">
                <div className="flex items-center gap-2 mb-2">
                  <Trophy className="w-4 h-4 text-purple-400" />
                  <span className="text-xs text-white/50">Completed</span>
                </div>
                <span className="text-2xl font-bold text-white">
                  {quests.filter((q) => q.status === "completed").length}
                </span>
              </div>
              <div className="p-4 rounded-xl border border-white/10 bg-white/5">
                <div className="flex items-center gap-2 mb-2">
                  <Flame className="w-4 h-4 text-amber-400" />
                  <span className="text-xs text-white/50">Best Streak</span>
                </div>
                <span className="text-2xl font-bold text-white">
                  {Math.max(...quests.map((q) => q.streak), 0)}
                </span>
              </div>
            </div>

            {/* Filters */}
            <div className="flex flex-col sm:flex-row gap-4 mb-6">
              {/* Search */}
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" />
                <input
                  type="text"
                  placeholder="Search quests..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 rounded-lg bg-white/5 border border-white/10 text-white placeholder:text-white/40 focus:outline-none focus:ring-2 focus:ring-primary/50"
                />
              </div>

              {/* Status Filter */}
              <div className="flex items-center gap-2">
                <Filter className="w-4 h-4 text-white/40" />
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value as QuestStatus | "all")}
                  className="px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white focus:outline-none focus:ring-2 focus:ring-primary/50"
                >
                  <option value="all">All Status</option>
                  <option value="active">Active</option>
                  <option value="completed">Completed</option>
                  <option value="not_started">Not Started</option>
                </select>
              </div>

              {/* View Toggle */}
              <div className="flex items-center gap-1 p-1 rounded-lg bg-white/5 border border-white/10">
                <Button
                  variant={viewMode === "grid" ? "secondary" : "ghost"}
                  size="sm"
                  onClick={() => setViewMode("grid")}
                >
                  <LayoutGrid className="w-4 h-4" />
                </Button>
                <Button
                  variant={viewMode === "list" ? "secondary" : "ghost"}
                  size="sm"
                  onClick={() => setViewMode("list")}
                >
                  <List className="w-4 h-4" />
                </Button>
              </div>
            </div>

            {/* Quests Display */}
            {filteredQuests.length === 0 ? (
              <div className="text-center py-16">
                <Map className="w-16 h-16 text-white/20 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-white/60 mb-2">No quests found</h3>
                <p className="text-sm text-white/40 mb-6">
                  {quests.length === 0
                    ? "Start your learning journey by creating your first quest"
                    : "Try adjusting your filters"}
                </p>
                {quests.length === 0 && (
                  <Button onClick={() => setShowWizard(true)} className="gap-2">
                    <Plus className="w-4 h-4" />
                    Create Your First Quest
                  </Button>
                )}
              </div>
            ) : viewMode === "grid" ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <AnimatePresence>
                  {filteredQuests.map(renderQuestCard)}
                </AnimatePresence>
              </div>
            ) : (
              <div className="space-y-3">
                <AnimatePresence>
                  {filteredQuests.map(renderQuestListItem)}
                </AnimatePresence>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
