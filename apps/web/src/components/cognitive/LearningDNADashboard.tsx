'use client';

/**
 * Learning DNA Dashboard - Metacognitive Mirror
 *
 * Research alignment:
 * - The Metacognitive Mirror: Shows learning personality (not grades)
 * - Teaches "how to learn", not just content
 * - Provides actionable feedback based on behavior patterns
 *
 * Key Features:
 * 1. Visualize learning behavior patterns
 * 2. Calculate personality metrics (Impulsivity, Resilience, etc.)
 * 3. Show modality preferences
 * 4. Generate actionable recommendations
 */

import React, { useEffect, useState, memo, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  LineChart,
  Line,
  Area,
  AreaChart,
} from 'recharts';

// Types
interface LearningTrait {
  name: string;
  value: number; // 0-100
  description: string;
  recommendation: string;
}

interface ModalityPreference {
  modality: string;
  timeSpent: number; // minutes
  engagement: number; // 0-100
  retention: number; // 0-100
}

interface BehaviorPattern {
  pattern: string;
  frequency: number;
  impact: 'positive' | 'negative' | 'neutral';
  description: string;
}

interface LearningSession {
  date: string;
  duration: number;
  focusScore: number;
  progressMade: number;
}

interface LearningDNAData {
  traits: LearningTrait[];
  modalityPreferences: ModalityPreference[];
  behaviorPatterns: BehaviorPattern[];
  recentSessions: LearningSession[];
  overallScore: number;
  primaryLearningStyle: string;
  recommendations: string[];
}

interface LearningDNADashboardProps {
  userId?: number;
  courseId?: number;
}

// Mock data generator (replace with API call)
const generateMockData = (): LearningDNAData => ({
  traits: [
    {
      name: 'Impulsivity',
      value: 65,
      description: 'Tendency to skip instructions',
      recommendation: 'Try reading the theory section before jumping to exercises',
    },
    {
      name: 'Resilience',
      value: 78,
      description: 'Retry count before giving up',
      recommendation: 'Great persistence! Keep pushing through challenges',
    },
    {
      name: 'Focus',
      value: 55,
      description: 'Session attention consistency',
      recommendation: 'Consider shorter, more frequent study sessions',
    },
    {
      name: 'Curiosity',
      value: 82,
      description: 'Exploration of optional content',
      recommendation: 'Channel curiosity into structured deep-dives',
    },
    {
      name: 'Collaboration',
      value: 45,
      description: 'Engagement with social features',
      recommendation: 'Teaching others can reinforce your learning',
    },
    {
      name: 'Reflection',
      value: 60,
      description: 'Time spent reviewing mistakes',
      recommendation: 'Spend more time understanding why answers were wrong',
    },
  ],
  modalityPreferences: [
    { modality: 'Text', timeSpent: 120, engagement: 65, retention: 70 },
    { modality: 'Video', timeSpent: 180, engagement: 85, retention: 75 },
    { modality: 'Interactive', timeSpent: 90, engagement: 90, retention: 85 },
    { modality: 'Podcast', timeSpent: 60, engagement: 70, retention: 60 },
    { modality: 'Practice', timeSpent: 150, engagement: 80, retention: 90 },
  ],
  behaviorPatterns: [
    {
      pattern: 'Skips Theory',
      frequency: 4,
      impact: 'negative',
      description: 'Skipped 4 theory sections this week',
    },
    {
      pattern: 'Night Owl',
      frequency: 8,
      impact: 'neutral',
      description: 'Most productive between 10PM-2AM',
    },
    {
      pattern: 'Marathon Sessions',
      frequency: 2,
      impact: 'negative',
      description: 'Two 3+ hour sessions detected',
    },
    {
      pattern: 'Consistent Practice',
      frequency: 5,
      impact: 'positive',
      description: 'Practiced coding 5 days this week',
    },
  ],
  recentSessions: [
    { date: '2024-01-15', duration: 45, focusScore: 75, progressMade: 12 },
    { date: '2024-01-16', duration: 60, focusScore: 82, progressMade: 18 },
    { date: '2024-01-17', duration: 30, focusScore: 65, progressMade: 8 },
    { date: '2024-01-18', duration: 90, focusScore: 58, progressMade: 15 },
    { date: '2024-01-19', duration: 45, focusScore: 88, progressMade: 20 },
    { date: '2024-01-20', duration: 55, focusScore: 72, progressMade: 14 },
  ],
  overallScore: 72,
  primaryLearningStyle: 'Visual-Kinesthetic',
  recommendations: [
    'You tend to skip the Theory section and get stuck on the Lab. Try the Podcast mode for theory—it might hold your attention better.',
    'Your best retention comes from Interactive content. Seek out more hands-on exercises.',
    'Consider breaking your marathon sessions into 45-minute focused blocks with breaks.',
  ],
});

// Sub-components
const TraitRadar = memo<{ traits: LearningTrait[] }>(function TraitRadar({ traits }) {
  const data = useMemo(() => traits.map((t) => ({
    trait: t.name,
    value: t.value,
    fullMark: 100,
  })), [traits]);

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-6 border border-cyan-500/20">
      <h3 className="text-lg font-semibold text-cyan-400 mb-4">Learning DNA Profile</h3>
      <ResponsiveContainer width="100%" height={300}>
        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
          <PolarGrid stroke="#164e63" />
          <PolarAngleAxis
            dataKey="trait"
            tick={{ fill: '#94a3b8', fontSize: 12 }}
          />
          <PolarRadiusAxis
            angle={30}
            domain={[0, 100]}
            tick={{ fill: '#64748b', fontSize: 10 }}
          />
          <Radar
            name="You"
            dataKey="value"
            stroke="#06b6d4"
            fill="#06b6d4"
            fillOpacity={0.3}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
});

const TraitDetails = memo<{ traits: LearningTrait[] }>(function TraitDetails({ traits }) {
  const [selectedTrait, setSelectedTrait] = useState<LearningTrait | null>(null);

  const getTraitColor = (value: number) => {
    if (value >= 75) return 'text-green-400';
    if (value >= 50) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-6 border border-purple-500/20">
      <h3 className="text-lg font-semibold text-purple-400 mb-4">Trait Breakdown</h3>
      <div className="space-y-3">
        {traits.map((trait) => (
          <motion.div
            key={trait.name}
            className="cursor-pointer"
            onClick={() => setSelectedTrait(selectedTrait?.name === trait.name ? null : trait)}
            whileHover={{ scale: 1.02 }}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="text-gray-300 text-sm">{trait.name}</span>
              <span className={`font-mono ${getTraitColor(trait.value)}`}>
                {trait.value}%
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <motion.div
                className="h-2 rounded-full bg-gradient-to-r from-purple-500 to-cyan-500"
                initial={{ width: 0 }}
                animate={{ width: `${trait.value}%` }}
                transition={{ duration: 0.8, ease: 'easeOut' }}
              />
            </div>
            <AnimatePresence>
              {selectedTrait?.name === trait.name && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-2 p-3 bg-gray-800/50 rounded-lg"
                >
                  <p className="text-gray-400 text-xs mb-2">{trait.description}</p>
                  <p className="text-cyan-400 text-xs">
                    <span className="font-semibold">Tip:</span> {trait.recommendation}
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        ))}
      </div>
    </div>
  );
});

const ModalityChart = memo<{ preferences: ModalityPreference[] }>(function ModalityChart({ preferences }) {
  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-6 border border-emerald-500/20">
      <h3 className="text-lg font-semibold text-emerald-400 mb-4">Modality Preferences</h3>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={preferences} layout="vertical">
          <XAxis type="number" domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 10 }} />
          <YAxis dataKey="modality" type="category" tick={{ fill: '#94a3b8', fontSize: 12 }} width={80} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1f2937',
              border: '1px solid #374151',
              borderRadius: '8px',
            }}
            labelStyle={{ color: '#f3f4f6' }}
          />
          <Bar dataKey="engagement" fill="#10b981" name="Engagement" radius={[0, 4, 4, 0]} />
          <Bar dataKey="retention" fill="#06b6d4" name="Retention" radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
      <div className="flex justify-center gap-6 mt-4">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-emerald-500 rounded" />
          <span className="text-gray-400 text-xs">Engagement</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-cyan-500 rounded" />
          <span className="text-gray-400 text-xs">Retention</span>
        </div>
      </div>
    </div>
  );
});

const BehaviorPatterns = memo<{ patterns: BehaviorPattern[] }>(function BehaviorPatterns({ patterns }) {
  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'positive':
        return 'bg-green-500/20 border-green-500/50 text-green-400';
      case 'negative':
        return 'bg-red-500/20 border-red-500/50 text-red-400';
      default:
        return 'bg-gray-500/20 border-gray-500/50 text-gray-400';
    }
  };

  const getImpactIcon = (impact: string) => {
    switch (impact) {
      case 'positive':
        return '↑';
      case 'negative':
        return '↓';
      default:
        return '→';
    }
  };

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-6 border border-amber-500/20">
      <h3 className="text-lg font-semibold text-amber-400 mb-4">Behavior Patterns Detected</h3>
      <div className="grid grid-cols-2 gap-3">
        {patterns.map((pattern) => (
          <motion.div
            key={pattern.pattern}
            className={`p-3 rounded-lg border ${getImpactColor(pattern.impact)}`}
            whileHover={{ scale: 1.02 }}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="font-medium text-sm">{pattern.pattern}</span>
              <span className="text-lg">{getImpactIcon(pattern.impact)}</span>
            </div>
            <p className="text-xs opacity-80">{pattern.description}</p>
          </motion.div>
        ))}
      </div>
    </div>
  );
});

const SessionTrends = memo<{ sessions: LearningSession[] }>(function SessionTrends({ sessions }) {
  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-6 border border-pink-500/20">
      <h3 className="text-lg font-semibold text-pink-400 mb-4">Session Trends</h3>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={sessions}>
          <defs>
            <linearGradient id="focusGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ec4899" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#ec4899" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="date"
            tick={{ fill: '#94a3b8', fontSize: 10 }}
            tickFormatter={(value) => value.split('-')[2]}
          />
          <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} domain={[0, 100]} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1f2937',
              border: '1px solid #374151',
              borderRadius: '8px',
            }}
          />
          <Area
            type="monotone"
            dataKey="focusScore"
            stroke="#ec4899"
            fill="url(#focusGradient)"
            strokeWidth={2}
          />
          <Line
            type="monotone"
            dataKey="progressMade"
            stroke="#8b5cf6"
            strokeWidth={2}
            dot={{ fill: '#8b5cf6', r: 4 }}
          />
        </AreaChart>
      </ResponsiveContainer>
      <div className="flex justify-center gap-6 mt-4">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-pink-500 rounded" />
          <span className="text-gray-400 text-xs">Focus Score</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-purple-500 rounded" />
          <span className="text-gray-400 text-xs">Progress</span>
        </div>
      </div>
    </div>
  );
});

const RecommendationCard = memo<{ recommendations: string[] }>(function RecommendationCard({ recommendations }) {
  return (
    <div className="bg-gradient-to-br from-cyan-900/30 to-purple-900/30 backdrop-blur-sm rounded-xl p-6 border border-cyan-500/30">
      <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <span className="text-2xl">💡</span>
        Personalized Recommendations
      </h3>
      <ul className="space-y-4">
        {recommendations.map((rec, index) => (
          <motion.li
            key={index}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.2 }}
            className="flex items-start gap-3"
          >
            <span className="flex-shrink-0 w-6 h-6 bg-cyan-500/20 rounded-full flex items-center justify-center text-cyan-400 text-sm font-bold">
              {index + 1}
            </span>
            <p className="text-gray-300 text-sm leading-relaxed">{rec}</p>
          </motion.li>
        ))}
      </ul>
    </div>
  );
});

const OverallScore = memo<{ score: number; style: string }>(function OverallScore({ score, style }) {
  const circumference = 2 * Math.PI * 45;
  const progress = ((100 - score) / 100) * circumference;

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-6 border border-blue-500/20 flex flex-col items-center">
      <h3 className="text-lg font-semibold text-blue-400 mb-4">Learning Effectiveness</h3>
      <div className="relative w-32 h-32">
        <svg className="transform -rotate-90 w-32 h-32">
          <circle
            cx="64"
            cy="64"
            r="45"
            stroke="#374151"
            strokeWidth="10"
            fill="transparent"
          />
          <motion.circle
            cx="64"
            cy="64"
            r="45"
            stroke="url(#scoreGradient)"
            strokeWidth="10"
            fill="transparent"
            strokeLinecap="round"
            initial={{ strokeDasharray: circumference, strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: progress }}
            transition={{ duration: 1.5, ease: 'easeOut' }}
          />
          <defs>
            <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#06b6d4" />
              <stop offset="100%" stopColor="#8b5cf6" />
            </linearGradient>
          </defs>
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span
            className="text-3xl font-bold text-white"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.5, type: 'spring' }}
          >
            {score}
          </motion.span>
          <span className="text-xs text-gray-400">Score</span>
        </div>
      </div>
      <div className="mt-4 text-center">
        <p className="text-gray-400 text-sm">Primary Learning Style</p>
        <p className="text-cyan-400 font-semibold">{style}</p>
      </div>
    </div>
  );
});

// Main Component
export const LearningDNADashboard = memo<LearningDNADashboardProps>(function LearningDNADashboard({
  userId,
  courseId,
}) {
  const [data, setData] = useState<LearningDNAData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // In production, fetch from API
    // const fetchData = async () => {
    //   const response = await api.get(`/cognitive/learning-dna/${userId}`);
    //   setData(response.data);
    // };

    // Mock data for now
    const timer = setTimeout(() => {
      setData(generateMockData());
      setLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, [userId, courseId]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <motion.div
          className="w-16 h-16 border-4 border-cyan-500/30 border-t-cyan-500 rounded-full"
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <p className="text-gray-400">Unable to load learning data</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 p-6">
      {/* Header */}
      <motion.div
        className="mb-8"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
          Your Learning DNA
        </h1>
        <p className="text-gray-400 mt-2">
          Understanding how you learn is the first step to learning better
        </p>
      </motion.div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column */}
        <div className="space-y-6">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
          >
            <OverallScore score={data.overallScore} style={data.primaryLearningStyle} />
          </motion.div>
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <TraitDetails traits={data.traits} />
          </motion.div>
        </div>

        {/* Center Column */}
        <div className="space-y-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <TraitRadar traits={data.traits} />
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <ModalityChart preferences={data.modalityPreferences} />
          </motion.div>
        </div>

        {/* Right Column */}
        <div className="space-y-6">
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 }}
          >
            <BehaviorPatterns patterns={data.behaviorPatterns} />
          </motion.div>
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.6 }}
          >
            <SessionTrends sessions={data.recentSessions} />
          </motion.div>
        </div>
      </div>

      {/* Recommendations (Full Width) */}
      <motion.div
        className="mt-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
      >
        <RecommendationCard recommendations={data.recommendations} />
      </motion.div>
    </div>
  );
});

export default LearningDNADashboard;
