"use client";

/**
 * Cognitive Dashboard - "Learning DNA" Visualization
 *
 * Displays the learner's cognitive profile including:
 * - Frustration levels with real-time tracking
 * - Calibration metrics (confidence vs performance)
 * - Intervention history
 * - Metacognitive insights
 */

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Brain,
  Activity,
  Target,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Minus,
  Sparkles,
  Lightbulb,
  RefreshCw,
  ChevronRight,
  Clock,
  Zap,
  Heart,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  useCognitiveProfile,
  useFrustration,
  useCalibration,
  useIntervention,
} from "@/hooks/use-cognitive";
import type {
  FrustrationLevel,
  CalibrationLevel,
  InterventionDecision,
} from "@/types/cognitive";

interface CognitiveDashboardProps {
  userId: string;
  onInterventionTriggered?: (intervention: InterventionDecision) => void;
}

// Frustration level colors
const frustrationColors: Record<FrustrationLevel, string> = {
  none: "text-emerald-400",
  mild: "text-yellow-400",
  moderate: "text-orange-400",
  high: "text-red-400",
  severe: "text-red-600",
};

const frustrationBgColors: Record<FrustrationLevel, string> = {
  none: "bg-emerald-500/20",
  mild: "bg-yellow-500/20",
  moderate: "bg-orange-500/20",
  high: "bg-red-500/20",
  severe: "bg-red-600/20",
};

// Calibration level colors
const calibrationColors: Record<CalibrationLevel, string> = {
  unknown: "text-gray-400",
  well_calibrated: "text-emerald-400",
  overconfident: "text-amber-400",
  underconfident: "text-blue-400",
  variable: "text-purple-400",
};

export function CognitiveDashboard({
  userId,
  onInterventionTriggered,
}: CognitiveDashboardProps) {
  const [activeTab, setActiveTab] = useState<"overview" | "frustration" | "calibration" | "interventions">("overview");

  const { profile, isLoading: profileLoading, loadProfile } = useCognitiveProfile({
    userId,
    autoLoad: true,
  });

  const {
    frustration,
    isLoading: frustrationLoading,
    detectFrustration,
    addEvent,
  } = useFrustration({ userId, autoDetect: false });

  const {
    calibration,
    feedback: calibrationFeedback,
    isLoading: calibrationLoading,
    calculateCalibration,
    getCalibrationFeedback,
  } = useCalibration({ userId });

  const {
    decision: intervention,
    history: interventionHistory,
    isLoading: interventionLoading,
    checkIntervention,
    getHistory,
    dismissIntervention,
  } = useIntervention({ userId, onIntervention: onInterventionTriggered });

  // Load data on mount
  useEffect(() => {
    calculateCalibration();
    getCalibrationFeedback();
    getHistory();
  }, [calculateCalibration, getCalibrationFeedback, getHistory]);

  const isLoading = profileLoading || frustrationLoading || calibrationLoading || interventionLoading;

  const renderOverview = () => (
    <div className="space-y-6">
      {/* Quick Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Frustration Level */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className={`p-4 rounded-xl border border-white/10 ${
            frustrationBgColors[frustration?.level || "none"]
          }`}
        >
          <div className="flex items-center gap-2 mb-2">
            <Activity className={`w-4 h-4 ${frustrationColors[frustration?.level || "none"]}`} />
            <span className="text-xs text-white/60">Frustration</span>
          </div>
          <div className={`text-2xl font-bold ${frustrationColors[frustration?.level || "none"]}`}>
            {frustration?.level || "None"}
          </div>
          <div className="text-xs text-white/40 mt-1">
            Score: {((frustration?.score || 0) * 100).toFixed(0)}%
          </div>
        </motion.div>

        {/* Calibration Level */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="p-4 rounded-xl border border-white/10 bg-white/5"
        >
          <div className="flex items-center gap-2 mb-2">
            <Target className={`w-4 h-4 ${calibrationColors[calibration?.calibration_level || "unknown"]}`} />
            <span className="text-xs text-white/60">Calibration</span>
          </div>
          <div className={`text-2xl font-bold capitalize ${calibrationColors[calibration?.calibration_level || "unknown"]}`}>
            {(calibration?.calibration_level || "unknown").replace("_", " ")}
          </div>
          <div className="text-xs text-white/40 mt-1">
            Error: {((calibration?.calibration_error || 0) * 100).toFixed(1)}%
          </div>
        </motion.div>

        {/* Confidence */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="p-4 rounded-xl border border-white/10 bg-white/5"
        >
          <div className="flex items-center gap-2 mb-2">
            <Sparkles className="w-4 h-4 text-purple-400" />
            <span className="text-xs text-white/60">Avg Confidence</span>
          </div>
          <div className="text-2xl font-bold text-purple-400">
            {((calibration?.mean_confidence || 0) * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-white/40 mt-1">
            {calibration?.data_points || 0} data points
          </div>
        </motion.div>

        {/* Performance */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="p-4 rounded-xl border border-white/10 bg-white/5"
        >
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-amber-400" />
            <span className="text-xs text-white/60">Avg Performance</span>
          </div>
          <div className="text-2xl font-bold text-amber-400">
            {((calibration?.mean_performance || 0) * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-white/40 mt-1">
            {calibration?.overconfidence_rate && calibration.overconfidence_rate > 0.3
              ? "Overconfident"
              : calibration?.underconfidence_rate && calibration.underconfidence_rate > 0.3
              ? "Underconfident"
              : "Well calibrated"}
          </div>
        </motion.div>
      </div>

      {/* Calibration Feedback */}
      {calibrationFeedback && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="p-4 rounded-xl border border-white/10 bg-gradient-to-r from-purple-500/10 to-blue-500/10"
        >
          <div className="flex items-start gap-3">
            <Lightbulb className="w-5 h-5 text-purple-400 mt-0.5" />
            <div>
              <h4 className="font-semibold text-white mb-1">Calibration Insight</h4>
              <p className="text-sm text-white/70">{calibrationFeedback.message}</p>
              {calibrationFeedback.recommendations.length > 0 && (
                <ul className="mt-2 space-y-1">
                  {calibrationFeedback.recommendations.map((rec, i) => (
                    <li key={i} className="text-xs text-white/50 flex items-center gap-2">
                      <ChevronRight className="w-3 h-3" />
                      {rec}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </motion.div>
      )}

      {/* Intervention History Summary */}
      {interventionHistory && interventionHistory.total_interventions > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="p-4 rounded-xl border border-white/10 bg-white/5"
        >
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Heart className="w-4 h-4 text-pink-400" />
              <span className="text-sm font-medium text-white">Intervention History</span>
            </div>
            <span className="text-xs text-white/50">
              {interventionHistory.total_interventions} total
            </span>
          </div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(interventionHistory.by_type || {}).map(([type, count]) => (
              <div
                key={type}
                className="px-2 py-1 rounded-full bg-white/10 text-xs text-white/70"
              >
                {type}: {count as number}
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Active Intervention */}
      <AnimatePresence>
        {intervention?.should_intervene && intervention.intervention && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -20 }}
            className={`p-4 rounded-xl border-2 ${
              intervention.intervention.priority === "critical"
                ? "border-red-500/50 bg-red-500/10"
                : intervention.intervention.priority === "high"
                ? "border-orange-500/50 bg-orange-500/10"
                : "border-blue-500/50 bg-blue-500/10"
            }`}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <AlertTriangle
                    className={`w-4 h-4 ${
                      intervention.intervention.priority === "critical"
                        ? "text-red-400"
                        : intervention.intervention.priority === "high"
                        ? "text-orange-400"
                        : "text-blue-400"
                    }`}
                  />
                  <span className="font-semibold text-white">
                    {intervention.intervention.title}
                  </span>
                </div>
                <p className="text-sm text-white/70">{intervention.intervention.message}</p>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={dismissIntervention}
                className="shrink-0"
              >
                Dismiss
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );

  const renderFrustrationDetails = () => (
    <div className="space-y-6">
      {/* Frustration Gauge */}
      <div className="p-6 rounded-xl border border-white/10 bg-white/5">
        <h4 className="font-semibold text-white mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          Frustration Analysis
        </h4>

        {/* Score Bar */}
        <div className="mb-6">
          <div className="flex justify-between text-xs text-white/50 mb-2">
            <span>None</span>
            <span>Mild</span>
            <span>Moderate</span>
            <span>High</span>
            <span>Severe</span>
          </div>
          <div className="h-3 bg-white/10 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${(frustration?.score || 0) * 100}%` }}
              transition={{ duration: 0.5 }}
              className={`h-full ${
                frustration?.level === "severe"
                  ? "bg-red-500"
                  : frustration?.level === "high"
                  ? "bg-red-400"
                  : frustration?.level === "moderate"
                  ? "bg-orange-400"
                  : frustration?.level === "mild"
                  ? "bg-yellow-400"
                  : "bg-emerald-400"
              }`}
            />
          </div>
        </div>

        {/* Struggle Type */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="p-3 rounded-lg bg-white/5">
            <div className="text-xs text-white/50 mb-1">Struggle Type</div>
            <div className="font-medium text-white capitalize">
              {frustration?.struggle_type || "None"}
            </div>
          </div>
          <div className="p-3 rounded-lg bg-white/5">
            <div className="text-xs text-white/50 mb-1">Confidence</div>
            <div className="font-medium text-white">
              {((frustration?.confidence || 0) * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        {/* Active Signals */}
        {frustration?.active_signals && frustration.active_signals.length > 0 && (
          <div>
            <div className="text-xs text-white/50 mb-2">Active Behavioral Signals</div>
            <div className="flex flex-wrap gap-2">
              {frustration.active_signals.map((signal) => (
                <div
                  key={signal}
                  className="px-2 py-1 rounded-full bg-red-500/20 text-red-400 text-xs"
                >
                  {signal.replace(/_/g, " ")}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recommended Action */}
        {frustration?.recommended_action && (
          <div className="mt-4 p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
            <div className="text-xs text-blue-400 mb-1">Recommended Action</div>
            <div className="text-sm text-white">{frustration.recommended_action}</div>
          </div>
        )}
      </div>
    </div>
  );

  const renderCalibrationDetails = () => (
    <div className="space-y-6">
      <div className="p-6 rounded-xl border border-white/10 bg-white/5">
        <h4 className="font-semibold text-white mb-4 flex items-center gap-2">
          <Target className="w-5 h-5" />
          Calibration Analysis
        </h4>

        {/* Confidence vs Performance */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="p-4 rounded-lg bg-white/5">
            <div className="flex items-center gap-2 mb-2">
              {calibration?.mean_confidence !== undefined &&
              calibration?.mean_performance !== undefined &&
              calibration.mean_confidence > calibration.mean_performance ? (
                <TrendingUp className="w-4 h-4 text-amber-400" />
              ) : calibration?.mean_confidence !== undefined &&
                calibration?.mean_performance !== undefined &&
                calibration.mean_confidence < calibration.mean_performance ? (
                <TrendingDown className="w-4 h-4 text-blue-400" />
              ) : (
                <Minus className="w-4 h-4 text-emerald-400" />
              )}
              <span className="text-xs text-white/50">Confidence Gap</span>
            </div>
            <div className="text-2xl font-bold text-white">
              {calibration?.mean_confidence !== undefined && calibration?.mean_performance !== undefined
                ? `${((calibration.mean_confidence - calibration.mean_performance) * 100).toFixed(1)}%`
                : "N/A"}
            </div>
            <div className="text-xs text-white/40">
              {calibration?.mean_confidence !== undefined &&
              calibration?.mean_performance !== undefined &&
              calibration.mean_confidence > calibration.mean_performance + 0.1
                ? "You may be overconfident"
                : calibration?.mean_confidence !== undefined &&
                  calibration?.mean_performance !== undefined &&
                  calibration.mean_confidence < calibration.mean_performance - 0.1
                ? "You may be underconfident"
                : "Well calibrated"}
            </div>
          </div>
          <div className="p-4 rounded-lg bg-white/5">
            <div className="text-xs text-white/50 mb-2">Calibration Error</div>
            <div className="text-2xl font-bold text-white">
              {((calibration?.calibration_error || 0) * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-white/40">
              Lower is better
            </div>
          </div>
        </div>

        {/* Overconfidence / Underconfidence Rates */}
        <div className="grid grid-cols-2 gap-4">
          <div className="p-3 rounded-lg bg-amber-500/10">
            <div className="text-xs text-amber-400 mb-1">Overconfidence Rate</div>
            <div className="font-medium text-white">
              {((calibration?.overconfidence_rate || 0) * 100).toFixed(0)}%
            </div>
          </div>
          <div className="p-3 rounded-lg bg-blue-500/10">
            <div className="text-xs text-blue-400 mb-1">Underconfidence Rate</div>
            <div className="font-medium text-white">
              {((calibration?.underconfidence_rate || 0) * 100).toFixed(0)}%
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="w-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
            <Brain className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">Learning DNA</h2>
            <p className="text-xs text-white/50">Cognitive Profile Dashboard</p>
          </div>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={loadProfile}
          disabled={isLoading}
        >
          <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
        {[
          { id: "overview", label: "Overview", icon: Brain },
          { id: "frustration", label: "Frustration", icon: Activity },
          { id: "calibration", label: "Calibration", icon: Target },
        ].map((tab) => (
          <Button
            key={tab.id}
            variant={activeTab === tab.id ? "secondary" : "ghost"}
            size="sm"
            onClick={() => setActiveTab(tab.id as any)}
            className="gap-2 whitespace-nowrap"
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </Button>
        ))}
      </div>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.2 }}
        >
          {activeTab === "overview" && renderOverview()}
          {activeTab === "frustration" && renderFrustrationDetails()}
          {activeTab === "calibration" && renderCalibrationDetails()}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

export default CognitiveDashboard;
