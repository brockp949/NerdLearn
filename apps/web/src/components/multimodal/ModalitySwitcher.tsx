"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FileText,
  GitBranch,
  Headphones,
  Loader2,
  Sparkles,
  ChevronDown,
} from "lucide-react";
import type {
  ContentModality,
  DiagramData,
  PodcastEpisode,
  ModalityRecommendation,
} from "@/types/multimodal";

interface ModalitySwitcherProps {
  currentModality: ContentModality;
  onModalityChange: (modality: ContentModality) => void;
  isLoading?: boolean;
  recommendation?: ModalityRecommendation | null;
  showRecommendation?: boolean;
}

const modalityConfig = {
  text: {
    icon: FileText,
    label: "Text",
    description: "Traditional written content",
    color: "from-blue-500 to-blue-600",
    bgColor: "bg-blue-500/10",
    borderColor: "border-blue-500/30",
    textColor: "text-blue-400",
  },
  diagram: {
    icon: GitBranch,
    label: "Diagram",
    description: "Interactive visual map",
    color: "from-purple-500 to-purple-600",
    bgColor: "bg-purple-500/10",
    borderColor: "border-purple-500/30",
    textColor: "text-purple-400",
  },
  podcast: {
    icon: Headphones,
    label: "Podcast",
    description: "Audio explanation",
    color: "from-green-500 to-green-600",
    bgColor: "bg-green-500/10",
    borderColor: "border-green-500/30",
    textColor: "text-green-400",
  },
};

export function ModalitySwitcher({
  currentModality,
  onModalityChange,
  isLoading = false,
  recommendation,
  showRecommendation = true,
}: ModalitySwitcherProps) {
  const [showDropdown, setShowDropdown] = useState(false);

  const currentConfig = modalityConfig[currentModality];
  const CurrentIcon = currentConfig.icon;

  return (
    <div className="relative">
      {/* Main Switcher Button */}
      <button
        onClick={() => setShowDropdown(!showDropdown)}
        disabled={isLoading}
        className={`
          flex items-center gap-3 px-4 py-2.5 rounded-xl
          border ${currentConfig.borderColor} ${currentConfig.bgColor}
          hover:bg-white/5 transition-all duration-200
          ${isLoading ? "opacity-50 cursor-not-allowed" : ""}
        `}
      >
        {isLoading ? (
          <Loader2 className={`w-5 h-5 ${currentConfig.textColor} animate-spin`} />
        ) : (
          <CurrentIcon className={`w-5 h-5 ${currentConfig.textColor}`} />
        )}
        <div className="flex flex-col items-start">
          <span className="text-sm font-medium text-white">
            {currentConfig.label}
          </span>
          <span className="text-xs text-white/50">
            {currentConfig.description}
          </span>
        </div>
        <ChevronDown
          className={`w-4 h-4 text-white/40 transition-transform ${
            showDropdown ? "rotate-180" : ""
          }`}
        />
      </button>

      {/* Dropdown Menu */}
      <AnimatePresence>
        {showDropdown && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute top-full left-0 right-0 mt-2 z-50"
          >
            <div className="rounded-xl border border-white/10 bg-black/90 backdrop-blur-xl overflow-hidden shadow-xl">
              {/* Modality Options */}
              {(Object.keys(modalityConfig) as ContentModality[]).map(
                (modality) => {
                  const config = modalityConfig[modality];
                  const Icon = config.icon;
                  const isActive = modality === currentModality;
                  const isRecommended =
                    recommendation?.recommended_modality === modality;

                  return (
                    <button
                      key={modality}
                      onClick={() => {
                        onModalityChange(modality);
                        setShowDropdown(false);
                      }}
                      disabled={isActive || isLoading}
                      className={`
                        w-full flex items-center gap-3 px-4 py-3
                        hover:bg-white/5 transition-colors
                        ${isActive ? "bg-white/10" : ""}
                        ${isActive ? "cursor-default" : "cursor-pointer"}
                      `}
                    >
                      <div
                        className={`
                        w-10 h-10 rounded-lg flex items-center justify-center
                        ${isActive ? `bg-gradient-to-br ${config.color}` : config.bgColor}
                      `}
                      >
                        <Icon
                          className={`w-5 h-5 ${
                            isActive ? "text-white" : config.textColor
                          }`}
                        />
                      </div>
                      <div className="flex-1 text-left">
                        <div className="flex items-center gap-2">
                          <span
                            className={`text-sm font-medium ${
                              isActive ? "text-white" : "text-white/80"
                            }`}
                          >
                            {config.label}
                          </span>
                          {isRecommended && showRecommendation && (
                            <span className="flex items-center gap-1 px-1.5 py-0.5 rounded text-xs bg-amber-500/20 text-amber-400">
                              <Sparkles className="w-3 h-3" />
                              Suggested
                            </span>
                          )}
                          {isActive && (
                            <span className="px-1.5 py-0.5 rounded text-xs bg-white/20 text-white/60">
                              Active
                            </span>
                          )}
                        </div>
                        <span className="text-xs text-white/50">
                          {config.description}
                        </span>
                      </div>
                    </button>
                  );
                }
              )}

              {/* Recommendation Reason */}
              {recommendation && showRecommendation && (
                <div className="px-4 py-3 border-t border-white/10 bg-white/5">
                  <div className="flex items-start gap-2">
                    <Sparkles className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="text-xs text-white/70">
                        <span className="font-medium text-amber-400">
                          AI Suggestion:
                        </span>{" "}
                        {recommendation.reason}
                      </p>
                      {recommendation.weak_concepts.length > 0 && (
                        <p className="text-xs text-white/50 mt-1">
                          Focus on: {recommendation.weak_concepts.slice(0, 3).join(", ")}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Click Outside Handler */}
      {showDropdown && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setShowDropdown(false)}
        />
      )}
    </div>
  );
}

// Compact inline version for toolbars
export function ModalitySwitcherCompact({
  currentModality,
  onModalityChange,
  isLoading = false,
}: Omit<ModalitySwitcherProps, "recommendation" | "showRecommendation">) {
  return (
    <div className="flex items-center gap-1 p-1 rounded-lg bg-white/5 border border-white/10">
      {(Object.keys(modalityConfig) as ContentModality[]).map((modality) => {
        const config = modalityConfig[modality];
        const Icon = config.icon;
        const isActive = modality === currentModality;

        return (
          <button
            key={modality}
            onClick={() => onModalityChange(modality)}
            disabled={isActive || isLoading}
            title={config.label}
            className={`
              p-2 rounded-md transition-all
              ${isActive
                ? `bg-gradient-to-br ${config.color} text-white shadow-lg`
                : "text-white/50 hover:text-white hover:bg-white/10"
              }
              ${isLoading ? "opacity-50 cursor-not-allowed" : ""}
            `}
          >
            {isLoading && isActive ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Icon className="w-4 h-4" />
            )}
          </button>
        );
      })}
    </div>
  );
}

export default ModalitySwitcher;
