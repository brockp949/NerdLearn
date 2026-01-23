"use client";

/**
 * Multimodal Player - Universal Content Viewer
 *
 * A unified component that can display content in any modality:
 * - Text: Markdown/rich text content
 * - Diagram: Interactive React Flow diagrams
 * - Podcast: Audio player with transcript
 *
 * Handles seamless morphing between modalities while preserving state.
 */

import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FileText,
  GitBranch,
  Headphones,
  Sparkles,
  RefreshCw,
  BarChart2,
  Clock,
  Loader2,
  Info,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useMultimodal, useMultimodalMetadata } from "@/hooks/use-multimodal";
import { ModalitySwitcher, ModalitySwitcherCompact } from "./ModalitySwitcher";
import { InteractiveDiagram } from "./InteractiveDiagram";
import { PodcastPlayer } from "./PodcastPlayer";
import type {
  ContentModality,
  DiagramData,
  PodcastEpisode,
  LearningSummary,
} from "@/types/multimodal";

interface MultimodalPlayerProps {
  userId: string;
  contentId: string;
  initialContent: string;
  initialModality?: ContentModality;
  topic?: string;
  showProgress?: boolean;
  onModalityChange?: (modality: ContentModality) => void;
}

export function MultimodalPlayer({
  userId,
  contentId,
  initialContent,
  initialModality = "text",
  topic = "Content",
  showProgress = true,
  onModalityChange,
}: MultimodalPlayerProps) {
  const [content, setContent] = useState<string>(initialContent);
  const [textContent, setTextContent] = useState<string>(initialContent);
  const [diagramData, setDiagramData] = useState<DiagramData | null>(null);
  const [podcastData, setPodcastData] = useState<PodcastEpisode | null>(null);

  const {
    currentModality,
    setCurrentModality,
    conceptualState,
    learningSummary,
    recommendation,
    isLoading,
    isMorphing,
    isGeneratingPodcast,
    isGeneratingDiagram,
    switchModality,
    generatePodcast,
    generateDiagram,
    refetchState,
    refetchSummary,
  } = useMultimodal(userId, contentId);

  const { modalities } = useMultimodalMetadata();

  // Initialize modality
  useEffect(() => {
    setCurrentModality(initialModality);
  }, [initialModality, setCurrentModality]);

  // Handle modality switch
  const handleModalityChange = useCallback(
    async (newModality: ContentModality) => {
      if (newModality === currentModality || isLoading) return;

      try {
        if (newModality === "diagram") {
          const result = await generateDiagram(textContent, {
            diagramType: "concept_map",
            title: topic,
          });
          setDiagramData(result);
        } else if (newModality === "podcast") {
          const result = await generatePodcast(textContent, topic, {
            durationMinutes: 5,
            style: "educational",
          });
          setPodcastData(result);
        }

        setCurrentModality(newModality);
        onModalityChange?.(newModality);
      } catch (error) {
        console.error("Failed to switch modality:", error);
      }
    },
    [
      currentModality,
      isLoading,
      textContent,
      topic,
      generateDiagram,
      generatePodcast,
      setCurrentModality,
      onModalityChange,
    ]
  );

  // Render text content
  const renderTextContent = () => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="prose prose-invert max-w-none"
    >
      <div className="p-6 rounded-xl border border-white/10 bg-white/5">
        {/* Simple markdown rendering - in production use react-markdown */}
        <div
          className="text-white/80 leading-relaxed"
          dangerouslySetInnerHTML={{
            __html: textContent
              .split("\n")
              .map((line) => {
                if (line.startsWith("# ")) {
                  return `<h1 class="text-2xl font-bold text-white mb-4">${line.slice(2)}</h1>`;
                }
                if (line.startsWith("## ")) {
                  return `<h2 class="text-xl font-semibold text-white mt-6 mb-3">${line.slice(3)}</h2>`;
                }
                if (line.startsWith("### ")) {
                  return `<h3 class="text-lg font-medium text-white mt-4 mb-2">${line.slice(4)}</h3>`;
                }
                if (line.startsWith("- ")) {
                  return `<li class="ml-4 mb-1">${line.slice(2)}</li>`;
                }
                if (line.trim() === "") {
                  return "<br />";
                }
                return `<p class="mb-3">${line}</p>`;
              })
              .join(""),
          }}
        />
      </div>
    </motion.div>
  );

  // Render diagram content
  const renderDiagramContent = () => (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className="rounded-xl border border-white/10 bg-white/5 overflow-hidden"
    >
      {diagramData ? (
        <InteractiveDiagram
          diagram={diagramData}
          onNodeClick={(nodeId) => console.log("Node clicked:", nodeId)}
        />
      ) : (
        <div className="h-96 flex items-center justify-center">
          {isGeneratingDiagram ? (
            <div className="text-center">
              <Loader2 className="w-8 h-8 text-purple-400 animate-spin mx-auto mb-3" />
              <p className="text-white/60">Generating diagram...</p>
            </div>
          ) : (
            <div className="text-center">
              <GitBranch className="w-8 h-8 text-white/30 mx-auto mb-3" />
              <p className="text-white/50">No diagram generated yet</p>
              <Button
                variant="secondary"
                size="sm"
                className="mt-3"
                onClick={() => handleModalityChange("diagram")}
              >
                Generate Diagram
              </Button>
            </div>
          )}
        </div>
      )}
    </motion.div>
  );

  // Render podcast content
  const renderPodcastContent = () => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="rounded-xl border border-white/10 bg-white/5"
    >
      {podcastData ? (
        <PodcastPlayer episode={podcastData} />
      ) : (
        <div className="h-64 flex items-center justify-center">
          {isGeneratingPodcast ? (
            <div className="text-center">
              <Loader2 className="w-8 h-8 text-green-400 animate-spin mx-auto mb-3" />
              <p className="text-white/60">Generating podcast...</p>
              <p className="text-xs text-white/40 mt-1">This may take a minute...</p>
            </div>
          ) : (
            <div className="text-center">
              <Headphones className="w-8 h-8 text-white/30 mx-auto mb-3" />
              <p className="text-white/50">No podcast generated yet</p>
              <Button
                variant="secondary"
                size="sm"
                className="mt-3"
                onClick={() => handleModalityChange("podcast")}
              >
                Generate Podcast
              </Button>
            </div>
          )}
        </div>
      )}
    </motion.div>
  );

  // Render progress panel
  const renderProgressPanel = () => {
    if (!showProgress || !learningSummary) return null;

    return (
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        className="p-4 rounded-xl border border-white/10 bg-white/5"
      >
        <div className="flex items-center gap-2 mb-4">
          <BarChart2 className="w-4 h-4 text-purple-400" />
          <span className="font-medium text-white text-sm">Learning Progress</span>
        </div>

        {/* Progress Bar */}
        <div className="mb-4">
          <div className="flex justify-between text-xs text-white/50 mb-1">
            <span>Concepts Mastered</span>
            <span>{learningSummary.progress_percent.toFixed(0)}%</span>
          </div>
          <div className="h-2 bg-white/10 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${learningSummary.progress_percent}%` }}
              transition={{ duration: 0.5 }}
              className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
            />
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="p-2 rounded-lg bg-white/5">
            <div className="text-white/50">Total Concepts</div>
            <div className="font-medium text-white">{learningSummary.total_concepts}</div>
          </div>
          <div className="p-2 rounded-lg bg-white/5">
            <div className="text-white/50">Modality Switches</div>
            <div className="font-medium text-white">{learningSummary.modality_switches}</div>
          </div>
        </div>

        {/* Weak Concepts */}
        {learningSummary.weak_concepts.length > 0 && (
          <div className="mt-4">
            <div className="text-xs text-white/50 mb-2">Focus Areas</div>
            <div className="flex flex-wrap gap-1">
              {learningSummary.weak_concepts.slice(0, 3).map((concept) => (
                <span
                  key={concept}
                  className="px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 text-xs"
                >
                  {concept}
                </span>
              ))}
            </div>
          </div>
        )}
      </motion.div>
    );
  };

  return (
    <div className="w-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <ModalitySwitcher
            currentModality={currentModality}
            onModalityChange={handleModalityChange}
            isLoading={isMorphing || isGeneratingPodcast || isGeneratingDiagram}
            recommendation={recommendation}
          />
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              refetchState();
              refetchSummary();
            }}
          >
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Content Display */}
        <div className="lg:col-span-3">
          <AnimatePresence mode="wait">
            {currentModality === "text" && renderTextContent()}
            {currentModality === "diagram" && renderDiagramContent()}
            {currentModality === "podcast" && renderPodcastContent()}
          </AnimatePresence>
        </div>

        {/* Side Panel */}
        <div className="space-y-4">
          {renderProgressPanel()}

          {/* AI Recommendation */}
          {recommendation && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-4 rounded-xl border border-amber-500/20 bg-amber-500/10"
            >
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="w-4 h-4 text-amber-400" />
                <span className="font-medium text-amber-400 text-sm">AI Suggestion</span>
              </div>
              <p className="text-xs text-white/70">{recommendation.reason}</p>
              {recommendation.recommended_modality !== currentModality && (
                <Button
                  variant="secondary"
                  size="sm"
                  className="mt-3 w-full"
                  onClick={() => handleModalityChange(recommendation.recommended_modality)}
                >
                  Try {recommendation.recommended_modality}
                </Button>
              )}
            </motion.div>
          )}

          {/* Session Info */}
          <div className="p-4 rounded-xl border border-white/10 bg-white/5">
            <div className="flex items-center gap-2 mb-3">
              <Info className="w-4 h-4 text-white/50" />
              <span className="text-xs text-white/50">Session Info</span>
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-white/50">Content ID</span>
                <span className="text-white/70 font-mono">{contentId.slice(0, 8)}...</span>
              </div>
              <div className="flex justify-between">
                <span className="text-white/50">Current Modality</span>
                <span className="text-white/70 capitalize">{currentModality}</span>
              </div>
              {conceptualState && (
                <div className="flex justify-between">
                  <span className="text-white/50">Concepts Tracked</span>
                  <span className="text-white/70">
                    {Object.keys(conceptualState.concepts || {}).length}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MultimodalPlayer;
