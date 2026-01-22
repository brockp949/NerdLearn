"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Volume2,
  VolumeX,
  Mic,
  User,
  GraduationCap,
  HelpCircle,
  Clock,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import type { PodcastEpisode, ScriptSegment, SpeakerRole } from "@/types/multimodal";

const speakerConfig: Record<SpeakerRole, {
  icon: typeof Mic;
  label: string;
  color: string;
  bgColor: string;
}> = {
  host: {
    icon: Mic,
    label: "Host",
    color: "text-blue-400",
    bgColor: "bg-blue-500/20",
  },
  guest: {
    icon: User,
    label: "Guest",
    color: "text-green-400",
    bgColor: "bg-green-500/20",
  },
  expert: {
    icon: GraduationCap,
    label: "Expert",
    color: "text-purple-400",
    bgColor: "bg-purple-500/20",
  },
  skeptic: {
    icon: HelpCircle,
    label: "Skeptic",
    color: "text-amber-400",
    bgColor: "bg-amber-500/20",
  },
};

interface PodcastPlayerProps {
  episode: PodcastEpisode;
  autoPlay?: boolean;
  onSegmentChange?: (segment: ScriptSegment, index: number) => void;
  onComplete?: () => void;
  className?: string;
}

export function PodcastPlayer({
  episode,
  autoPlay = false,
  onSegmentChange,
  onComplete,
  className = "",
}: PodcastPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const [currentSegmentIndex, setCurrentSegmentIndex] = useState(0);
  const [segmentProgress, setSegmentProgress] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [showTranscript, setShowTranscript] = useState(true);
  const [speechRate, setSpeechRate] = useState(1);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const speechRef = useRef<SpeechSynthesisUtterance | null>(null);
  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const currentSegment = episode.script_segments[currentSegmentIndex];
  const totalDuration = episode.total_duration_seconds;

  // Calculate overall progress
  const previousSegmentsDuration = episode.script_segments
    .slice(0, currentSegmentIndex)
    .reduce((acc, seg) => acc + seg.duration_seconds, 0);
  const overallProgress =
    ((previousSegmentsDuration + segmentProgress * currentSegment.duration_seconds) /
      totalDuration) *
    100;

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const getCurrentTime = () => {
    return previousSegmentsDuration + segmentProgress * currentSegment.duration_seconds;
  };

  // Use Web Speech API for text-to-speech (fallback when no audio URL)
  const speakSegment = useCallback(
    (segment: ScriptSegment) => {
      if (!episode.audio_url && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(segment.text);
        utterance.rate = speechRate;
        utterance.volume = isMuted ? 0 : volume;

        // Different voice for different speakers
        const voices = window.speechSynthesis.getVoices();
        const voiceIndex =
          segment.speaker === "host"
            ? 0
            : segment.speaker === "expert"
            ? 1
            : segment.speaker === "skeptic"
            ? 2
            : 3;
        if (voices[voiceIndex % voices.length]) {
          utterance.voice = voices[voiceIndex % voices.length];
        }

        utterance.onend = () => {
          if (currentSegmentIndex < episode.script_segments.length - 1) {
            setCurrentSegmentIndex((prev) => prev + 1);
          } else {
            setIsPlaying(false);
            onComplete?.();
          }
        };

        speechRef.current = utterance;
        window.speechSynthesis.speak(utterance);
      }
    },
    [episode.audio_url, episode.script_segments.length, speechRate, volume, isMuted, currentSegmentIndex, onComplete]
  );

  // Play/pause control
  useEffect(() => {
    if (isPlaying) {
      if (episode.audio_url && audioRef.current) {
        audioRef.current.play();
      } else {
        speakSegment(currentSegment);
      }

      // Progress simulation for TTS
      if (!episode.audio_url) {
        progressIntervalRef.current = setInterval(() => {
          setSegmentProgress((prev) => {
            const increment = 0.1 / (currentSegment.duration_seconds / speechRate);
            if (prev + increment >= 1) {
              return 1;
            }
            return prev + increment;
          });
        }, 100);
      }
    } else {
      if (episode.audio_url && audioRef.current) {
        audioRef.current.pause();
      } else {
        window.speechSynthesis?.pause();
      }

      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
    }

    return () => {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
    };
  }, [isPlaying, currentSegment, episode.audio_url, speakSegment, speechRate]);

  // Handle segment change
  useEffect(() => {
    setSegmentProgress(0);
    onSegmentChange?.(currentSegment, currentSegmentIndex);

    if (isPlaying && !episode.audio_url) {
      speakSegment(currentSegment);
    }
  }, [currentSegmentIndex]);

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handlePrevious = () => {
    if (currentSegmentIndex > 0) {
      setCurrentSegmentIndex((prev) => prev - 1);
    }
  };

  const handleNext = () => {
    if (currentSegmentIndex < episode.script_segments.length - 1) {
      setCurrentSegmentIndex((prev) => prev + 1);
    }
  };

  const handleSegmentClick = (index: number) => {
    setCurrentSegmentIndex(index);
    setSegmentProgress(0);
  };

  const toggleMute = () => {
    setIsMuted(!isMuted);
  };

  return (
    <div
      className={`rounded-xl border border-white/10 bg-black/40 overflow-hidden ${className}`}
    >
      {/* Audio element for actual audio files */}
      {episode.audio_url && (
        <audio
          ref={audioRef}
          src={episode.audio_url}
          onEnded={onComplete}
        />
      )}

      {/* Header */}
      <div className="px-4 py-3 border-b border-white/10 bg-gradient-to-r from-green-500/10 to-transparent">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium text-white">{episode.title}</h3>
            <p className="text-xs text-white/50 mt-0.5">
              {episode.script_segments.length} segments • {formatTime(totalDuration)}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-white/40 px-2 py-1 rounded bg-white/10">
              {episode.audio_url ? "Audio" : "TTS"}
            </span>
          </div>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="px-4 py-2 border-b border-white/5">
        <div className="flex items-center gap-3">
          <span className="text-xs text-white/50 w-10">
            {formatTime(getCurrentTime())}
          </span>
          <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-green-500 to-emerald-500 rounded-full"
              style={{ width: `${overallProgress}%` }}
              layout
            />
          </div>
          <span className="text-xs text-white/50 w-10 text-right">
            {formatTime(totalDuration)}
          </span>
        </div>
      </div>

      {/* Current Segment Display */}
      <div className="px-4 py-4">
        <div className="flex items-start gap-3">
          {(() => {
            const config = speakerConfig[currentSegment.speaker];
            const Icon = config.icon;
            return (
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${config.bgColor}`}
              >
                <Icon className={`w-5 h-5 ${config.color}`} />
              </div>
            );
          })()}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span
                className={`text-sm font-medium ${
                  speakerConfig[currentSegment.speaker].color
                }`}
              >
                {speakerConfig[currentSegment.speaker].label}
              </span>
              <span className="text-xs text-white/30">
                {currentSegment.emotion}
              </span>
            </div>
            <p className="text-sm text-white/80 leading-relaxed">
              {currentSegment.text}
            </p>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="px-4 py-3 border-t border-white/10 bg-white/5">
        <div className="flex items-center justify-between">
          {/* Playback Controls */}
          <div className="flex items-center gap-2">
            <button
              onClick={handlePrevious}
              disabled={currentSegmentIndex === 0}
              className="p-2 rounded-lg hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <SkipBack className="w-4 h-4 text-white/70" />
            </button>

            <button
              onClick={handlePlayPause}
              className="p-3 rounded-full bg-gradient-to-br from-green-500 to-emerald-600 hover:from-green-400 hover:to-emerald-500 transition-all shadow-lg"
            >
              {isPlaying ? (
                <Pause className="w-5 h-5 text-white" />
              ) : (
                <Play className="w-5 h-5 text-white ml-0.5" />
              )}
            </button>

            <button
              onClick={handleNext}
              disabled={currentSegmentIndex === episode.script_segments.length - 1}
              className="p-2 rounded-lg hover:bg-white/10 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <SkipForward className="w-4 h-4 text-white/70" />
            </button>
          </div>

          {/* Right Controls */}
          <div className="flex items-center gap-3">
            {/* Speed Control */}
            <select
              value={speechRate}
              onChange={(e) => setSpeechRate(parseFloat(e.target.value))}
              className="text-xs bg-white/10 border border-white/10 rounded px-2 py-1 text-white/70"
            >
              <option value="0.75">0.75x</option>
              <option value="1">1x</option>
              <option value="1.25">1.25x</option>
              <option value="1.5">1.5x</option>
              <option value="2">2x</option>
            </select>

            {/* Volume */}
            <button
              onClick={toggleMute}
              className="p-2 rounded-lg hover:bg-white/10 transition-colors"
            >
              {isMuted ? (
                <VolumeX className="w-4 h-4 text-white/70" />
              ) : (
                <Volume2 className="w-4 h-4 text-white/70" />
              )}
            </button>

            {/* Transcript Toggle */}
            <button
              onClick={() => setShowTranscript(!showTranscript)}
              className="flex items-center gap-1 px-2 py-1 rounded-lg hover:bg-white/10 transition-colors"
            >
              <span className="text-xs text-white/70">Transcript</span>
              {showTranscript ? (
                <ChevronUp className="w-3 h-3 text-white/50" />
              ) : (
                <ChevronDown className="w-3 h-3 text-white/50" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Full Transcript */}
      <AnimatePresence>
        {showTranscript && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-t border-white/10 overflow-hidden"
          >
            <div className="max-h-60 overflow-y-auto">
              {episode.script_segments.map((segment, index) => {
                const config = speakerConfig[segment.speaker];
                const Icon = config.icon;
                const isActive = index === currentSegmentIndex;

                return (
                  <button
                    key={index}
                    onClick={() => handleSegmentClick(index)}
                    className={`
                      w-full flex items-start gap-3 px-4 py-3 text-left
                      hover:bg-white/5 transition-colors
                      ${isActive ? "bg-white/10" : ""}
                    `}
                  >
                    <div
                      className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 ${
                        isActive ? config.bgColor : "bg-white/5"
                      }`}
                    >
                      <Icon
                        className={`w-3 h-3 ${
                          isActive ? config.color : "text-white/40"
                        }`}
                      />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-0.5">
                        <span
                          className={`text-xs font-medium ${
                            isActive ? config.color : "text-white/50"
                          }`}
                        >
                          {config.label}
                        </span>
                        <span className="text-xs text-white/20">
                          {formatTime(segment.duration_seconds)}
                        </span>
                      </div>
                      <p
                        className={`text-xs leading-relaxed line-clamp-2 ${
                          isActive ? "text-white/80" : "text-white/50"
                        }`}
                      >
                        {segment.text}
                      </p>
                    </div>
                  </button>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default PodcastPlayer;
