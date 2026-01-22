"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Send, Sparkles, BrainCircuit, Zap } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { ragApi, adaptiveApi } from "@/lib/api";
import { Source, SourceViewer } from "@/components/dojo/SourceViewer";
import { MasteryGraph } from "@/components/dojo/MasteryGraph";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

type Message = {
    role: "user" | "assistant";
    content: string;
    sources?: Source[];
};

interface Metric {
    response_time_ms: number;
    correct: boolean;
    content_difficulty: number;
}

export default function DojoPage() {
    const [messages, setMessages] = useState<Message[]>([
        { role: "assistant", content: "Welcome back, Architect. Ready to push your cognitive limits? We're currently in **Interleaved Practice** mode to boost long-term retention." }
    ]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [cognitiveLoad, setCognitiveLoad] = useState({ score: 0.7, level: "Optimal (Flow)" });
    const [metricsHistory, setMetricsHistory] = useState<Metric[]>([]);
    const [startTime, setStartTime] = useState<number>(Date.now());
    const [selectedSource, setSelectedSource] = useState<any | null>(null);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const endTime = Date.now();
        const responseTimeMs = endTime - startTime;
        const userMsg = input;
        setMessages(prev => [...prev, { role: "user", content: userMsg }]);
        setInput("");
        setIsLoading(true);

        try {
            const data = await ragApi.chat(userMsg);
            setMessages(prev => [...prev, {
                role: "assistant",
                content: data.message,
                sources: data.citations?.map((c: any) => ({
                    title: c.module_title,
                    type: c.module_type,
                    page: c.page_number,
                    timestamp: c.timestamp_start ? `${Math.floor(c.timestamp_start / 60)}:${Math.floor(c.timestamp_start % 60).toString().padStart(2, '0')}` : undefined,
                    content: c.chunk_text
                })) || []
            }]);

            // Update metrics and HUD
            const newMetric = {
                response_time_ms: responseTimeMs,
                correct: true, // Assuming helpful interaction in chat
                content_difficulty: 5.0
            };
            const updatedMetrics = [...metricsHistory, newMetric].slice(-5);
            setMetricsHistory(updatedMetrics);

            const loadData = await adaptiveApi.estimateCognitiveLoad(updatedMetrics);
            setCognitiveLoad({
                score: loadData.score,
                level: loadData.level.charAt(0).toUpperCase() + loadData.level.slice(1)
            });

        } catch (error) {
            console.error("Chat Error", error);
            setMessages(prev => [...prev, { role: "assistant", content: "Display system offline. Unable to connect to neural core." }]);
        } finally {
            setIsLoading(false);
            setStartTime(Date.now()); // Reset timer for next interaction
        }
    };

    return (
        <div className="flex h-[calc(100vh-64px)] overflow-hidden">
            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col relative bg-gradient-to-b from-background to-background/50">
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    <AnimatePresence>
                        {messages.map((msg, i) => (
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                key={i}
                                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                            >
                                <div
                                    className={`max-w-[80%] rounded-2xl p-4 ${msg.role === "user"
                                        ? "bg-primary text-primary-foreground"
                                        : "bg-white/5 border border-white/10 backdrop-blur-md"
                                        }`}
                                >
                                    <div className="prose prose-invert prose-sm max-w-none">
                                        <ReactMarkdown
                                            remarkPlugins={[remarkGfm]}
                                            rehypePlugins={[rehypeKatex]}
                                        >
                                            {msg.content}
                                        </ReactMarkdown>
                                    </div>
                                    {msg.sources && msg.sources.length > 0 && (
                                        <div className="mt-2 pt-2 border-t border-white/10 text-[10px] text-muted-foreground flex flex-wrap gap-1">
                                            <span className="opacity-50 uppercase font-bold mr-1">Sources:</span>
                                            {msg.sources.map((source, idx) => (
                                                <button
                                                    key={idx}
                                                    onClick={() => setSelectedSource(source)}
                                                    className="bg-white/5 hover:bg-primary/20 hover:text-primary px-1.5 py-0.5 rounded border border-white/10 transition-colors flex items-center gap-1"
                                                >
                                                    {source.title}
                                                </button>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </motion.div>
                        ))}
                        {isLoading && (
                            <motion.div
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className="flex justify-start"
                            >
                                <div className="bg-white/5 border border-white/10 backdrop-blur-md rounded-2xl p-4 flex items-center gap-3">
                                    <Sparkles className="size-4 text-primary animate-pulse" />
                                    <span className="text-sm text-muted-foreground">Neural core processing...</span>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>

                {/* Input Area */}
                <div className="p-4 border-t border-white/10 bg-background/80 backdrop-blur-xl">
                    <div className="max-w-4xl mx-auto flex gap-2">
                        <input
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && handleSend()}
                            placeholder="Ask a question or solve a problem..."
                            className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all placeholder:text-muted-foreground"
                            disabled={isLoading}
                        />
                        <Button size="icon" variant="neon" onClick={handleSend} disabled={isLoading}>
                            <Send className="size-5" />
                        </Button>
                    </div>
                </div>
            </div>

            {/* Adaptive Scheduler Sidebar (The "HUD") */}
            <div className="w-80 border-l border-white/10 bg-black/20 backdrop-blur-xl p-4 hidden lg:flex flex-col gap-6">
                <div>
                    <h3 className="text-sm font-bold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-2">
                        <BrainCircuit className="size-4" /> Cognitive Load
                    </h3>
                    <div className="space-y-2">
                        <div className="flex justify-between text-xs">
                            <span>Current</span>
                            <span className={cognitiveLoad.score > 0.8 ? "text-amber-400" : "text-emerald-400"}>
                                {cognitiveLoad.level}
                            </span>
                        </div>
                        <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                            <motion.div
                                initial={{ width: "70%" }}
                                animate={{ width: `${cognitiveLoad.score * 100}%` }}
                                className={`h-full bg-gradient-to-r ${cognitiveLoad.score > 0.8
                                    ? "from-amber-500 to-red-500"
                                    : "from-emerald-500 to-cyan-500"
                                    }`}
                            />
                        </div>
                    </div>
                </div>

                <div>
                    <h3 className="text-sm font-bold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-2">
                        <Zap className="size-4" /> Active Strategy
                    </h3>
                    <div className="bg-gradient-to-br from-indigo-900/50 to-purple-900/50 border border-indigo-500/30 rounded-xl p-4 relative overflow-hidden group">
                        <div className="absolute inset-0 bg-indigo-500/10 opacity-0 group-hover:opacity-100 transition-opacity" />
                        <div className="relative z-10">
                            <div className="text-lg font-bold text-indigo-100">Interleaved Practice</div>
                            <p className="text-xs text-indigo-200/70 mt-1">
                                Mixing topics A, B, and C.
                            </p>
                            <div className="mt-3 text-[10px] bg-black/40 rounded px-2 py-1 inline-block border border-indigo-500/30 text-indigo-300">
                                Retain +76% more
                            </div>
                        </div>
                    </div>
                    <p className="text-xs text-muted-foreground mt-2 italic">
                        "It feels harder, but your brain is building stronger connections."
                    </p>
                </div>

                <div>
                    <h3 className="text-sm font-bold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-2">
                        <Sparkles className="size-4" /> Neural Map
                    </h3>
                    <MasteryGraph />
                </div>

                <div>
                    <h3 className="text-sm font-bold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-2">
                        🏆 Rewards & Milestones
                    </h3>
                    <div className="flex gap-2">
                        <div className="size-10 rounded-lg bg-yellow-500/20 border border-yellow-500/50 flex items-center justify-center text-yellow-500">
                            🏆
                        </div>
                        <div className="size-10 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center opacity-50">
                            ?
                        </div>
                        <div className="size-10 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center opacity-50">
                            ?
                        </div>
                    </div>
                </div>
            </div>

            <AnimatePresence>
                {selectedSource && (
                    <SourceViewer
                        source={selectedSource}
                        onClose={() => setSelectedSource(null)}
                    />
                )}
            </AnimatePresence>
        </div>
    );
}
