"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Send, Sparkles, BrainCircuit, Zap } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { ragApi } from "@/lib/api";

type Message = {
    role: "user" | "assistant";
    content: string;
    sources?: string[];
};

export default function DojoPage() {
    const [messages, setMessages] = useState<Message[]>([
        { role: "assistant", content: "Welcome back, Architect. Ready to push your cognitive limits? We're currently in **Interleaved Practice** mode to boost long-term retention." }
    ]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMsg = input;
        setMessages(prev => [...prev, { role: "user", content: userMsg }]);
        setInput("");
        setIsLoading(true);

        try {
            const data = await ragApi.chat(userMsg);
            setMessages(prev => [...prev, {
                role: "assistant",
                content: data.message, // Backend specific: 'message' not 'response'
                sources: data.citations?.map((c: any) => c.text || "Source") || [] // Backend specific: 'citations'
            }]);
        } catch (error) {
            console.error("Chat Error", error);
            setMessages(prev => [...prev, { role: "assistant", content: "Display system offline. Unable to connect to neural core." }]);
        } finally {
            setIsLoading(false);
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
                                    {/* Basic markdown rendering can be added here */}
                                    {msg.content}
                                    {msg.sources && msg.sources.length > 0 && (
                                        <div className="mt-2 pt-2 border-t border-white/10 text-xs text-muted-foreground">
                                            Sources: {msg.sources.join(", ")}
                                        </div>
                                    )}
                                </div>
                            </motion.div>
                        ))}
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
                            <span className="text-emerald-400">Optimal (Flow)</span>
                        </div>
                        <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                            <div className="h-full w-[70%] bg-gradient-to-r from-emerald-500 to-cyan-500" />
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
                        <Sparkles className="size-4" /> Next Rewards
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
        </div>
    );
}
