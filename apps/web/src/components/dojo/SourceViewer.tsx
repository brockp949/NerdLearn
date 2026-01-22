"use client";

import { motion, AnimatePresence } from "framer-motion";
import { X, FileText, Play, ExternalLink } from "lucide-react";

export interface Source {
    title: string;
    type: "pdf" | "video" | "unknown";
    page?: number;
    timestamp?: string;
    content?: string;
}

interface SourceViewerProps {
    source: Source | null;
    onClose: () => void;
}

export function SourceViewer({ source, onClose }: SourceViewerProps) {
    if (!source) return null;

    return (
        <motion.div
            initial={{ x: 400, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 400, opacity: 0 }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="w-96 border-l border-white/10 bg-white/5 backdrop-blur-md flex flex-col z-50 h-full overflow-hidden shadow-2xl"
        >
            <div className="p-4 border-b border-white/10 flex items-center justify-between bg-white/5">
                <div className="flex items-center gap-2">
                    {source.type === "video" ? <Play className="size-4 text-primary" /> : <FileText className="size-4 text-primary" />}
                    <h3 className="text-sm font-bold truncate max-w-[200px]">{source.title}</h3>
                </div>
                <button onClick={onClose} className="p-1 hover:bg-white/10 rounded-lg transition-colors">
                    <X className="size-4" />
                </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6 space-y-6">
                <div className="space-y-2">
                    <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-muted-foreground font-bold">
                        <span>Reference Details</span>
                        {source.type !== "unknown" && (
                            <span className="bg-primary/20 text-primary px-2 py-0.5 rounded border border-primary/30">
                                {source.type === "pdf" ? `Page ${source.page || "N/A"}` : `Time ${source.timestamp || "N/A"}`}
                            </span>
                        )}
                    </div>
                </div>

                <div className="aspect-[3/4] rounded-xl border border-white/10 bg-white/5 relative overflow-hidden group">
                    <div className="absolute inset-0 flex flex-col items-center justify-center p-8 text-center space-y-4">
                        <div className="size-16 rounded-2xl bg-primary/10 flex items-center justify-center text-primary group-hover:scale-110 transition-transform">
                            {source.type === "video" ? <Play className="size-8" /> : <FileText className="size-8" />}
                        </div>
                        <div>
                            <p className="text-sm font-medium">Neural core simulation</p>
                            <p className="text-xs text-muted-foreground mt-1">
                                Full {source.type} rendering engine initializing...
                            </p>
                        </div>
                        <button className="flex items-center gap-2 text-xs bg-white/10 hover:bg-white/20 px-4 py-2 rounded-lg transition-colors border border-white/10">
                            <ExternalLink className="size-3" /> Open in New Hub
                        </button>
                    </div>
                    {/* Visual noise/gradient overlay for "premium" feel */}
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent pointer-events-none" />
                </div>

                <div className="space-y-4">
                    <h4 className="text-xs font-bold uppercase tracking-widest text-muted-foreground">Contextual Excerpt</h4>
                    <div className="p-4 rounded-xl bg-white/5 border border-white/10 text-sm leading-relaxed text-indigo-100/80 italic">
                        "{source.content || "Loading neural core context..."}"
                    </div>
                </div>
            </div>
        </motion.div>
    );
}
