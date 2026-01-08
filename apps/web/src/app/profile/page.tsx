"use client";

import { motion } from "framer-motion";

export default function ProfilePage() {
    // Mock data for heatmap
    const weeks = 52;
    const days = 7;
    const heatmapData = Array.from({ length: weeks * days }, () => Math.random());

    return (
        <div className="container mx-auto p-6 space-y-8">
            <header>
                <h1 className="text-4xl font-bold bg-gradient-to-r from-primary to-purple-400 bg-clip-text text-transparent">
                    Neural Architecture
                </h1>
                <p className="text-muted-foreground mt-2">
                    Visualizing your cognitive restructuring over time.
                </p>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Stats Card */}
                <div className="md:col-span-1 space-y-6">
                    <div className="glass-card p-6 rounded-2xl">
                        <h2 className="text-xl font-bold mb-4">Architect Status</h2>
                        <div className="space-y-4">
                            <div>
                                <div className="text-sm text-muted-foreground">Level</div>
                                <div className="text-3xl font-mono">04</div>
                            </div>
                            <div>
                                <div className="text-sm text-muted-foreground">Synaptic Density (XP)</div>
                                <div className="text-3xl font-mono">12,450</div>
                            </div>
                            <div>
                                <div className="text-sm text-muted-foreground">Current Streak</div>
                                <div className="text-3xl font-mono text-emerald-400">12 Days</div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Heatmap */}
                <div className="md:col-span-2 glass-card p-6 rounded-2xl">
                    <h2 className="text-xl font-bold mb-4">Consistency Matrix</h2>
                    <div className="flex flex-wrap gap-1 h-[200px] overflow-hidden content-start">
                        {heatmapData.map((value, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ delay: i * 0.001 }}
                                className="size-3 rounded-sm"
                                style={{
                                    backgroundColor:
                                        value > 0.8
                                            ? "rgba(16, 185, 129, 0.9)" // high
                                            : value > 0.5
                                                ? "rgba(16, 185, 129, 0.5)" // medium
                                                : value > 0.2
                                                    ? "rgba(16, 185, 129, 0.2)" // low
                                                    : "rgba(255, 255, 255, 0.05)", // none
                                }}
                            />
                        ))}
                    </div>
                    <p className="text-xs text-muted-foreground mt-4">
                        Each cell represents a day of cognitive training. Darker cells indicate higher intensity.
                    </p>
                </div>
            </div>
        </div>
    );
}
