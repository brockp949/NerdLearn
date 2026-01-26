"use client"

import React, { useEffect, useState } from "react"
import { useTelemetryContext } from "@/components/providers/telemetry-provider"
import { Intervention } from "@/lib/telemetry"
import { motion, AnimatePresence } from "framer-motion"
import { X, Sparkles, MessageSquare } from "lucide-react"
import { cn } from "@/lib/utils"

export function InterventionToast() {
    const { intervention } = useTelemetryContext()
    const [isVisible, setIsVisible] = useState(false)
    const [currentIntervention, setCurrentIntervention] = useState<Intervention | null>(null)

    useEffect(() => {
        if (intervention) {
            setCurrentIntervention(intervention)
            setIsVisible(true)

            // Auto-dismiss after 10 seconds if it's just a notification
            if (intervention.type === 'notification') {
                const timer = setTimeout(() => setIsVisible(false), 10000)
                return () => clearTimeout(timer)
            }
        }
    }, [intervention])

    if (!isVisible || !currentIntervention) return null

    return (
        <AnimatePresence>
            {isVisible && (
                <motion.div
                    initial={{ opacity: 0, y: 50, x: "-50%" }}
                    animate={{ opacity: 1, y: 0, x: "-50%" }}
                    exit={{ opacity: 0, y: 20, x: "-50%" }}
                    className="fixed bottom-8 left-1/2 z-50 w-full max-w-md -translate-x-1/2 px-4"
                >
                    <div className={cn(
                        "flex items-start gap-4 rounded-lg border p-4 shadow-lg backdrop-blur-md",
                        currentIntervention.type === 'prompt'
                            ? "bg-primary/10 border-primary/20 text-foreground"
                            : "bg-background/80 border-border"
                    )}>
                        <div className={cn(
                            "rounded-full p-2",
                            currentIntervention.type === 'prompt' ? "bg-primary/20 text-primary" : "bg-muted text-muted-foreground"
                        )}>
                            {currentIntervention.type === 'prompt' ? <Sparkles size={18} /> : <MessageSquare size={18} />}
                        </div>

                        <div className="flex-1 pt-1">
                            <h4 className="mb-1 font-semibold leading-none tracking-tight">
                                AI Suggestion
                            </h4>
                            <p className="text-sm text-muted-foreground">
                                {currentIntervention.message}
                            </p>

                            {currentIntervention.action && (
                                <button
                                    onClick={() => setIsVisible(false)}
                                    className="mt-3 text-xs font-medium text-primary hover:underline"
                                >
                                    {currentIntervention.action} →
                                </button>
                            )}
                        </div>

                        <button
                            onClick={() => setIsVisible(false)}
                            className="text-muted-foreground hover:text-foreground"
                        >
                            <X size={16} />
                            <span className="sr-only">Close</span>
                        </button>
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    )
}
