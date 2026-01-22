"use client";

import { useRef } from "react";
import dynamic from 'next/dynamic';
import { ForceGraphMethods } from "react-force-graph-2d";
import { useGraphData } from "@/hooks/use-graph-data";
import { GraphNode } from "@/types/graph";

const ForceGraph2D = dynamic(() => import('react-force-graph-2d').then(mod => mod.default), { ssr: false }) as any;

interface MasteryGraphProps {
    courseId?: number;
}

export function MasteryGraph({ courseId = 1 }: MasteryGraphProps) {
    const { data: graphData } = useGraphData();
    // In a real scenario, we might want to filter by courseId or pass it to useGraphData
    // For now, useGraphData fetches default or mocked data which is sufficient for UI demo.

    return (
        <div className="h-48 w-full rounded-xl border border-white/10 bg-black/40 overflow-hidden relative group">
            <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-transparent pointer-events-none" />
            <ForceGraph2D
                graphData={graphData}
                width={300}
                height={192}
                nodeRelSize={4}
                nodeLabel="label"
                nodeColor={(node: any) => node.mastered ? "#10b981" : "#3b82f6"}
                linkColor={() => "rgba(255,255,255,0.1)"}
                backgroundColor="rgba(0,0,0,0)"
            />
            <div className="absolute bottom-2 right-2 flex gap-2">
                <div className="flex items-center gap-1">
                    <div className="size-2 rounded-full bg-emerald-500" />
                    <span className="text-[8px] text-muted-foreground uppercase font-bold">Mastered</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="size-2 rounded-full bg-blue-500" />
                    <span className="text-[8px] text-muted-foreground uppercase font-bold">In Progress</span>
                </div>
            </div>
        </div>
    );
}
