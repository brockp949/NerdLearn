"use client";

import { useEffect, useState, useRef } from "react";
// import { ForceGraph3D } from "react-force-graph-3d"; // Import dynamically to avoid SSR issues
import dynamic from 'next/dynamic';
import { graphApi } from "@/lib/api";
import { Loader2 } from "lucide-react";

const ForceGraph3D = dynamic(() => import('react-force-graph-3d'), { ssr: false });

export default function BrainPage() {
    const [graphData, setGraphData] = useState({ nodes: [], links: [] });
    const [loading, setLoading] = useState(true);
    const graphRef = useRef<any>(null);

    useEffect(() => {
        const fetchGraph = async () => {
            try {
                const data = await graphApi.getGraph();
                // Ensure data is in the format expected by ForceGraph3D (nodes, links)
                // Adjust if backend returns different keys. Assuming standard { nodes: [], links: [] }
                setGraphData(data);
            } catch (error) {
                console.error("Failed to fetch graph", error);
                // Fallback or error state
            } finally {
                setLoading(false);
            }
        };

        fetchGraph();
    }, []);

    if (loading) {
        return (
            <div className="h-[calc(100vh-64px)] flex items-center justify-center">
                <Loader2 className="size-10 text-primary animate-spin" />
                <span className="ml-3 text-lg font-mono text-muted-foreground">Loading Neural Map...</span>
            </div>
        );
    }

    return (
        <div className="h-[calc(100vh-64px)] relative bg-black">
            <div className="absolute top-4 left-4 z-10 glass p-4 rounded-xl max-w-xs">
                <h1 className="text-xl font-bold text-white mb-1">Knowledge Structure</h1>
                <p className="text-xs text-white/70">
                    Interactive visualization of your learned concepts.
                    Nodes represent concepts, links represent dependencies.
                </p>
                <div className="mt-2 flex gap-4 text-xs font-mono text-primary">
                    <span>Nodes: {graphData.nodes.length}</span>
                    <span>Links: {graphData.links.length}</span>
                </div>
            </div>

            <ForceGraph3D
                ref={graphRef}
                graphData={graphData}
                nodeLabel="id"
                nodeColor={() => "#3b82f6"} // Primary blue
                linkColor={() => "rgba(255,255,255,0.2)"}
                backgroundColor="#000000"
                linkOpacity={0.3}
                nodeRelSize={6}
                onNodeClick={(node: any) => {
                    // Ideally navigate to a concept detail page or invoke the Dojo
                    const distance = 40;
                    const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
                    graphRef.current?.cameraPosition(
                        { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }, // new position
                        node, // lookAt ({ x, y, z })
                        3000  // ms transition duration
                    );
                }}

            />
        </div>
    );
}
