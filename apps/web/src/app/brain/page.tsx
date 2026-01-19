"use client";

import { useRef, useMemo, useEffect } from "react";
import dynamic from 'next/dynamic';
import * as THREE from 'three';
import { Loader2 } from "lucide-react";
import { motion } from "framer-motion";
import { useGraphData } from "@/hooks/use-graph-data";
import { GraphNode } from "@/types/graph";

const ForceGraph3D = dynamic(() => import('react-force-graph-3d'), { ssr: false });

export default function BrainPage() {
    const { data: graphData, loading } = useGraphData();
    const graphRef = useRef<any>(null);

    const nodeThreeObject = useMemo(() => (node: any) => {
        const n = node as GraphNode;
        const color = n.type === 'concept' ? "#3b82f6" : "#ffffff";

        // Use a standard mesh with emissive properties for "mastery pulsing"
        const geometry = new THREE.SphereGeometry(6);
        const material = new THREE.MeshStandardMaterial({
            color: color,
            transparent: true,
            opacity: 0.9,
            emissive: color,
            emissiveIntensity: 0.8
        });

        const mesh = new THREE.Mesh(geometry, material);

        // Add a subtle bloom-like aura if mastery is high (placeholder logic)
        if (n.difficulty !== undefined && n.difficulty < 3) {
            const auraGeom = new THREE.SphereGeometry(8);
            const auraMat = new THREE.MeshBasicMaterial({
                color: color,
                transparent: true,
                opacity: 0.2
            });
            const aura = new THREE.Mesh(auraGeom, auraMat);
            mesh.add(aura);
        }

        return mesh;
    }, []);

    // Animation hook for pulsing
    useEffect(() => {
        if (!graphRef.current) return;

        let frameId: number;
        const animate = () => {
            const time = Date.now() * 0.002;
            graphRef.current.scene().traverse((obj: any) => {
                if (obj.isMesh && obj.material && obj.material.emissiveIntensity !== undefined) {
                    obj.material.emissiveIntensity = 0.5 + Math.sin(time + obj.id) * 0.5;
                }
            });
            frameId = requestAnimationFrame(animate);
        };

        frameId = requestAnimationFrame(animate);
        return () => cancelAnimationFrame(frameId);
    }, [loading]);

    if (loading) {
        return (
            <div className="h-[calc(100vh-64px)] flex items-center justify-center bg-black">
                <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="p-8 rounded-2xl bg-white/5 backdrop-blur-lg border border-white/10 flex flex-col items-center gap-4"
                >
                    <Loader2 className="size-10 text-primary animate-spin" />
                    <span className="text-lg font-mono text-white/80 animate-pulse">Initializing Neural Map...</span>
                </motion.div>
            </div>
        );
    }

    return (
        <div className="h-[calc(100vh-64px)] relative bg-black overflow-hidden">
            <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="absolute top-4 left-4 z-10 p-4 rounded-xl max-w-xs bg-white/5 backdrop-blur-md border border-white/10 shadow-xl"
            >
                <h1 className="text-xl font-bold text-white mb-1 tracking-tight">Knowledge Structure</h1>
                <p className="text-xs text-white/70 leading-relaxed">
                    Interactive visualization of your learned concepts.
                    <br />
                    <span className="text-primary/80">Nodes</span>: Concepts | <span className="text-white/50">Links</span>: Dependencies
                </p>
                <div className="mt-4 flex gap-4 text-xs font-mono text-primary border-t border-white/5 pt-3">
                    <div className="flex flex-col">
                        <span className="text-white/40 text-[10px] uppercase">Nodes</span>
                        <span className="text-lg font-bold">{graphData.nodes.length}</span>
                    </div>
                    <div className="flex flex-col">
                        <span className="text-white/40 text-[10px] uppercase">Links</span>
                        <span className="text-lg font-bold">{graphData.links.length}</span>
                    </div>
                </div>
            </motion.div>

            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 1 }}
                className="w-full h-full"
            >
                <ForceGraph3D
                    ref={graphRef}
                    graphData={graphData}
                    nodeLabel="label"
                    nodeThreeObject={nodeThreeObject}
                    linkColor={() => "rgba(255,255,255,0.15)"}
                    backgroundColor="#000000"
                    linkOpacity={0.3}
                    nodeRelSize={6}
                    onNodeClick={(node: any) => {
                        const distance = 40;
                        const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
                        graphRef.current?.cameraPosition(
                            { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
                            node,
                            3000
                        );
                    }}
                />
            </motion.div>
        </div>
    );
}
