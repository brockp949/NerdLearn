"use client";

import { useEffect, useCallback } from "react";
import ReactFlow, {
    Controls,
    Background,
    useNodesState,
    useEdgesState,
    Node,
    Edge,
    MarkerType,
    ConnectionLineType
} from "reactflow";
import "reactflow/dist/style.css";
import ELK from "elkjs/lib/elk.bundled";
import { useGraphData } from "@/hooks/use-graph-data";
import { GraphNode, GraphLink } from "@/types/graph";
import { Loader2 } from "lucide-react";

const elk = new ELK();

const layoutOptions = {
    'elk.algorithm': 'layered',
    'elk.direction': 'RIGHT',
    'elk.layered.spacing.nodeNodeBetweenLayers': '100',
    'elk.spacing.nodeNode': '80',
    'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF',
};

interface MasteryGraphProps {
    courseId?: number;
}

export function MasteryGraph({ courseId = 1 }: MasteryGraphProps) {
    const { data: graphData, loading } = useGraphData();
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);

    const getLayoutedElements = useCallback(async (nodes: Node[], edges: Edge[]) => {
        const graph = {
            id: "root",
            layoutOptions: layoutOptions,
            children: nodes.map((node) => ({
                ...node,
                width: 150,
                height: 50,
            })),
            edges: edges.map((edge) => ({
                ...edge,
                sources: [edge.source],
                targets: [edge.target],
            })),
        };

        try {
            const layoutedGraph = await elk.layout(graph as any);
            return {
                nodes: layoutedGraph.children?.map((node: any) => ({
                    ...node,
                    position: { x: node.x, y: node.y },
                })) || [],
                edges: edges,
            };
        } catch (e) {
            console.error("ELK Layout Error", e);
            return { nodes, edges };
        }
    }, []);

    useEffect(() => {
        if (!graphData || graphData.nodes.length === 0) return;

        const initialNodes: Node[] = graphData.nodes.map((node) => ({
            id: node.id,
            data: { label: node.label },
            position: { x: 0, y: 0 },
            style: {
                background: '#1e293b',
                color: '#fff',
                border: '1px solid #3b82f6',
                borderRadius: '8px',
                padding: '10px',
                fontSize: '12px',
                width: 150,
            },
        }));

        const initialEdges: Edge[] = graphData.links.map((link, i) => ({
            id: `e${i}`,
            source: typeof link.source === 'object' ? link.source.id : link.source,
            target: typeof link.target === 'object' ? link.target.id : link.target,
            type: 'smoothstep',
            animated: true,
            style: { stroke: '#64748b' },
            markerEnd: {
                type: MarkerType.ArrowClosed,
                color: '#64748b',
            },
        }));

        getLayoutedElements(initialNodes, initialEdges).then(({ nodes: layoutedNodes, edges: layoutedEdges }) => {
            setNodes(layoutedNodes);
            setEdges(layoutedEdges);
        });

    }, [graphData, getLayoutedElements, setNodes, setEdges]);

    if (loading) {
        return (
            <div className="h-48 w-full rounded-xl border border-white/10 bg-black/40 flex items-center justify-center">
                <Loader2 className="animate-spin text-muted-foreground" />
            </div>
        );
    }

    return (
        <div className="h-64 w-full rounded-xl border border-white/10 bg-black/40 overflow-hidden relative group">
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                fitView
                attributionPosition="bottom-right"
            >
                <Background color="#aaa" gap={16} size={1} style={{ opacity: 0.1 }} />
                <Controls showInteractive={false} className="opacity-0 group-hover:opacity-100 transition-opacity" />
            </ReactFlow>

            <div className="absolute bottom-2 right-2 flex gap-2 pointer-events-none">
                <div className="flex items-center gap-1 backdrop-blur-sm bg-black/20 p-1 rounded">
                    <div className="size-2 rounded-full bg-blue-500" />
                    <span className="text-[8px] text-muted-foreground uppercase font-bold">Concept</span>
                </div>
            </div>
        </div>
    );
}
