import React, { useCallback, useEffect, useMemo } from 'react';
import ReactFlow, {
    Background,
    Controls,
    MiniMap,
    useNodesState,
    useEdgesState,
    Panel,
    Node,
    Edge,
    Position,
    ConnectionLineType,
    MarkerType
} from 'reactflow';
import ELK, { ElkNode } from 'elkjs/lib/elk.bundled';
import 'reactflow/dist/style.css';

// Initial ELK setup
const elk = new ELK();

// Types for our diagram data
export interface DiagramNode {
    id: string;
    label: string;
    type?: string;
    data?: any;
}

export interface DiagramEdge {
    id: string;
    source: string;
    target: string;
    label?: string;
}

interface DiagramGeneratorProps {
    initialNodes: DiagramNode[];
    initialEdges: DiagramEdge[];
    onNodeClick?: (event: React.MouseEvent, node: Node) => void;
    direction?: 'DOWN' | 'RIGHT';
    className?: string;
}

/**
 * DiagramGenerator Component
 * 
 * Research alignment:
 * - Generative Diagrams: Visualizes concept relationships
 * - Interactive Canvas: Zoom, pan, click
 * - Auto-layout: Uses Elkjs for force-directed placement
 */
export const DiagramGenerator: React.FC<DiagramGeneratorProps> = ({
    initialNodes,
    initialEdges,
    onNodeClick,
    direction = 'DOWN',
    className = "h-[500px] w-full border rounded-lg bg-white"
}) => {
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);

    // Transform simple props into ReactFlow format
    const getLayoutedElements = useCallback(async (
        simpleNodes: DiagramNode[],
        simpleEdges: DiagramEdge[],
        dir: 'DOWN' | 'RIGHT'
    ) => {
        const isHorizontal = dir === 'RIGHT';

        // Construct ELK graph
        const graph: ElkNode = {
            id: 'root',
            layoutOptions: {
                'elk.algorithm': 'layered',
                'elk.direction': dir,
                'elk.spacing.nodeNode': '60',
                'elk.layered.spacing.nodeNodeBetweenLayers': '100',
                'elk.padding': '[top=50,left=50,bottom=50,right=50]'
            },
            children: simpleNodes.map((n) => ({
                id: n.id,
                width: 180,
                height: 50,
                // Add layout constraints if needed
            })),
            edges: simpleEdges.map((e) => ({
                id: e.id,
                sources: [e.source],
                targets: [e.target]
            }))
        };

        // calculate layout
        try {
            const layoutedGraph = await elk.layout(graph);

            const layoutedNodes: Node[] = (layoutedGraph.children || []).map((node) => ({
                id: node.id,
                position: { x: node.x || 0, y: node.y || 0 },
                data: {
                    label: simpleNodes.find(n => n.id === node.id)?.label || node.id,
                    ...simpleNodes.find(n => n.id === node.id)?.data
                },
                type: simpleNodes.find(n => n.id === node.id)?.type || 'default',
                style: {
                    width: 180,
                    background: '#ffffff',
                    border: '1px solid #e1e4e8',
                    borderRadius: '8px',
                    padding: '10px',
                    fontSize: '14px',
                    textAlign: 'center',
                    boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                },
                sourcePosition: isHorizontal ? Position.Right : Position.Bottom,
                targetPosition: isHorizontal ? Position.Left : Position.Top,
            }));

            const layoutedEdges: Edge[] = simpleEdges.map((e) => ({
                id: e.id,
                source: e.source,
                target: e.target,
                type: 'smoothstep',
                label: e.label,
                markerEnd: { type: MarkerType.ArrowClosed },
                animated: true,
                style: { stroke: '#64748b', strokeWidth: 2 }
            }));

            return { nodes: layoutedNodes, edges: layoutedEdges };
        } catch (err) {
            console.error('ELK Layout Error:', err);
            return { nodes: [], edges: [] };
        }
    }, []);

    // Effect to update layout when props change
    useEffect(() => {
        const loadLayout = async () => {
            const { nodes: layoutedNodes, edges: layoutedEdges } = await getLayoutedElements(
                initialNodes,
                initialEdges,
                direction
            );
            setNodes(layoutedNodes);
            setEdges(layoutedEdges);
        };

        loadLayout();
    }, [initialNodes, initialEdges, direction, getLayoutedElements, setNodes, setEdges]);

    return (
        <div className={className}>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onNodeClick={onNodeClick}
                fitView
                attributionPosition="bottom-right"
            >
                <Controls />
                <MiniMap zoomable pannable />
                <Background gap={12} size={1} />
                <Panel position="top-right">
                    <div className="bg-white p-2 rounded shadow text-xs text-gray-500">
                        Concepts: {nodes.length} | Relationships: {edges.length}
                    </div>
                </Panel>
            </ReactFlow>
        </div>
    );
};
