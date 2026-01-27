'use client'

import { useState, useCallback, useEffect } from 'react'
import ReactFlow, {
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  Node,
  Edge,
  MarkerType,
} from 'reactflow'
import 'reactflow/dist/style.css'
import ELK from 'elkjs/lib/elk.bundled'
import { Loader2 } from 'lucide-react'

const elk = new ELK()

// Define layout options for ELK
const layoutOptions = {
  'elk.algorithm': 'layered',
  'elk.direction': 'DOWN',
  'elk.layered.spacing.nodeNodeBetweenLayers': '100',
  'elk.spacing.nodeNode': '80',
  'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF',
}

import { GraphNode, GraphLink as GraphEdge, GraphData } from '@/types/graph'

export interface KnowledgeGraphViewProps {
  data: GraphData
  onNodeClick?: (node: GraphNode) => void
  height?: number
}

export function KnowledgeGraphView({ data, onNodeClick, height = 600 }: KnowledgeGraphViewProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const [loading, setLoading] = useState(true)

  const getLayoutedElements = useCallback(async (graphNodes: any[], graphEdges: any[]) => {
    const graph = {
      id: 'root',
      layoutOptions: layoutOptions,
      children: graphNodes.map((node) => ({
        ...node,
        width: 180,
        height: 60,
      })),
      edges: graphEdges.map((edge) => ({
        ...edge,
        sources: [edge.source],
        targets: [edge.target],
      })),
    }

    try {
      const layoutedGraph = await elk.layout(graph as any)
      return {
        nodes: layoutedGraph.children?.map((node: any) => ({
          ...node,
          position: { x: node.x, y: node.y },
        })) || [],
        edges: graphEdges,
      }
    } catch (e) {
      console.error('ELK Layout Error', e)
      return { nodes: graphNodes, edges: graphEdges }
    }
  }, [])

  useEffect(() => {
    if (!data || data.nodes.length === 0) {
      setLoading(false);
      return;
    }

    const initialNodes: Node[] = data.nodes.map((node) => {
      // Determine color based on mastery and lock status (Research-aligned)
      const mastery = node.mastery || 0;
      const isLocked = node.isLocked ?? false;
      
      let borderColor = '#94a3b8'; // default slate-400
      let bgGradient = 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)'; // default dark
      let textColor = '#f8fafc';

      if (isLocked) {
        borderColor = '#475569'; // slate-600
        bgGradient = 'linear-gradient(135deg, #334155 0%, #1e293b 100%)';
        textColor = '#94a3b8';
      } else if (mastery >= 0.95) {
        borderColor = '#10b981'; // emerald-500
        bgGradient = 'linear-gradient(135deg, #064e3b 0%, #065f46 100%)';
      } else if (mastery >= 0.50) {
        borderColor = '#f97316'; // orange-500
        bgGradient = 'linear-gradient(135deg, #7c2d12 0%, #9a3412 100%)';
      } else {
        // Unlocked (0-50%)
        borderColor = '#eab308'; // yellow-500
        bgGradient = 'linear-gradient(135deg, #713f12 0%, #854d0e 100%)';
      }

      return {
        id: node.id,
        data: { label: node.name || node.label, originalNode: node },
        position: { x: 0, y: 0 },
        style: {
          background: '#1f2937', 
          backgroundImage: bgGradient,
          color: textColor,
          border: `2px solid ${borderColor}`,
          borderRadius: '12px',
          padding: '10px',
          fontSize: '12px',
          fontWeight: isLocked ? 'normal' : 'bold',
          width: 180,
          textAlign: 'center',
          boxShadow: isLocked ? 'none' : '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
          opacity: isLocked ? 0.6 : 1,
        },
      };
    });

    const edgesSource = data.links || (data as any).edges || [];
    const initialEdges: Edge[] = edgesSource.map((edge: any, i: number) => ({
      id: `e${i}-${edge.source}-${edge.target}`,
      source: typeof edge.source === 'object' ? edge.source.id : edge.source,
      target: typeof edge.target === 'object' ? edge.target.id : edge.target,
      type: 'default', // 'smoothstep' can be better but 'default' (bezier) is standard
      animated: true,
      style: { stroke: '#4b5563', strokeWidth: 2 },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: '#4b5563',
      },
    }))

    getLayoutedElements(initialNodes, initialEdges).then(({ nodes: layoutedNodes, edges: layoutedEdges }) => {
      setNodes(layoutedNodes)
      setEdges(layoutedEdges)
      setLoading(false)
    })
  }, [data, getLayoutedElements, setNodes, setEdges])

  const onNodeClickCallback = useCallback((event: React.MouseEvent, node: Node) => {
    if (onNodeClick && node.data && node.data.originalNode) {
      onNodeClick(node.data.originalNode);
    }
  }, [onNodeClick]);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8 bg-gray-50 rounded-lg border border-gray-200" style={{ height }}>
        <Loader2 className="animate-spin text-blue-500 w-8 h-8" />
        <span className="ml-2 text-gray-500">Optimizing Graph Layout...</span>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden flex flex-col">
      <div className="p-6 border-b z-10 bg-white relative">
        <h3 className="text-lg font-semibold mb-2">🕸️ Knowledge Graph</h3>
        <p className="text-sm text-gray-600">
          Explore concept relationships and prerequisites.
        </p>
      </div>

      <div className="relative flex-grow bg-slate-50" style={{ height }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClickCallback}
          fitView
          attributionPosition="bottom-right"
        >
          <Background color="#cbd5e1" gap={16} size={1} />
          <Controls />
        </ReactFlow>

        {/* Legend Overlay */}
        <div className="absolute bottom-4 left-4 bg-white/90 backdrop-blur-sm rounded-lg p-4 shadow-lg border border-gray-200 text-xs">
          <div className="font-semibold text-gray-700 mb-2">Skill Tree Status</div>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-emerald-500" />
              <span className="text-gray-600">Mastered (≥95%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-orange-500" />
              <span className="text-gray-600">Learning (50-94%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-yellow-500" />
              <span className="text-gray-600">Unlocked (0-49%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-slate-600" />
              <span className="text-gray-600">Locked (Prereqs missing)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
