"use client";

import { useCallback, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { motion, AnimatePresence } from "framer-motion";
import { ZoomIn, ZoomOut, Maximize2, Download, RefreshCw, Eye, EyeOff } from "lucide-react";
import type { DiagramData, DiagramNode, DiagramEdge } from "@/types/multimodal";
import type { Node, Edge } from 'reactflow';

// Dynamically import ReactFlow to avoid SSR issues
const ReactFlow = dynamic(
  () => import("reactflow").then((mod) => mod.default),
  { ssr: false }
);

const Background = dynamic(
  () => import("reactflow").then((mod) => mod.Background),
  { ssr: false }
);

const Controls = dynamic(
  () => import("reactflow").then((mod) => mod.Controls),
  { ssr: false }
);

const MiniMap = dynamic(
  () => import("reactflow").then((mod) => mod.MiniMap),
  { ssr: false }
);

// Custom node styles based on type
const getNodeStyle = (type: string, selected: boolean) => {
  const baseStyle = {
    padding: "12px 16px",
    borderRadius: "8px",
    border: selected ? "2px solid #8b5cf6" : "1px solid rgba(255,255,255,0.1)",
    fontSize: "13px",
    fontWeight: 500,
    transition: "all 0.2s ease",
    boxShadow: selected
      ? "0 0 20px rgba(139, 92, 246, 0.3)"
      : "0 4px 12px rgba(0,0,0,0.3)",
  };

  const typeStyles: Record<string, React.CSSProperties> = {
    default: {
      ...baseStyle,
      backgroundColor: "rgba(59, 130, 246, 0.2)",
      color: "#93c5fd",
    },
    input: {
      ...baseStyle,
      backgroundColor: "rgba(16, 185, 129, 0.2)",
      color: "#6ee7b7",
      borderRadius: "50%",
    },
    output: {
      ...baseStyle,
      backgroundColor: "rgba(239, 68, 68, 0.2)",
      color: "#fca5a5",
    },
    concept: {
      ...baseStyle,
      backgroundColor: "rgba(139, 92, 246, 0.2)",
      color: "#c4b5fd",
    },
    process: {
      ...baseStyle,
      backgroundColor: "rgba(14, 165, 233, 0.2)",
      color: "#7dd3fc",
      borderRadius: "16px",
    },
    decision: {
      ...baseStyle,
      backgroundColor: "rgba(245, 158, 11, 0.2)",
      color: "#fcd34d",
      transform: "rotate(45deg)",
    },
    group: {
      ...baseStyle,
      backgroundColor: "rgba(107, 114, 128, 0.2)",
      color: "#d1d5db",
      border: "2px dashed rgba(255,255,255,0.2)",
    },
    annotation: {
      ...baseStyle,
      backgroundColor: "rgba(168, 85, 247, 0.15)",
      color: "#d8b4fe",
      fontStyle: "italic",
    },
  };

  return typeStyles[type] || typeStyles.default;
};

interface InteractiveDiagramProps {
  diagram: DiagramData;
  onNodeClick?: (node: DiagramNode) => void;
  onEdgeClick?: (edge: DiagramEdge) => void;
  className?: string;
  showMiniMap?: boolean;
  showControls?: boolean;
  interactive?: boolean;
  onDiagramUpdate?: (nodes: DiagramNode[], edges: DiagramEdge[]) => void;
}

export function InteractiveDiagram({
  diagram,
  onNodeClick,
  onEdgeClick,
  className = "",
  showMiniMap = true,
  showControls = true,
  interactive = true,
  onDiagramUpdate,
}: InteractiveDiagramProps) {
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [showLabels, setShowLabels] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);




  // Convert diagram data to ReactFlow format
  const nodes: Node[] = useMemo(() => {
    return diagram.nodes.map((node) => ({
      id: node.id,
      type: "default",
      position: node.position,
      data: {
        label: showLabels ? node.data.label : "",
        description: node.data.description,
        originalType: node.type,
      },
      style: getNodeStyle(node.type, selectedNode === node.id),
      draggable: interactive,
    }));
  }, [diagram.nodes, selectedNode, showLabels, interactive]);

  const edges: Edge[] = useMemo(() => {
    return diagram.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: edge.type === "smoothstep" ? "smoothstep" : "default",
      label: edge.label,
      animated: edge.animated,
      style: {
        stroke: "rgba(139, 92, 246, 0.5)",
        strokeWidth: 2,
      },
      labelStyle: {
        fill: "#c4b5fd",
        fontSize: 11,
        fontWeight: 500,
      },
      markerEnd: edge.markerEnd
        ? {
          type: "arrowclosed" as any, // Cast to any or MarkerType if imported
          color: "rgba(139, 92, 246, 0.7)",
        }
        : undefined,
    }));
  }, [diagram.edges]);

  const handleNodeClick = useCallback(
    (_: any, node: any) => {
      setSelectedNode(node.id);
      const originalNode = diagram.nodes.find((n) => n.id === node.id);
      if (originalNode && onNodeClick) {
        onNodeClick(originalNode);
      }
    },
    [diagram.nodes, onNodeClick]
  );

  const handleEdgeClick = useCallback(
    (_: any, edge: any) => {
      const originalEdge = diagram.edges.find((e) => e.id === edge.id);
      if (originalEdge && onEdgeClick) {
        onEdgeClick(originalEdge);
      }
    },
    [diagram.edges, onEdgeClick]
  );

  const handleNodesChange = useCallback(
    (changes: any) => {
      if (onDiagramUpdate && interactive) {
        // Handle position updates
        const positionChanges = changes.filter(
          (c: any) => c.type === "position" && c.dragging === false
        );
        if (positionChanges.length > 0) {
          const updatedNodes = diagram.nodes.map((node) => {
            const change = positionChanges.find(
              (c: any) => c.id === node.id
            );
            if (change && change.position) {
              return { ...node, position: change.position };
            }
            return node;
          });
          onDiagramUpdate(updatedNodes, diagram.edges);
        }
      }
    },
    [diagram.nodes, diagram.edges, onDiagramUpdate, interactive]
  );

  const downloadMermaid = useCallback(() => {
    const blob = new Blob([diagram.mermaidSource], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${diagram.title.replace(/\s+/g, "_")}.mmd`;
    a.click();
    URL.revokeObjectURL(url);
  }, [diagram]);

  return (
    <div
      className={`relative rounded-xl border border-white/10 bg-black/40 overflow-hidden ${className} ${isFullscreen ? "fixed inset-4 z-50" : ""
        }`}
    >
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between px-4 py-2 bg-gradient-to-b from-black/80 to-transparent">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-white/80">
            {diagram.title}
          </span>
          <span className="text-xs text-white/40 px-2 py-0.5 rounded bg-white/10">
            {diagram.type}
          </span>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={() => setShowLabels(!showLabels)}
            className="p-1.5 rounded hover:bg-white/10 transition-colors"
            title={showLabels ? "Hide labels" : "Show labels"}
          >
            {showLabels ? (
              <Eye className="w-4 h-4 text-white/60" />
            ) : (
              <EyeOff className="w-4 h-4 text-white/60" />
            )}
          </button>
          <button
            onClick={downloadMermaid}
            className="p-1.5 rounded hover:bg-white/10 transition-colors"
            title="Download Mermaid source"
          >
            <Download className="w-4 h-4 text-white/60" />
          </button>
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1.5 rounded hover:bg-white/10 transition-colors"
            title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
          >
            <Maximize2 className="w-4 h-4 text-white/60" />
          </button>
        </div>
      </div>

      {/* React Flow Canvas */}
      <div className="w-full h-full min-h-[400px]">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodeClick={handleNodeClick}
          onEdgeClick={handleEdgeClick}
          onNodesChange={handleNodesChange}
          fitView
          attributionPosition="bottom-left"
          proOptions={{ hideAttribution: true }}
          defaultEdgeOptions={{
            type: "smoothstep",
            animated: false,
          }}
        >
          <Background
            color="rgba(139, 92, 246, 0.1)"
            gap={20}
            size={1}
          />
          {showControls && (
            <Controls
              showZoom={true}
              showFitView={true}
              showInteractive={interactive}
              position="bottom-right"
            />
          )}
          {showMiniMap && (
            <MiniMap
              nodeColor={(node) => {
                const type = node.data?.originalType || "default";
                const colors: Record<string, string> = {
                  default: "#3b82f6",
                  input: "#10b981",
                  output: "#ef4444",
                  concept: "#8b5cf6",
                  process: "#0ea5e9",
                  decision: "#f59e0b",
                };
                return colors[type] || colors.default;
              }}
              maskColor="rgba(0, 0, 0, 0.8)"
              style={{
                backgroundColor: "rgba(0, 0, 0, 0.5)",
                border: "1px solid rgba(255,255,255,0.1)",
              }}
            />
          )}
        </ReactFlow>
      </div>

      {/* Node Details Panel */}
      <AnimatePresence>
        {selectedNode && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="absolute bottom-4 left-4 right-4 p-4 rounded-lg bg-black/80 border border-white/10 backdrop-blur-sm"
          >
            {(() => {
              const node = diagram.nodes.find((n) => n.id === selectedNode);
              if (!node) return null;
              return (
                <div className="flex items-start justify-between">
                  <div>
                    <h4 className="text-sm font-medium text-white">
                      {node.data.label}
                    </h4>
                    {node.data.description && (
                      <p className="text-xs text-white/60 mt-1">
                        {node.data.description}
                      </p>
                    )}
                    <div className="flex items-center gap-2 mt-2">
                      <span className="text-xs text-white/40 px-2 py-0.5 rounded bg-white/10">
                        {node.type}
                      </span>
                      <span className="text-xs text-white/40">
                        ID: {node.id}
                      </span>
                    </div>
                  </div>
                  <button
                    onClick={() => setSelectedNode(null)}
                    className="text-white/40 hover:text-white/80 transition-colors"
                  >
                    ×
                  </button>
                </div>
              );
            })()}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Stats Badge */}
      <div className="absolute bottom-4 right-4 flex items-center gap-2 text-xs text-white/40">
        <span>{diagram.metadata.node_count} nodes</span>
        <span>•</span>
        <span>{diagram.metadata.edge_count} edges</span>
      </div>
    </div>
  );
}

export default InteractiveDiagram;
