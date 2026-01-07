'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import dynamic from 'next/dynamic'

// Dynamically import ForceGraph2D to avoid SSR issues
const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
        <p className="text-gray-600">Loading graph...</p>
      </div>
    </div>
  )
})

export interface GraphNode {
  id: string
  name: string
  domain: string
  mastery: number // 0-1 scale
  cardsReviewed: number
  totalCards: number
  bloomLevel?: string
}

export interface GraphEdge {
  source: string
  target: string
  weight: number
  type: 'prerequisite'
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface KnowledgeGraphViewProps {
  data: GraphData
  onNodeClick?: (node: GraphNode) => void
  height?: number
}

function getNodeColor(mastery: number): string {
  if (mastery >= 0.8) return '#10b981' // green - mastered
  if (mastery >= 0.4) return '#f59e0b' // yellow - learning
  if (mastery > 0) return '#ef4444' // red - struggling
  return '#9ca3af' // gray - not started
}

function getNodeSize(cardsReviewed: number, totalCards: number): number {
  const progress = totalCards > 0 ? cardsReviewed / totalCards : 0
  return 5 + progress * 5 // 5-10 range
}

export function KnowledgeGraphView({ data, onNodeClick, height = 600 }: KnowledgeGraphViewProps) {
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [highlightNodes, setHighlightNodes] = useState<Set<string>>(new Set())
  const [highlightLinks, setHighlightLinks] = useState<Set<string>>(new Set())
  const [hoverNode, setHoverNode] = useState<GraphNode | null>(null)
  const graphRef = useRef<any>()

  // Format data for react-force-graph
  const graphData = {
    nodes: data.nodes.map(node => ({
      ...node,
      color: getNodeColor(node.mastery),
      size: getNodeSize(node.cardsReviewed, node.totalCards)
    })),
    links: data.edges.map(edge => ({
      ...edge,
      source: edge.source,
      target: edge.target
    }))
  }

  const handleNodeClick = useCallback((node: any) => {
    const graphNode = node as GraphNode
    setSelectedNode(graphNode)
    if (onNodeClick) {
      onNodeClick(graphNode)
    }

    // Highlight connected nodes
    const connectedNodeIds = new Set<string>()
    const connectedLinkIds = new Set<string>()

    data.edges.forEach(edge => {
      if (edge.source === node.id) {
        connectedNodeIds.add(edge.target)
        connectedLinkIds.add(`${edge.source}-${edge.target}`)
      }
      if (edge.target === node.id) {
        connectedNodeIds.add(edge.source)
        connectedLinkIds.add(`${edge.source}-${edge.target}`)
      }
    })

    connectedNodeIds.add(node.id)
    setHighlightNodes(connectedNodeIds)
    setHighlightLinks(connectedLinkIds)
  }, [data.edges, onNodeClick])

  const handleNodeHover = useCallback((node: any | null) => {
    setHoverNode(node as GraphNode | null)
  }, [])

  const handleBackgroundClick = useCallback(() => {
    setSelectedNode(null)
    setHighlightNodes(new Set())
    setHighlightLinks(new Set())
  }, [])

  // Custom node rendering
  const nodeCanvasObject = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const label = node.name
    const fontSize = 12 / globalScale
    const nodeSize = node.size || 5

    // Draw node circle
    ctx.beginPath()
    ctx.arc(node.x, node.y, nodeSize, 0, 2 * Math.PI)
    ctx.fillStyle = node.color
    ctx.fill()

    // Add stroke for selected/highlighted nodes
    if (selectedNode?.id === node.id || highlightNodes.has(node.id)) {
      ctx.strokeStyle = '#1f2937'
      ctx.lineWidth = 2 / globalScale
      ctx.stroke()
    }

    // Draw label
    ctx.font = `${fontSize}px Sans-Serif`
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    ctx.fillStyle = '#1f2937'
    ctx.fillText(label, node.x, node.y + nodeSize + 2 / globalScale)

    // Draw mastery percentage for hovered node
    if (hoverNode?.id === node.id) {
      ctx.font = `${fontSize}px Sans-Serif`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'bottom'
      ctx.fillStyle = '#6b7280'
      ctx.fillText(`${Math.round(node.mastery * 100)}%`, node.x, node.y - nodeSize - 2 / globalScale)
    }
  }, [selectedNode, highlightNodes, hoverNode])

  // Custom link rendering
  const linkCanvasObject = useCallback((link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const linkId = `${link.source.id}-${link.target.id}`
    const isHighlighted = highlightLinks.has(linkId)

    const start = link.source
    const end = link.target

    // Calculate midpoint for arrow
    const midX = (start.x + end.x) / 2
    const midY = (start.y + end.y) / 2

    // Draw line
    ctx.beginPath()
    ctx.moveTo(start.x, start.y)
    ctx.lineTo(end.x, end.y)
    ctx.strokeStyle = isHighlighted ? '#3b82f6' : '#d1d5db'
    ctx.lineWidth = (isHighlighted ? 2 : 1) / globalScale
    ctx.stroke()

    // Draw arrow
    const angle = Math.atan2(end.y - start.y, end.x - start.x)
    const arrowSize = 6 / globalScale

    ctx.save()
    ctx.translate(midX, midY)
    ctx.rotate(angle)
    ctx.beginPath()
    ctx.moveTo(0, 0)
    ctx.lineTo(-arrowSize, -arrowSize / 2)
    ctx.lineTo(-arrowSize, arrowSize / 2)
    ctx.closePath()
    ctx.fillStyle = isHighlighted ? '#3b82f6' : '#d1d5db'
    ctx.fill()
    ctx.restore()
  }, [highlightLinks])

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      <div className="p-6 border-b">
        <h3 className="text-lg font-semibold mb-2">🕸️ Knowledge Graph</h3>
        <p className="text-sm text-gray-600">
          Explore concept relationships and prerequisites. Click a node to see details.
        </p>
      </div>

      <div className="relative" style={{ height }}>
        <ForceGraph2D
          ref={graphRef}
          graphData={graphData}
          nodeLabel="name"
          nodeCanvasObject={nodeCanvasObject}
          linkCanvasObject={linkCanvasObject}
          onNodeClick={handleNodeClick}
          onNodeHover={handleNodeHover}
          onBackgroundClick={handleBackgroundClick}
          cooldownTime={3000}
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.3}
          enableNodeDrag={true}
          enableZoomInteraction={true}
          enablePanInteraction={true}
        />

        {/* Legend */}
        <div className="absolute bottom-4 left-4 bg-white bg-opacity-95 rounded-lg p-4 shadow-lg">
          <div className="text-xs font-semibold text-gray-700 mb-2">Mastery Levels</div>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500" />
              <span className="text-xs text-gray-600">Mastered (≥80%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-yellow-500" />
              <span className="text-xs text-gray-600">Learning (40-79%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <span className="text-xs text-gray-600">Struggling (1-39%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-gray-400" />
              <span className="text-xs text-gray-600">Not Started (0%)</span>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="absolute top-4 right-4 bg-white bg-opacity-95 rounded-lg p-3 shadow-lg">
          <div className="text-xs font-semibold text-gray-700 mb-2">Controls</div>
          <div className="space-y-1 text-xs text-gray-600">
            <div>• Click & drag to pan</div>
            <div>• Scroll to zoom</div>
            <div>• Click node to select</div>
            <div>• Drag nodes to reposition</div>
          </div>
        </div>
      </div>
    </div>
  )
}
