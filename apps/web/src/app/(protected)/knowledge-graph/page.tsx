'use client'

import { useState, useEffect } from 'react'
import { KnowledgeGraphView, type GraphData, type GraphNode } from '@/components/analytics/KnowledgeGraphView'
import { ConceptDetail } from '@/components/analytics/ConceptDetail'

// Mock data generator
const generateMockGraphData = (): GraphData => {
  const concepts = [
    { id: '1', name: 'Variables', domain: 'Python', mastery: 0.98, cardsReviewed: 12, totalCards: 15, bloomLevel: 'REMEMBER', isLocked: false },
    { id: '2', name: 'Functions', domain: 'Python', mastery: 0.72, cardsReviewed: 10, totalCards: 15, bloomLevel: 'UNDERSTAND', isLocked: false },
    { id: '3', name: 'Loops', domain: 'Python', mastery: 0.65, cardsReviewed: 8, totalCards: 12, bloomLevel: 'APPLY', isLocked: false },
    { id: '4', name: 'Lists', domain: 'Python', mastery: 0.78, cardsReviewed: 11, totalCards: 14, bloomLevel: 'UNDERSTAND', isLocked: false },
    { id: '5', name: 'Dictionaries', domain: 'Python', mastery: 0.55, cardsReviewed: 6, totalCards: 12, bloomLevel: 'UNDERSTAND', isLocked: false },
    { id: '6', name: 'Control Flow', domain: 'Python', mastery: 0.80, cardsReviewed: 9, totalCards: 11, bloomLevel: 'APPLY', isLocked: false },
    { id: '7', name: 'Recursion', domain: 'Python', mastery: 0.35, cardsReviewed: 4, totalCards: 10, bloomLevel: 'ANALYZE', isLocked: false },
    { id: '8', name: 'Error Handling', domain: 'Python', mastery: 0.42, cardsReviewed: 5, totalCards: 10, bloomLevel: 'APPLY', isLocked: false },
    { id: '9', name: 'File I/O', domain: 'Python', mastery: 0.0, cardsReviewed: 0, totalCards: 8, bloomLevel: 'APPLY', isLocked: true },
    { id: '10', name: 'Classes', domain: 'Python', mastery: 0.0, cardsReviewed: 0, totalCards: 12, bloomLevel: 'CREATE', isLocked: true }
  ]

  const prerequisites = [
    { source: '2', target: '1', weight: 0.9, type: 'prerequisite' as const }, // Functions → Variables
    { source: '3', target: '1', weight: 0.8, type: 'prerequisite' as const }, // Loops → Variables
    { source: '3', target: '6', weight: 0.7, type: 'prerequisite' as const }, // Loops → Control Flow
    { source: '4', target: '1', weight: 0.8, type: 'prerequisite' as const }, // Lists → Variables
    { source: '5', target: '4', weight: 0.9, type: 'prerequisite' as const }, // Dictionaries → Lists
    { source: '7', target: '2', weight: 0.9, type: 'prerequisite' as const }, // Recursion → Functions
    { source: '8', target: '2', weight: 0.7, type: 'prerequisite' as const }, // Error Handling → Functions
    { source: '9', target: '8', weight: 0.8, type: 'prerequisite' as const }, // File I/O → Error Handling
    { source: '10', target: '2', weight: 0.9, type: 'prerequisite' as const }, // Classes → Functions
    { source: '10', target: '5', weight: 0.8, type: 'prerequisite' as const }  // Classes → Dictionaries
  ]

  return {
    nodes: concepts,
    edges: prerequisites
  }
}

export default function KnowledgeGraphPage() {
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [selectedConcept, setSelectedConcept] = useState<GraphNode | null>(null)
  const [prerequisites, setPrerequisites] = useState<GraphNode[]>([])
  const [dependents, setDependents] = useState<GraphNode[]>([])
  const [loading, setLoading] = useState(true)
  const [viewMode, setViewMode] = useState<'graph' | 'detail'>('graph')

  useEffect(() => {
    // TODO: Replace with actual API call
    // const fetchGraphData = async () => {
    //   const userId = getUserId()
    //   const data = await fetch(`/api/knowledge-graph/${userId}`).then(r => r.json())
    //   setGraphData(data)
    // }

    // For now, use mock data
    setTimeout(() => {
      const data = generateMockGraphData()
      setGraphData(data)
      setLoading(false)
    }, 500)
  }, [])

  const handleNodeClick = (node: GraphNode) => {
    if (!graphData) return

    setSelectedConcept(node)

    // Find prerequisites (concepts that this concept depends on)
    const prereqIds = graphData.edges
      .filter(edge => edge.source === node.id)
      .map(edge => edge.target)

    const prereqNodes = graphData.nodes.filter(n => prereqIds.includes(n.id))
    setPrerequisites(prereqNodes)

    // Find dependents (concepts that depend on this concept)
    const depIds = graphData.edges
      .filter(edge => edge.target === node.id)
      .map(edge => edge.source)

    const depNodes = graphData.nodes.filter(n => depIds.includes(n.id))
    setDependents(depNodes)

    // On mobile, switch to detail view
    if (window.innerWidth < 1024) {
      setViewMode('detail')
    }
  }

  const handleCloseDetail = () => {
    setSelectedConcept(null)
    setPrerequisites([])
    setDependents([])
    setViewMode('graph')
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-8">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-center h-96">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-600">Loading Knowledge Graph...</p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (!graphData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-8">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-center h-96">
            <div className="text-center text-gray-400">
              <p className="text-lg mb-2">🕸️</p>
              <p>No knowledge graph data available</p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-4 md:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">🕸️ Knowledge Graph</h1>
            <p className="text-gray-600">
              Visualize concept relationships and track your learning path
            </p>
          </div>

          {/* Mobile View Toggle */}
          <div className="flex gap-2 lg:hidden">
            <button
              onClick={() => setViewMode('graph')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                viewMode === 'graph'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-700 border border-gray-300'
              }`}
            >
              Graph
            </button>
            <button
              onClick={() => setViewMode('detail')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                viewMode === 'detail'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-700 border border-gray-300'
              }`}
              disabled={!selectedConcept}
            >
              Details
            </button>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white rounded-lg shadow-md p-4">
            <div className="text-sm text-gray-600 mb-1">Total Concepts</div>
            <div className="text-2xl font-bold text-gray-900">{graphData.nodes.length}</div>
          </div>
          <div className="bg-white rounded-lg shadow-md p-4">
            <div className="text-sm text-gray-600 mb-1">Mastered</div>
            <div className="text-2xl font-bold text-green-600">
              {graphData.nodes.filter(n => n.mastery >= 0.8).length}
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-md p-4">
            <div className="text-sm text-gray-600 mb-1">In Progress</div>
            <div className="text-2xl font-bold text-yellow-600">
              {graphData.nodes.filter(n => n.mastery > 0 && n.mastery < 0.8).length}
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-md p-4">
            <div className="text-sm text-gray-600 mb-1">Not Started</div>
            <div className="text-2xl font-bold text-gray-400">
              {graphData.nodes.filter(n => n.mastery === 0).length}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Graph View */}
          <div className={`lg:col-span-2 ${viewMode === 'detail' ? 'hidden lg:block' : ''}`}>
            <KnowledgeGraphView
              data={graphData}
              onNodeClick={handleNodeClick}
              height={600}
            />
          </div>

          {/* Concept Detail */}
          <div className={`lg:col-span-1 ${viewMode === 'graph' ? 'hidden lg:block' : ''}`}>
            <ConceptDetail
              concept={selectedConcept}
              prerequisites={prerequisites}
              dependents={dependents}
              onClose={handleCloseDetail}
            />
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-4">
          <button
            onClick={() => window.location.href = '/learn'}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
          >
            Start Learning
          </button>
          <button
            onClick={() => window.location.href = '/progress'}
            className="px-6 py-3 bg-white text-gray-700 border border-gray-300 rounded-lg font-medium hover:bg-gray-50 transition-colors"
          >
            View Progress
          </button>
          <button
            onClick={() => window.location.href = '/dashboard'}
            className="px-6 py-3 bg-white text-gray-700 border border-gray-300 rounded-lg font-medium hover:bg-gray-50 transition-colors"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    </div>
  )
}
