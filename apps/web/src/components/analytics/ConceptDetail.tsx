'use client'

import { GraphNode } from './KnowledgeGraphView'

export interface ConceptDetailProps {
  concept: GraphNode | null
  prerequisites?: GraphNode[]
  dependents?: GraphNode[]
  onClose?: () => void
}

function getMasteryLabel(mastery: number): string {
  if (mastery >= 0.8) return 'Mastered'
  if (mastery >= 0.6) return 'Proficient'
  if (mastery >= 0.4) return 'Learning'
  if (mastery > 0) return 'Struggling'
  return 'Not Started'
}

function getMasteryColor(mastery: number): string {
  if (mastery >= 0.8) return 'text-green-600 bg-green-100'
  if (mastery >= 0.6) return 'text-blue-600 bg-blue-100'
  if (mastery >= 0.4) return 'text-yellow-600 bg-yellow-100'
  if (mastery > 0) return 'text-red-600 bg-red-100'
  return 'text-gray-600 bg-gray-100'
}

function getProgressColor(mastery: number): string {
  if (mastery >= 0.8) return 'bg-green-500'
  if (mastery >= 0.6) return 'bg-blue-500'
  if (mastery >= 0.4) return 'bg-yellow-500'
  if (mastery > 0) return 'bg-red-500'
  return 'bg-gray-300'
}

export function ConceptDetail({ concept, prerequisites = [], dependents = [], onClose }: ConceptDetailProps) {
  if (!concept) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="text-center text-gray-400 py-12">
          <p className="text-lg mb-2">🎯</p>
          <p>Select a concept to view details</p>
        </div>
      </div>
    )
  }

  const masteryPercentage = Math.round(concept.mastery * 100)
  const cardsProgress = concept.totalCards > 0
    ? Math.round((concept.cardsReviewed / concept.totalCards) * 100)
    : 0

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-500 to-indigo-600 p-6 text-white">
        <div className="flex justify-between items-start">
          <div className="flex-1">
            <h3 className="text-2xl font-bold mb-2">{concept.name}</h3>
            <div className="flex items-center gap-3">
              <span className="text-sm opacity-90">Domain: {concept.domain}</span>
              {concept.bloomLevel && (
                <>
                  <span className="opacity-50">•</span>
                  <span className="text-sm opacity-90">Level: {concept.bloomLevel}</span>
                </>
              )}
            </div>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="text-white hover:text-gray-200 transition-colors"
              aria-label="Close"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Mastery Status */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <h4 className="font-semibold text-gray-900">Mastery Status</h4>
            <span className={`px-3 py-1 rounded-full text-xs font-medium ${getMasteryColor(concept.mastery)}`}>
              {getMasteryLabel(concept.mastery)}
            </span>
          </div>

          {/* Progress Bar */}
          <div className="mb-3">
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-600">Mastery Level</span>
              <span className="font-medium text-gray-900">{masteryPercentage}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className={`h-3 rounded-full transition-all ${getProgressColor(concept.mastery)}`}
                style={{ width: `${masteryPercentage}%` }}
              />
            </div>
          </div>

          {/* Cards Progress */}
          <div className="grid grid-cols-3 gap-3 text-sm">
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-gray-600 text-xs mb-1">Cards Reviewed</div>
              <div className="text-xl font-bold text-gray-900">{concept.cardsReviewed}</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-gray-600 text-xs mb-1">Total Cards</div>
              <div className="text-xl font-bold text-gray-900">{concept.totalCards}</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="text-gray-600 text-xs mb-1">Progress</div>
              <div className="text-xl font-bold text-gray-900">{cardsProgress}%</div>
            </div>
          </div>
        </div>

        {/* Prerequisites */}
        {prerequisites.length > 0 && (
          <div>
            <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
              <span>⬅️</span>
              <span>Prerequisites</span>
              <span className="text-xs text-gray-500">({prerequisites.length})</span>
            </h4>
            <div className="space-y-2">
              {prerequisites.map((prereq, idx) => (
                <div key={idx} className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                  <div
                    className="w-3 h-3 rounded-full flex-shrink-0"
                    style={{ backgroundColor: getProgressColor(prereq.mastery) }}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-gray-900 truncate">{prereq.name}</div>
                    <div className="text-xs text-gray-500">
                      {Math.round(prereq.mastery * 100)}% mastered
                    </div>
                  </div>
                  {prereq.mastery < 0.6 && (
                    <span className="text-xs text-orange-600 bg-orange-100 px-2 py-1 rounded">
                      Review recommended
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Dependents (Unlocks) */}
        {dependents.length > 0 && (
          <div>
            <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
              <span>➡️</span>
              <span>Unlocks</span>
              <span className="text-xs text-gray-500">({dependents.length})</span>
            </h4>
            <div className="space-y-2">
              {dependents.map((dependent, idx) => (
                <div key={idx} className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                  <div
                    className="w-3 h-3 rounded-full flex-shrink-0"
                    style={{ backgroundColor: getProgressColor(dependent.mastery) }}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-gray-900 truncate">{dependent.name}</div>
                    <div className="text-xs text-gray-500">
                      {dependent.mastery > 0
                        ? `${Math.round(dependent.mastery * 100)}% mastered`
                        : 'Not yet started'
                      }
                    </div>
                  </div>
                  {concept.mastery >= 0.8 && dependent.mastery === 0 && (
                    <span className="text-xs text-green-600 bg-green-100 px-2 py-1 rounded">
                      Ready to unlock!
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recommendations */}
        <div className="bg-blue-50 rounded-lg p-4">
          <h4 className="font-semibold text-gray-900 mb-2">💡 Recommendation</h4>
          <p className="text-sm text-gray-700">
            {concept.mastery >= 0.8
              ? `Excellent work! You've mastered ${concept.name}. ${dependents.length > 0 ? `Consider moving on to ${dependents[0].name}.` : 'Keep up the great work!'}`
              : concept.mastery >= 0.6
                ? `You're doing well with ${concept.name}. A few more reviews will help solidify your understanding.`
                : concept.mastery >= 0.4
                  ? `Keep practicing ${concept.name}. Focus on the cards you find challenging.`
                  : concept.mastery > 0
                    ? `${concept.name} needs more attention. ${prerequisites.length > 0 && prerequisites.some(p => p.mastery < 0.6) ? `Consider reviewing the prerequisites first: ${prerequisites.filter(p => p.mastery < 0.6).map(p => p.name).join(', ')}.` : 'Take your time and review the fundamentals.'}`
                    : `Ready to start learning ${concept.name}? ${prerequisites.length > 0 ? `Make sure you're comfortable with: ${prerequisites.map(p => p.name).join(', ')}.` : 'Begin with the first few cards to build a foundation.'}`
            }
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3">
          <button
            onClick={() => window.location.href = `/learn?concept=${concept.id}`}
            className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
          >
            Practice This Concept
          </button>
          <button
            onClick={() => window.location.href = `/concepts/${concept.id}`}
            className="px-4 py-2 bg-white text-gray-700 border border-gray-300 rounded-lg font-medium hover:bg-gray-50 transition-colors"
          >
            View Cards
          </button>
        </div>
      </div>
    </div>
  )
}
