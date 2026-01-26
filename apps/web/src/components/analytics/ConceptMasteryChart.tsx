'use client'

import { memo, useMemo } from 'react'
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip, Legend } from 'recharts'

export interface ConceptMastery {
  conceptName: string
  mastery: number // 0-1 scale
  cardsReviewed: number
  totalCards: number
}

export interface ConceptMasteryChartProps {
  concepts: ConceptMastery[]
  title?: string
  height?: number
}

export const ConceptMasteryChart = memo(function ConceptMasteryChart({ concepts, title = '\ud83d\udcda Concept Mastery', height = 400 }: ConceptMasteryChartProps) {
  // Memoize all computed data to avoid recalculation on re-renders
  const { chartData, avgMastery, strengths, weaknesses } = useMemo(() => {
    // Format data for radar chart (convert to 0-100 scale)
    const chartData = concepts.map(concept => ({
      concept: concept.conceptName,
      mastery: Math.round(concept.mastery * 100),
      fullName: concept.conceptName,
      reviewed: concept.cardsReviewed,
      total: concept.totalCards
    }))

    // Calculate average mastery
    const avgMastery = concepts.length > 0
      ? Math.round((concepts.reduce((sum, c) => sum + c.mastery, 0) / concepts.length) * 100)
      : 0

    // Sort by mastery to show strengths and weaknesses
    const sortedConcepts = [...concepts].sort((a, b) => b.mastery - a.mastery)
    const strengths = sortedConcepts.slice(0, 3)
    const weaknesses = sortedConcepts.slice(-3).reverse()

    return { chartData, avgMastery, strengths, weaknesses }
  }, [concepts])

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold mb-4">{title}</h3>

      {concepts.length === 0 ? (
        <div className="flex items-center justify-center h-64 text-gray-400">
          <div className="text-center">
            <p className="text-lg mb-2">📚</p>
            <p>No concept data yet</p>
            <p className="text-sm">Learn different concepts to see your mastery!</p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Radar Chart */}
          <div>
            <ResponsiveContainer width="100%" height={height}>
              <RadarChart data={chartData}>
                <PolarGrid stroke="#e5e7eb" />
                <PolarAngleAxis
                  dataKey="concept"
                  tick={{ fontSize: 11, fill: '#6b7280' }}
                />
                <PolarRadiusAxis
                  angle={90}
                  domain={[0, 100]}
                  tick={{ fontSize: 10 }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#ffffff',
                    border: '1px solid #e5e7eb',
                    borderRadius: '6px',
                    fontSize: '12px'
                  }}
                  formatter={(value: number, name: string, props: any) => [
                    `${value}% (${props.payload.reviewed}/${props.payload.total} cards)`,
                    'Mastery'
                  ]}
                  labelFormatter={(label) => `${label}`}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Radar
                  name="Mastery"
                  dataKey="mastery"
                  stroke="#3b82f6"
                  fill="#3b82f6"
                  fillOpacity={0.6}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          {/* Strengths and Weaknesses */}
          <div className="space-y-6">
            {/* Average Mastery */}
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-4">
              <div className="text-sm text-gray-600 mb-1">Average Mastery</div>
              <div className="text-3xl font-bold text-blue-600">{avgMastery}%</div>
              <div className="text-xs text-gray-500 mt-1">
                Across {concepts.length} concept{concepts.length !== 1 ? 's' : ''}
              </div>
            </div>

            {/* Strengths */}
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
                <span>💪</span>
                <span>Strengths</span>
              </h4>
              <div className="space-y-2">
                {strengths.map((concept, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <div className="flex-1">
                      <div className="flex justify-between text-xs mb-1">
                        <span className="font-medium text-gray-700">{concept.conceptName}</span>
                        <span className="text-gray-500">{Math.round(concept.mastery * 100)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-1.5">
                        <div
                          className="bg-green-500 h-1.5 rounded-full"
                          style={{ width: `${concept.mastery * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Weaknesses */}
            {weaknesses.length > 0 && weaknesses[0].mastery < 0.8 && (
              <div>
                <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
                  <span>📚</span>
                  <span>Focus Areas</span>
                </h4>
                <div className="space-y-2">
                  {weaknesses.map((concept, idx) => (
                    <div key={idx} className="flex items-center gap-2">
                      <div className="flex-1">
                        <div className="flex justify-between text-xs mb-1">
                          <span className="font-medium text-gray-700">{concept.conceptName}</span>
                          <span className="text-gray-500">{Math.round(concept.mastery * 100)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-1.5">
                          <div
                            className={`h-1.5 rounded-full ${
                              concept.mastery < 0.4 ? 'bg-red-500' : 'bg-yellow-500'
                            }`}
                            style={{ width: `${concept.mastery * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
})
