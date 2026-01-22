'use client'

import { useState, useEffect } from 'react'
import { ProgressChart, type ProgressData } from '@/components/analytics/ProgressChart'
import { SuccessRateChart, type SuccessRateData } from '@/components/analytics/SuccessRateChart'
import { ConceptMasteryChart, type ConceptMastery } from '@/components/analytics/ConceptMasteryChart'
import { PerformanceMetrics, type PerformanceStats } from '@/components/analytics/PerformanceMetrics'

// Mock data for now - will be replaced with API calls
const generateMockProgressData = (): ProgressData[] => {
  const data: ProgressData[] = []
  const startDate = new Date()
  startDate.setDate(startDate.getDate() - 30)

  for (let i = 0; i < 30; i++) {
    const date = new Date(startDate)
    date.setDate(date.getDate() + i)
    data.push({
      date: date.toISOString().split('T')[0],
      xp: Math.floor(10 + (i * 8) + Math.random() * 5),
      level: Math.floor((10 + i * 8) / 100) + 1
    })
  }
  return data
}

const generateMockSuccessRateData = (): SuccessRateData[] => {
  const data: SuccessRateData[] = []
  const startDate = new Date()
  startDate.setDate(startDate.getDate() - 30)

  for (let i = 0; i < 30; i++) {
    const date = new Date(startDate)
    date.setDate(date.getDate() + i)
    data.push({
      date: date.toISOString().split('T')[0],
      rate: 0.4 + Math.random() * 0.35, // 40-75% range
      cardsReviewed: Math.floor(5 + Math.random() * 10)
    })
  }
  return data
}

const generateMockConceptData = (): ConceptMastery[] => {
  const concepts = [
    'Variables',
    'Functions',
    'Loops',
    'Lists',
    'Dictionaries',
    'Control Flow',
    'Recursion',
    'Error Handling'
  ]

  return concepts.map(name => ({
    conceptName: name,
    mastery: 0.3 + Math.random() * 0.6, // 30-90% range
    cardsReviewed: Math.floor(5 + Math.random() * 15),
    totalCards: Math.floor(10 + Math.random() * 10)
  }))
}

const generateMockStats = (): PerformanceStats => {
  return {
    avgAccuracy: 0.68,
    totalStudyTimeMs: 7 * 60 * 60 * 1000, // 7 hours
    avgEngagement: 0.72,
    cardsReviewed: 145,
    sessionsCompleted: 23,
    currentStreak: 7,
    longestStreak: 12,
    avgDwellTimeMs: 12000 // 12 seconds per card
  }
}

export default function ProgressPage() {
  const [progressData, setProgressData] = useState<ProgressData[]>([])
  const [successRateData, setSuccessRateData] = useState<SuccessRateData[]>([])
  const [conceptData, setConceptData] = useState<ConceptMastery[]>([])
  const [stats, setStats] = useState<PerformanceStats | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // TODO: Replace with actual API calls
    // Example:
    // const fetchData = async () => {
    //   const userId = getUserId()
    //   const [progress, successRate, concepts, statistics] = await Promise.all([
    //     fetch(`/api/analytics/progress/${userId}`).then(r => r.json()),
    //     fetch(`/api/analytics/success-rate/${userId}`).then(r => r.json()),
    //     fetch(`/api/analytics/concepts/${userId}`).then(r => r.json()),
    //     fetch(`/api/analytics/performance/${userId}`).then(r => r.json())
    //   ])
    //   setProgressData(progress)
    //   setSuccessRateData(successRate)
    //   setConceptData(concepts)
    //   setStats(statistics)
    // }

    // For now, use mock data
    setTimeout(() => {
      setProgressData(generateMockProgressData())
      setSuccessRateData(generateMockSuccessRateData())
      setConceptData(generateMockConceptData())
      setStats(generateMockStats())
      setLoading(false)
    }, 500)
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-8">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-600">Loading analytics...</p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">📊 Your Learning Analytics</h1>
          <p className="text-gray-600">Track your progress, identify strengths, and optimize your learning journey</p>
        </div>

        {/* Performance Metrics */}
        {stats && <PerformanceMetrics stats={stats} />}

        {/* Charts Row 1 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ProgressChart data={progressData} />
          <SuccessRateChart data={successRateData} />
        </div>

        {/* Concept Mastery */}
        <ConceptMasteryChart concepts={conceptData} />

        {/* Additional Insights */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">💡 Insights & Recommendations</h3>

          <div className="space-y-4">
            {/* Learning Pattern */}
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0">
                <span>🌟</span>
              </div>
              <div>
                <h4 className="font-medium text-gray-900">Peak Performance Time</h4>
                <p className="text-sm text-gray-600">
                  You learn best in the morning (9 AM - 12 PM). Your success rate is 15% higher during this time.
                </p>
              </div>
            </div>

            {/* Strength */}
            {conceptData.length > 0 && (
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center flex-shrink-0">
                  <span>💪</span>
                </div>
                <div>
                  <h4 className="font-medium text-gray-900">Top Strength</h4>
                  <p className="text-sm text-gray-600">
                    You're excelling at {conceptData.sort((a, b) => b.mastery - a.mastery)[0].conceptName}
                    {' '}({Math.round(conceptData.sort((a, b) => b.mastery - a.mastery)[0].mastery * 100)}% mastery).
                    Consider helping others with this topic!
                  </p>
                </div>
              </div>
            )}

            {/* Focus Area */}
            {conceptData.length > 0 && (
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-full bg-orange-100 flex items-center justify-center flex-shrink-0">
                  <span>📚</span>
                </div>
                <div>
                  <h4 className="font-medium text-gray-900">Recommended Focus</h4>
                  <p className="text-sm text-gray-600">
                    Consider reviewing {conceptData.sort((a, b) => a.mastery - b.mastery)[0].conceptName}
                    {' '}({Math.round(conceptData.sort((a, b) => a.mastery - b.mastery)[0].mastery * 100)}% mastery).
                    Spending 15-20 minutes will help solidify this concept.
                  </p>
                </div>
              </div>
            )}

            {/* Streak Goal */}
            {stats && (
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-full bg-orange-100 flex items-center justify-center flex-shrink-0">
                  <span>🔥</span>
                </div>
                <div>
                  <h4 className="font-medium text-gray-900">Streak Goal</h4>
                  <p className="text-sm text-gray-600">
                    You're {stats.longestStreak - stats.currentStreak} day(s) away from beating your longest streak!
                    Keep it up to reach {stats.longestStreak + 1} days.
                  </p>
                </div>
              </div>
            )}

            {/* Next Session Suggestion */}
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-purple-100 flex items-center justify-center flex-shrink-0">
                <span>🎯</span>
              </div>
              <div className="flex-1">
                <h4 className="font-medium text-gray-900">Next Steps</h4>
                <p className="text-sm text-gray-600 mb-2">
                  Based on your progress, we recommend a 15-minute review session focusing on intermediate concepts.
                </p>
                <button
                  onClick={() => window.location.href = '/learn'}
                  className="text-sm bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Start Learning →
                </button>
              </div>
            </div>
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
            onClick={() => window.location.href = '/knowledge-graph'}
            className="px-6 py-3 bg-white text-gray-700 border border-gray-300 rounded-lg font-medium hover:bg-gray-50 transition-colors"
          >
            View Knowledge Graph
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
