'use client'

import { useState, useEffect } from 'react'
import { ProtectedRoute } from '@/components/auth/ProtectedRoute'
import { useAuth } from '@/lib/auth-context'
import { ContentViewer } from '@/components/learning/ContentViewer'
import { QuestionCard, LearningCard, Rating } from '@/components/learning/QuestionCard'
import { ScaffoldingPanel } from '@/components/learning/ScaffoldingPanel'
import { LearningStats } from '@/components/learning/LearningStats'
import Link from 'next/link'

interface SessionState {
  session_id: string
  learner_id: string
  current_card: LearningCard | null
  cards_reviewed: number
  cards_correct: number
  total_xp_earned: number
  current_streak: number
  zpd_zone: string
  scaffolding_active: string[]
  started_at: string
  achievements_unlocked: string[]
}

interface AnswerResponse {
  correct: boolean
  xp_earned: number
  new_total_xp: number
  level: number
  level_progress: number
  next_card: LearningCard | null
  zpd_zone: string
  zpd_message: string
  scaffolding: any
  achievement_unlocked: any
}

type ViewMode = 'idle' | 'content' | 'question' | 'completed'

export default function LearnPage() {
  const { user, getAccessToken } = useAuth()
  const [session, setSession] = useState<SessionState | null>(null)
  const [viewMode, setViewMode] = useState<ViewMode>('idle')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [lastResponse, setLastResponse] = useState<AnswerResponse | null>(null)
  const [dwellStartTime, setDwellStartTime] = useState<number>(0)

  const API_URL = 'http://localhost:8005'

  const startSession = async () => {
    setLoading(true)
    setError('')

    try {
      const response = await fetch(`${API_URL}/session/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          learner_id: user?.id || 'demo_user',
          domain: 'Python',
          limit: 20
        })
      })

      if (!response.ok) {
        throw new Error('Failed to start session')
      }

      const data: SessionState = await response.json()
      setSession(data)
      setViewMode('content')
      setDwellStartTime(Date.now())
    } catch (err: any) {
      setError(err.message || 'Failed to start learning session')
    } finally {
      setLoading(false)
    }
  }

  const handleContentContinue = () => {
    setViewMode('question')
    setDwellStartTime(Date.now())
  }

  const handleAnswer = async (rating: Rating) => {
    if (!session || !session.current_card) return

    setLoading(true)
    const dwellTime = Date.now() - dwellStartTime

    try {
      const response = await fetch(`${API_URL}/session/answer`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: session.session_id,
          card_id: session.current_card.card_id,
          rating: rating,
          dwell_time_ms: dwellTime,
          hesitation_count: 0
        })
      })

      if (!response.ok) {
        throw new Error('Failed to process answer')
      }

      const data: AnswerResponse = await response.json()
      setLastResponse(data)

      // Update session with new stats
      if (data.next_card) {
        setSession({
          ...session,
          current_card: data.next_card,
          cards_reviewed: session.cards_reviewed + 1,
          cards_correct: session.cards_correct + (data.correct ? 1 : 0),
          total_xp_earned: session.total_xp_earned + data.xp_earned,
          zpd_zone: data.zpd_zone
        })
        setViewMode('content')
        setDwellStartTime(Date.now())
      } else {
        // No more cards
        setViewMode('completed')
      }
    } catch (err: any) {
      setError(err.message || 'Failed to process answer')
    } finally {
      setLoading(false)
    }
  }

  const endSession = async () => {
    if (!session) return

    try {
      const response = await fetch(`${API_URL}/session/${session.session_id}/end`, {
        method: 'POST'
      })

      if (response.ok) {
        const summary = await response.json()
        console.log('Session summary:', summary)
      }
    } catch (err) {
      console.error('Failed to end session:', err)
    }

    setSession(null)
    setViewMode('idle')
    setLastResponse(null)
  }

  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-pink-50">
        {/* Header */}
        <header className="bg-white shadow-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
            <div className="flex items-center space-x-4">
              <Link href="/dashboard" className="text-2xl">🧠</Link>
              <h1 className="text-xl font-bold text-gray-900">NerdLearn</h1>
            </div>
            <div className="flex items-center space-x-4">
              <Link href="/dashboard" className="text-sm text-gray-600 hover:text-gray-900">
                ← Back to Dashboard
              </Link>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Main Learning Area */}
            <div className="lg:col-span-2 space-y-6">
              {/* Idle State - Start Session */}
              {viewMode === 'idle' && (
                <div className="bg-white rounded-lg shadow-lg p-12 text-center">
                  <span className="text-6xl mb-4 block">🎓</span>
                  <h2 className="text-3xl font-bold text-gray-900 mb-4">
                    Ready to Learn?
                  </h2>
                  <p className="text-gray-600 mb-8 max-w-md mx-auto">
                    Start your adaptive learning session. We'll personalize the experience to keep you in the optimal learning zone.
                  </p>
                  <button
                    onClick={startSession}
                    disabled={loading}
                    className="px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-semibold rounded-lg hover:from-purple-700 hover:to-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed shadow-lg"
                  >
                    {loading ? 'Starting...' : 'Start Learning Session'}
                  </button>
                </div>
              )}

              {/* Content View */}
              {viewMode === 'content' && session?.current_card && (
                <>
                  {lastResponse && (
                    <ScaffoldingPanel
                      scaffolding={lastResponse.scaffolding}
                      zpd_zone={lastResponse.zpd_zone}
                      zpd_message={lastResponse.zpd_message}
                    />
                  )}
                  <ContentViewer
                    card={session.current_card}
                    onContinue={handleContentContinue}
                  />
                </>
              )}

              {/* Question View */}
              {viewMode === 'question' && session?.current_card && (
                <>
                  {lastResponse && (
                    <ScaffoldingPanel
                      scaffolding={lastResponse.scaffolding}
                      zpd_zone={lastResponse.zpd_zone}
                      zpd_message={lastResponse.zpd_message}
                    />
                  )}
                  <QuestionCard
                    card={session.current_card}
                    onAnswer={handleAnswer}
                    loading={loading}
                  />
                </>
              )}

              {/* Completed State */}
              {viewMode === 'completed' && session && (
                <div className="bg-white rounded-lg shadow-lg p-12 text-center">
                  <span className="text-6xl mb-4 block">🎉</span>
                  <h2 className="text-3xl font-bold text-gray-900 mb-4">
                    Session Complete!
                  </h2>
                  <div className="max-w-md mx-auto space-y-4 mb-8">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-4 bg-blue-50 rounded-lg">
                        <p className="text-sm text-gray-600">Cards Reviewed</p>
                        <p className="text-2xl font-bold text-gray-900">{session.cards_reviewed}</p>
                      </div>
                      <div className="p-4 bg-green-50 rounded-lg">
                        <p className="text-sm text-gray-600">Success Rate</p>
                        <p className="text-2xl font-bold text-gray-900">
                          {session.cards_reviewed > 0
                            ? Math.round((session.cards_correct / session.cards_reviewed) * 100)
                            : 0}%
                        </p>
                      </div>
                      <div className="p-4 bg-purple-50 rounded-lg">
                        <p className="text-sm text-gray-600">XP Earned</p>
                        <p className="text-2xl font-bold text-gray-900">{session.total_xp_earned}</p>
                      </div>
                      <div className="p-4 bg-orange-50 rounded-lg">
                        <p className="text-sm text-gray-600">Achievements</p>
                        <p className="text-2xl font-bold text-gray-900">{session.achievements_unlocked.length}</p>
                      </div>
                    </div>
                  </div>
                  <div className="flex space-x-4 justify-center">
                    <button
                      onClick={startSession}
                      className="px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-semibold rounded-lg hover:from-purple-700 hover:to-blue-700 transition"
                    >
                      Start New Session
                    </button>
                    <Link
                      href="/dashboard"
                      className="px-6 py-3 bg-gray-200 text-gray-700 font-semibold rounded-lg hover:bg-gray-300 transition"
                    >
                      Back to Dashboard
                    </Link>
                  </div>
                </div>
              )}
            </div>

            {/* Sidebar - Session Stats */}
            <div className="space-y-6">
              {/* Session Progress */}
              {session && viewMode !== 'idle' && (
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Session Progress</h3>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-600">Cards Reviewed</span>
                        <span className="font-semibold text-gray-900">{session.cards_reviewed}</span>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-600">Correct Answers</span>
                        <span className="font-semibold text-gray-900">{session.cards_correct}</span>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-600">Success Rate</span>
                        <span className="font-semibold text-gray-900">
                          {session.cards_reviewed > 0
                            ? Math.round((session.cards_correct / session.cards_reviewed) * 100)
                            : 0}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-green-500 h-2 rounded-full transition-all"
                          style={{
                            width: `${session.cards_reviewed > 0
                              ? (session.cards_correct / session.cards_reviewed) * 100
                              : 0}%`
                          }}
                        />
                      </div>
                    </div>
                    <div className="pt-4 border-t">
                      <button
                        onClick={endSession}
                        className="w-full py-2 px-4 bg-red-100 text-red-700 font-medium rounded-lg hover:bg-red-200 transition text-sm"
                      >
                        End Session
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Stats Display */}
              {lastResponse && (
                <LearningStats
                  xp_earned={lastResponse.xp_earned}
                  new_total_xp={lastResponse.new_total_xp}
                  level={lastResponse.level}
                  level_progress={lastResponse.level_progress}
                  achievement={lastResponse.achievement_unlocked}
                  showAnimation={true}
                />
              )}

              {/* Tips */}
              <div className="bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg shadow-md p-6 text-white">
                <h3 className="text-lg font-semibold mb-3">💡 Learning Tips</h3>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start space-x-2">
                    <span>•</span>
                    <span>Read content carefully before answering</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span>•</span>
                    <span>Be honest with your ratings for best results</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span>•</span>
                    <span>Review daily to maintain your streak</span>
                  </li>
                  <li className="flex items-start space-x-2">
                    <span>•</span>
                    <span>ZPD adapts to keep you challenged</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </main>
      </div>
    </ProtectedRoute>
  )
}
