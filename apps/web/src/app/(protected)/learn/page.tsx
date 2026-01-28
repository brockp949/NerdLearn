'use client'

import { useState, useEffect, useCallback } from 'react'
import { ProtectedRoute } from '@/components/auth/ProtectedRoute'
import { useAuth, getAccessToken } from '@/lib/auth-context'
import { ContentViewer } from '@/components/learning/ContentViewer'
import { QuestionCard, LearningCard, Rating } from '@/components/learning/QuestionCard'
import { ScaffoldingPanel } from '@/components/learning/ScaffoldingPanel'
import { LearningStats } from '@/components/learning/LearningStats'
import { EngagementMeter } from '@/components/learning/EngagementMeter'
import { TelemetryTracker, AffectState, FrustrationIndex } from '@/lib/telemetry-tracker'
import Link from 'next/link'
import { SplitScreenLayout } from '@/components/learning/split-screen'
import { SidebarTabs } from '@/components/learning/SidebarTabs'
import { ChatInterface } from '@/components/chat/chat-interface'
import { KnowledgeGraphView } from '@/components/analytics/KnowledgeGraphView'
import { useGraphData } from '@/hooks/use-graph-data'
import { XPGainNotification } from '@/components/learning/XPGainNotification'
import { TestingPanel } from '@/components/testing/TestingPanel'

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
  const { user } = useAuth()
  const [session, setSession] = useState<SessionState | null>(null)
  const [viewMode, setViewMode] = useState<ViewMode>('idle')
  const { data: graphData, loading: graphLoading } = useGraphData()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [lastResponse, setLastResponse] = useState<AnswerResponse | null>(null)
  const [dwellStartTime, setDwellStartTime] = useState<number>(0)

  // Telemetry state
  const [telemetryClient, setTelemetryClient] = useState<TelemetryTracker | null>(null)
  const [affectState, setAffectState] = useState<AffectState>('neutral')
  const [frustrationIndex, setFrustrationIndex] = useState<FrustrationIndex | null>(null)
  const [telemetryConnected, setTelemetryConnected] = useState(false)
  const [hesitationCount, setHesitationCount] = useState(0)

  const API_URL = '/api/session'

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

      // Initialize telemetry
      // Initialize telemetry
      const client = new TelemetryTracker({
        apiEndpoint: process.env.NEXT_PUBLIC_TELEMETRY_URL || 'http://localhost:8002/api/telemetry',
        batchSize: 50
      })

      // client.onEngagement(setEngagement) // Tracker handles this differently now, adapt or remove
      // client.onConnection(setTelemetryConnected) // Tracker manages its own connection state implicitly
      client.init(user?.id ? parseInt(user.id) : 0) // Initialize

      setTelemetryClient(client)

      // Subscribe to affect changes
      client.onAffectChange((state) => {
        setAffectState(state)
        setFrustrationIndex(client.getFrustrationIndex())
      })

    } catch (err: any) {
      setError(err.message || 'Failed to start learning session')
    } finally {
      setLoading(false)
    }
  }

  // Cleanup telemetry on unmount
  useEffect(() => {
    return () => {
      if (telemetryClient) {
        telemetryClient.destroy()
      }
    }
  }, [telemetryClient])

  // Mouse tracking is now handled internally by TelemetryTracker
  // We don't need to manually forward mouse events

  // Mouse event listener management is handled by TelemetryTracker

  const handleContentContinue = () => {
    // Track content dwell time
    const dwellTime = Date.now() - dwellStartTime
    if (telemetryClient && session?.current_card) {
      telemetryClient.trackEvent('content_interaction', {
        eventId: 'dwell_time',
        cardId: session.current_card.card_id,
        duration: dwellTime
      })
    }

    setViewMode('question')
    setDwellStartTime(Date.now())
    setHesitationCount(0) // Reset hesitation counter
  }

  const handleAnswer = async (rating: Rating) => {
    if (!session || !session.current_card) return

    setLoading(true)
    const dwellTime = Date.now() - dwellStartTime

    // Track interaction
    if (telemetryClient) {
      telemetryClient.trackEvent('content_interaction', {
        eventId: 'answer_submitted',
        cardId: session.current_card.card_id,
        rating,
        dwellTime,
        hesitationCount
      })
    }

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
          hesitation_count: hesitationCount
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
      <div className="h-screen flex flex-col bg-gray-50 overflow-hidden">
        {/* Header */}
        <header className="bg-white shadow-sm z-10 flex-shrink-0">
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
        {error && (
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 w-full">
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          </div>
        )}

        {(viewMode === 'content' || viewMode === 'question') && session ? (
          <SplitScreenLayout
            sidebar={
              <SidebarTabs
                chatContent={<ChatInterface className="h-full w-full border-none shadow-none rounded-none" />}
                graphContent={<div className="h-full w-full overflow-hidden flex flex-col"><KnowledgeGraphView data={graphData} height={600} /></div>}
                testingContent={<TestingPanel />}
                progressContent={
                  <div className="space-y-6">
                    {/* Engagement Meter */}
                    <EngagementMeter
                      engagement={{
                        score: frustrationIndex?.score || 0,
                        cognitive_load: affectState === 'confused' ? 'high' : 'medium',
                        attention_level: affectState === 'bored' ? 'low' : 'high',
                        timestamp: new Date().toISOString()
                      }}
                      connected={true}
                    />

                    {/* Session Progress */}
                    <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-100">
                      <h3 className="text-sm font-semibold text-gray-900 mb-3">Session Progress</h3>
                      <div className="space-y-3">
                        <div className="flex justify-between text-xs text-gray-600">
                          <span>Progress</span>
                          <span className="font-medium text-gray-900">{session.cards_reviewed} / 20</span>
                        </div>
                        <div className="w-full bg-gray-100 rounded-full h-1.5">
                          <div
                            className="bg-blue-500 h-1.5 rounded-full transition-all"
                            style={{
                              width: `${session.cards_reviewed > 0
                                ? (session.cards_correct / session.cards_reviewed) * 100
                                : 0}%`
                            }}
                          />
                        </div>
                        <div className="flex justify-between text-xs border-t pt-2 mt-2">
                          <span className="text-gray-500">Correct</span>
                          <span className="font-medium text-green-600">{session.cards_correct}</span>
                        </div>
                        <button
                          onClick={endSession}
                          className="w-full mt-2 py-1.5 px-3 bg-red-50 text-red-600 text-xs font-medium rounded hover:bg-red-100 transition"
                        >
                          End Session
                        </button>
                      </div>
                    </div>

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
                    <div className="bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg shadow-md p-4 text-white">
                      <h3 className="text-sm font-semibold mb-2">💡 Tips</h3>
                      <ul className="space-y-1 text-xs opacity-90">
                        <li>• Read carefully</li>
                        <li>• Rate honestly</li>
                        <li>• Maintain streak</li>
                      </ul>
                    </div>
                  </div>
                }
              />
            }
          >
            <div className="max-w-4xl mx-auto py-6">
              {viewMode === 'content' && session.current_card && (
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

              {viewMode === 'question' && session.current_card && (
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
            </div>
          </SplitScreenLayout>
        ) : (
          <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 overflow-auto w-full">
            {/* Idle State */}
            {viewMode === 'idle' && (
              <div className="bg-white rounded-lg shadow-lg p-12 text-center max-w-2xl mx-auto mt-10">
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

            {/* Completed State */}
            {viewMode === 'completed' && session && (
              <div className="bg-white rounded-lg shadow-lg p-12 text-center max-w-2xl mx-auto mt-10">
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
          </main>
        )}

        {/* Global Notifications */}
        {lastResponse && (lastResponse.xp_earned > 0 || (lastResponse as any).level_up) && (
          <XPGainNotification 
            xp={lastResponse.xp_earned} 
            levelUp={(lastResponse as any).level_up}
            newLevel={lastResponse.level}
            ageGroup={(user?.age_group as any) || 'adult'} 
            onComplete={() => {
              // Optionally clear the XP notification state here if needed
            }}
          />
        )}
      </div>
    </ProtectedRoute>
  )
}
