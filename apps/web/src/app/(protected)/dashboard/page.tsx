'use client'

import { useState, useEffect } from 'react'
import { ProtectedRoute } from '@/components/auth/ProtectedRoute'
import { useAuth } from '@/lib/auth-context'
import { QuickStats, type QuickStatsData } from '@/components/dashboard/QuickStats'
import { ActivityTimeline, type Activity } from '@/components/dashboard/ActivityTimeline'
import { InsightsPanel, type LearningInsights } from '@/components/dashboard/InsightsPanel'
import { RecentAchievements, type Achievement as AchievementData } from '@/components/dashboard/RecentAchievements'

// Mock data generators
const generateMockStats = (): QuickStatsData => ({
  level: 5,
  totalXP: 1247,
  xpToNextLevel: 223,
  levelProgress: 82, // Percentage of progress in current level
  currentStreak: 12,
  streakShields: 2,   // Added streak shields
  cardsReviewed: 145,
  conceptsMastered: 4
})

const generateMockAchievements = (): AchievementData[] => ([
  {
    id: 'a1',
    name: 'First Steps',
    description: 'Complete your first module',
    icon: '🎯',
    unlockedAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
    rarity: 'common'
  },
  {
    id: 'a2',
    name: 'Week Warrior',
    description: 'Maintain a 7-day learning streak',
    icon: '🔥',
    unlockedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
    rarity: 'rare'
  },
  {
    id: 'a3',
    name: 'Speed Learner',
    description: 'Complete a course in under 7 days',
    icon: '⚡',
    unlockedAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
    rarity: 'rare'
  }
])

const generateMockActivities = (): Activity[] => {
  const now = new Date()
  return [
    {
      id: '1',
      type: 'session_completed',
      title: 'Completed Learning Session',
      description: 'Reviewed 10 cards on Python Functions',
      timestamp: new Date(now.getTime() - 2 * 60 * 60 * 1000), // 2 hours ago
      metadata: { xp_earned: 85, cards_reviewed: 10 }
    },
    {
      id: '2',
      type: 'achievement_unlocked',
      title: 'Achievement Unlocked!',
      description: 'Week Warrior - Completed 7-day streak',
      timestamp: new Date(now.getTime() - 3 * 60 * 60 * 1000), // 3 hours ago
      metadata: { achievement_name: 'Week Warrior' }
    },
    {
      id: '3',
      type: 'level_up',
      title: 'Level Up!',
      description: 'Reached Level 5',
      timestamp: new Date(now.getTime() - 1 * 24 * 60 * 60 * 1000), // 1 day ago
      metadata: { level: 5, xp_earned: 150 }
    },
    {
      id: '4',
      type: 'concept_mastered',
      title: 'Concept Mastered',
      description: 'Achieved 80%+ mastery in Python Variables',
      timestamp: new Date(now.getTime() - 2 * 24 * 60 * 60 * 1000), // 2 days ago
      metadata: { concept_name: 'Python Variables' }
    },
    {
      id: '5',
      type: 'session_completed',
      title: 'Completed Learning Session',
      description: 'Reviewed 8 cards on Loops',
      timestamp: new Date(now.getTime() - 2 * 24 * 60 * 60 * 1000), // 2 days ago
      metadata: { xp_earned: 62, cards_reviewed: 8 }
    }
  ]
}

const generateMockInsights = (): LearningInsights => ({
  bestTimeOfDay: 'in the morning (9 AM - 12 PM)',
  mostProductiveDay: 'Tuesday',
  averageSessionLength: 22,
  topConcepts: ['Variables', 'Functions', 'Control Flow'],
  weakConcepts: ['Recursion', 'Error Handling'],
  recommendation: "Based on your progress, we recommend a 20-minute session focusing on Recursion. Your prerequisites are solid, so you're ready to tackle this concept.",
  streakMessage: "Great job! You're on a 12-day streak. Keep it going to reach your longest streak of 15 days!",
  motivationalMessage: "The expert in anything was once a beginner. Every card you review brings you closer to mastery."
})

export default function DashboardPage() {
  const { user, logout } = useAuth()
  const [stats, setStats] = useState<QuickStatsData | null>(null)
  const [activities, setActivities] = useState<Activity[]>([])
  const [achievements, setAchievements] = useState<AchievementData[]>([])
  const [insights, setInsights] = useState<LearningInsights | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // TODO: Replace with actual API calls
    // const fetchDashboardData = async () => {
    //   const userId = user?.id
    //   const [statsData, activitiesData, insightsData] = await Promise.all([
    //     fetch(`/api/dashboard/stats/${userId}`).then(r => r.json()),
    //     fetch(`/api/dashboard/activities/${userId}`).then(r => r.json()),
    //     fetch(`/api/dashboard/insights/${userId}`).then(r => r.json())
    //   ])
    //   setStats(statsData)
    //   setActivities(activitiesData)
    //   setInsights(insightsData)
    // }

    // For now, use mock data
    setTimeout(() => {
      setStats(generateMockStats())
      setActivities(generateMockActivities())
      setAchievements(generateMockAchievements())
      setInsights(generateMockInsights())
      setLoading(false)
    }, 500)
  }, [user])

  if (loading) {
    return (
      <ProtectedRoute>
        <div className="min-h-screen bg-gray-50 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading dashboard...</p>
          </div>
        </div>
      </ProtectedRoute>
    )
  }

  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
        {/* Header */}
        <header className="bg-white shadow">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
            <div className="flex items-center space-x-4">
              <h1 className="text-2xl font-bold text-gray-900">
                🧠 NerdLearn
              </h1>
              <nav className="hidden md:flex space-x-4">
                <a href="/dashboard" className="text-blue-600 font-medium">Dashboard</a>
                <a href="/learn" className="text-gray-600 hover:text-gray-900">Learn</a>
                <a href="/progress" className="text-gray-600 hover:text-gray-900">Progress</a>
                <a href="/knowledge-graph" className="text-gray-600 hover:text-gray-900">Knowledge Graph</a>
              </nav>
            </div>

            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-700">
                Welcome, <span className="font-medium">{user?.username}</span>
              </span>
              <button
                onClick={logout}
                className="px-4 py-2 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-md"
              >
                Logout
              </button>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
          {/* Welcome Section */}
          <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg shadow-lg p-8 text-white">
            <h2 className="text-3xl font-bold mb-2">
              Welcome back, {user?.username}! 👋
            </h2>
            <p className="text-blue-100 mb-4">
              Ready to continue your learning journey?
            </p>
            <div className="flex flex-wrap gap-3">
              <a
                href="/learn"
                className="inline-block bg-white text-blue-600 px-6 py-3 rounded-lg font-semibold hover:bg-blue-50 transition"
              >
                Start Learning →
              </a>
              <a
                href="/progress"
                className="inline-block bg-white bg-opacity-20 text-white px-6 py-3 rounded-lg font-semibold hover:bg-opacity-30 transition border border-white border-opacity-30"
              >
                View Progress
              </a>
            </div>
          </div>

          {/* Quick Stats */}
          {stats && <QuickStats stats={stats} ageGroup={user?.age_group} />}

          {/* Two Column Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Main Content */}
            <div className="lg:col-span-2 space-y-6">
              {/* Recent Achievements */}
              <RecentAchievements achievements={achievements} />

              {/* Activity Timeline */}
              <ActivityTimeline activities={activities} maxItems={5} />

              {/* ZPD Status (from original) */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  🌡️ Zone of Proximal Development
                </h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-green-50 border-l-4 border-green-500 rounded-lg">
                    <div>
                      <p className="font-medium text-gray-900">Python Functions</p>
                      <p className="text-sm text-gray-600">Success Rate: 62% - Optimal Zone</p>
                    </div>
                    <span className="text-2xl">✅</span>
                  </div>

                  <div className="flex items-center justify-between p-4 bg-red-50 border-l-4 border-red-500 rounded-lg">
                    <div>
                      <p className="font-medium text-gray-900">Recursion</p>
                      <p className="text-sm text-gray-600">Success Rate: 28% - Need Help</p>
                    </div>
                    <span className="text-2xl">⚠️</span>
                  </div>

                  <div className="flex items-center justify-between p-4 bg-blue-50 border-l-4 border-blue-500 rounded-lg">
                    <div>
                      <p className="font-medium text-gray-900">Variables</p>
                      <p className="text-sm text-gray-600">Success Rate: 85% - Comfort Zone</p>
                    </div>
                    <span className="text-2xl">🎉</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Sidebar */}
            <div className="space-y-6">
              {/* Insights Panel */}
              {insights && <InsightsPanel insights={insights} />}

              {/* Quick Actions */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Quick Actions
                </h3>
                <div className="space-y-3">
                  <a
                    href="/learn"
                    className="block w-full text-left px-4 py-3 bg-blue-50 hover:bg-blue-100 rounded-lg transition"
                  >
                    <div className="flex items-center space-x-3">
                      <span className="text-2xl">📖</span>
                      <div>
                        <p className="font-medium text-gray-900">Review Due Cards</p>
                        <p className="text-xs text-gray-600">15 cards waiting</p>
                      </div>
                    </div>
                  </a>

                  <a
                    href="/knowledge-graph"
                    className="block w-full text-left px-4 py-3 bg-purple-50 hover:bg-purple-100 rounded-lg transition"
                  >
                    <div className="flex items-center space-x-3">
                      <span className="text-2xl">🕸️</span>
                      <div>
                        <p className="font-medium text-gray-900">Knowledge Graph</p>
                        <p className="text-xs text-gray-600">Visualize progress</p>
                      </div>
                    </div>
                  </a>

                  <a
                    href="/progress"
                    className="block w-full text-left px-4 py-3 bg-green-50 hover:bg-green-100 rounded-lg transition"
                  >
                    <div className="flex items-center space-x-3">
                      <span className="text-2xl">📊</span>
                      <div>
                        <p className="font-medium text-gray-900">View Analytics</p>
                        <p className="text-xs text-gray-600">Charts and insights</p>
                      </div>
                    </div>
                  </a>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </ProtectedRoute>
  )
}
