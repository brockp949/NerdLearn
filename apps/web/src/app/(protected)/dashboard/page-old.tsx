'use client'

import { ProtectedRoute } from '@/components/auth/ProtectedRoute'
import { useAuth } from '@/lib/auth-context'

export default function DashboardPage() {
  const { user, logout } = useAuth()

  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-gray-50">
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
                <a href="/graph" className="text-gray-600 hover:text-gray-900">Knowledge Graph</a>
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
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Welcome Section */}
          <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg shadow-lg p-8 text-white mb-8">
            <h2 className="text-3xl font-bold mb-2">
              Welcome back, {user?.username}! 👋
            </h2>
            <p className="text-blue-100 mb-4">
              Ready to continue your learning journey?
            </p>
            <a href="/learn" className="inline-block bg-white text-blue-600 px-6 py-3 rounded-lg font-semibold hover:bg-blue-50 transition">
              Start Learning →
            </a>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            {/* Total XP */}
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Total XP</p>
                  <p className="text-3xl font-bold text-gray-900">1,247</p>
                </div>
                <div className="text-4xl">⚡</div>
              </div>
              <div className="mt-4 flex items-center text-sm">
                <span className="text-green-600 font-medium">+124</span>
                <span className="text-gray-600 ml-1">this week</span>
              </div>
            </div>

            {/* Level */}
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Level</p>
                  <p className="text-3xl font-bold text-gray-900">5</p>
                </div>
                <div className="text-4xl">🎯</div>
              </div>
              <div className="mt-4">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-blue-600 h-2 rounded-full" style={{ width: '68%' }}></div>
                </div>
                <p className="text-xs text-gray-600 mt-1">320 / 470 XP to Level 6</p>
              </div>
            </div>

            {/* Streak */}
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Day Streak</p>
                  <p className="text-3xl font-bold text-gray-900">12</p>
                </div>
                <div className="text-4xl">🔥</div>
              </div>
              <div className="mt-4 flex items-center text-sm">
                <span className="text-orange-600 font-medium">2 freezes</span>
                <span className="text-gray-600 ml-1">available</span>
              </div>
            </div>

            {/* Cards Reviewed */}
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Cards Today</p>
                  <p className="text-3xl font-bold text-gray-900">23</p>
                </div>
                <div className="text-4xl">📚</div>
              </div>
              <div className="mt-4 flex items-center text-sm">
                <span className="text-blue-600 font-medium">15 due</span>
                <span className="text-gray-600 ml-1">remaining</span>
              </div>
            </div>
          </div>

          {/* Two Column Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Main Content - Recent Activity */}
            <div className="lg:col-span-2 space-y-6">
              {/* ZPD Status */}
              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  🌡️ Zone of Proximal Development
                </h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-green-50 border-l-4 border-green-500 rounded">
                    <div>
                      <p className="font-medium text-gray-900">Python Functions</p>
                      <p className="text-sm text-gray-600">Success Rate: 62% - Optimal Zone</p>
                    </div>
                    <span className="text-2xl">✅</span>
                  </div>

                  <div className="flex items-center justify-between p-4 bg-red-50 border-l-4 border-red-500 rounded">
                    <div>
                      <p className="font-medium text-gray-900">Recursion</p>
                      <p className="text-sm text-gray-600">Success Rate: 28% - Need Help</p>
                    </div>
                    <span className="text-2xl">⚠️</span>
                  </div>

                  <div className="flex items-center justify-between p-4 bg-yellow-50 border-l-4 border-yellow-500 rounded">
                    <div>
                      <p className="font-medium text-gray-900">Variables</p>
                      <p className="text-sm text-gray-600">Success Rate: 85% - Too Easy</p>
                    </div>
                    <span className="text-2xl">🎉</span>
                  </div>
                </div>
              </div>

              {/* Recent Achievements */}
              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  🏆 Recent Achievements
                </h3>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3 p-3 bg-purple-50 rounded-lg">
                    <div className="text-3xl">🔥</div>
                    <div>
                      <p className="font-medium text-gray-900">Week Warrior</p>
                      <p className="text-sm text-gray-600">Completed 7-day streak</p>
                    </div>
                    <div className="ml-auto text-sm text-gray-500">2 hours ago</div>
                  </div>

                  <div className="flex items-center space-x-3 p-3 bg-blue-50 rounded-lg">
                    <div className="text-3xl">⚡</div>
                    <div>
                      <p className="font-medium text-gray-900">XP Master</p>
                      <p className="text-sm text-gray-600">Earned 1,000 total XP</p>
                    </div>
                    <div className="ml-auto text-sm text-gray-500">1 day ago</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Sidebar - Quick Actions */}
            <div className="space-y-6">
              {/* Quick Actions */}
              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Quick Actions
                </h3>
                <div className="space-y-3">
                  <a href="/learn" className="block w-full text-left px-4 py-3 bg-blue-50 hover:bg-blue-100 rounded-lg transition">
                    <div className="flex items-center space-x-3">
                      <span className="text-2xl">📖</span>
                      <div>
                        <p className="font-medium text-gray-900">Review Due Cards</p>
                        <p className="text-xs text-gray-600">15 cards waiting</p>
                      </div>
                    </div>
                  </a>

                  <a href="/learn" className="block w-full text-left px-4 py-3 bg-green-50 hover:bg-green-100 rounded-lg transition">
                    <div className="flex items-center space-x-3">
                      <span className="text-2xl">🎓</span>
                      <div>
                        <p className="font-medium text-gray-900">Learn New Concepts</p>
                        <p className="text-xs text-gray-600">5 available</p>
                      </div>
                    </div>
                  </a>

                  <button className="w-full text-left px-4 py-3 bg-purple-50 hover:bg-purple-100 rounded-lg transition">
                    <div className="flex items-center space-x-3">
                      <span className="text-2xl">🌳</span>
                      <div>
                        <p className="font-medium text-gray-900">View Progress</p>
                        <p className="text-xs text-gray-600">Knowledge Graph</p>
                      </div>
                    </div>
                  </button>
                </div>
              </div>

              {/* Next Level */}
              <div className="bg-gradient-to-br from-yellow-400 to-orange-500 rounded-lg shadow p-6 text-white">
                <h3 className="text-lg font-semibold mb-2">
                  🎯 Next Level
                </h3>
                <p className="text-sm text-yellow-100 mb-4">
                  Just 150 XP away from Level 6!
                </p>
                <div className="bg-white/20 rounded-full h-2 mb-2">
                  <div className="bg-white h-2 rounded-full" style={{ width: '68%' }}></div>
                </div>
                <p className="text-xs text-yellow-100">
                  Keep up the great work!
                </p>
              </div>
            </div>
          </div>
        </main>
      </div>
    </ProtectedRoute>
  )
}
