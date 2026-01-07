'use client'

export interface PerformanceStats {
  avgAccuracy: number // 0-1 scale
  totalStudyTimeMs: number
  avgEngagement: number // 0-1 scale
  cardsReviewed: number
  sessionsCompleted: number
  currentStreak: number
  longestStreak: number
  avgDwellTimeMs: number
}

export interface PerformanceMetricsProps {
  stats: PerformanceStats
  title?: string
}

function formatDuration(ms: number): string {
  const hours = Math.floor(ms / (1000 * 60 * 60))
  const minutes = Math.floor((ms % (1000 * 60 * 60)) / (1000 * 60))

  if (hours > 0) {
    return `${hours}h ${minutes}m`
  }
  return `${minutes}m`
}

interface StatCardProps {
  title: string
  value: string | number
  subtitle?: string
  icon: string
  color: 'blue' | 'green' | 'purple' | 'orange' | 'red'
  trend?: 'up' | 'down' | 'stable'
}

function StatCard({ title, value, subtitle, icon, color, trend }: StatCardProps) {
  const colorClasses = {
    blue: 'from-blue-50 to-indigo-50 text-blue-600',
    green: 'from-green-50 to-emerald-50 text-green-600',
    purple: 'from-purple-50 to-pink-50 text-purple-600',
    orange: 'from-orange-50 to-amber-50 text-orange-600',
    red: 'from-red-50 to-rose-50 text-red-600'
  }

  const trendIcons = {
    up: '📈',
    down: '📉',
    stable: '➡️'
  }

  return (
    <div className={`bg-gradient-to-br ${colorClasses[color]} rounded-lg p-6 shadow-md`}>
      <div className="flex items-start justify-between mb-2">
        <div className="text-2xl">{icon}</div>
        {trend && <div className="text-lg">{trendIcons[trend]}</div>}
      </div>
      <div className="text-sm text-gray-600 mb-1">{title}</div>
      <div className={`text-3xl font-bold ${colorClasses[color].split(' ')[2]}`}>
        {value}
      </div>
      {subtitle && (
        <div className="text-xs text-gray-500 mt-1">{subtitle}</div>
      )}
    </div>
  )
}

export function PerformanceMetrics({ stats, title = '📊 Performance Metrics' }: PerformanceMetricsProps) {
  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold">{title}</h3>

      {/* Primary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <StatCard
          icon="🎯"
          title="Average Accuracy"
          value={`${Math.round(stats.avgAccuracy * 100)}%`}
          subtitle="Success rate across all cards"
          color="blue"
        />

        <StatCard
          icon="⏱️"
          title="Total Study Time"
          value={formatDuration(stats.totalStudyTimeMs)}
          subtitle={`Avg ${formatDuration(stats.avgDwellTimeMs)} per card`}
          color="green"
        />

        <StatCard
          icon="🧠"
          title="Engagement Level"
          value={`${Math.round(stats.avgEngagement * 100)}%`}
          subtitle="Based on telemetry data"
          color="purple"
        />
      </div>

      {/* Secondary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon="📝"
          title="Cards Reviewed"
          value={stats.cardsReviewed}
          subtitle="Total reviews completed"
          color="blue"
        />

        <StatCard
          icon="🎓"
          title="Sessions"
          value={stats.sessionsCompleted}
          subtitle="Learning sessions"
          color="green"
        />

        <StatCard
          icon="🔥"
          title="Current Streak"
          value={`${stats.currentStreak} days`}
          subtitle={`Best: ${stats.longestStreak} days`}
          color="orange"
        />

        <StatCard
          icon="📚"
          title="Avg Cards/Session"
          value={stats.sessionsCompleted > 0
            ? Math.round(stats.cardsReviewed / stats.sessionsCompleted)
            : 0
          }
          subtitle="Learning intensity"
          color="purple"
        />
      </div>

      {/* Performance Breakdown */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h4 className="text-md font-semibold mb-4">Performance Breakdown</h4>
        <div className="space-y-3">
          {/* Accuracy */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-600">Accuracy</span>
              <span className="font-medium">{Math.round(stats.avgAccuracy * 100)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${
                  stats.avgAccuracy >= 0.7 ? 'bg-green-500' :
                  stats.avgAccuracy >= 0.35 ? 'bg-yellow-500' :
                  'bg-red-500'
                }`}
                style={{ width: `${stats.avgAccuracy * 100}%` }}
              />
            </div>
          </div>

          {/* Engagement */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-600">Engagement</span>
              <span className="font-medium">{Math.round(stats.avgEngagement * 100)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${
                  stats.avgEngagement >= 0.7 ? 'bg-purple-500' :
                  stats.avgEngagement >= 0.4 ? 'bg-blue-500' :
                  'bg-gray-400'
                }`}
                style={{ width: `${stats.avgEngagement * 100}%` }}
              />
            </div>
          </div>

          {/* Consistency (based on streak) */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-600">Consistency</span>
              <span className="font-medium">
                {Math.min(100, Math.round((stats.currentStreak / 30) * 100))}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-orange-500 h-2 rounded-full"
                style={{ width: `${Math.min(100, (stats.currentStreak / 30) * 100)}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
