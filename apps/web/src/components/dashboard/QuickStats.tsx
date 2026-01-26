'use client'

export interface QuickStatsData {
  level: number
  totalXP: number
  xpToNextLevel: number
  levelProgress: number
  currentStreak: number
  streakShields: number
  cardsReviewed: number
  conceptsMastered: number
}

export interface QuickStatsProps {
  stats: QuickStatsData
}

interface StatCardProps {
  icon: string
  label: string
  value: string | number
  subtitle?: string
  color: 'blue' | 'green' | 'purple' | 'orange'
  showProgress?: boolean
  progress?: number // 0-100
}

function StatCard({ icon, label, value, subtitle, color, showProgress, progress }: StatCardProps) {
  const colorClasses = {
    blue: 'from-blue-500 to-indigo-600',
    green: 'from-green-500 to-emerald-600',
    purple: 'from-purple-500 to-pink-600',
    orange: 'from-orange-500 to-red-600'
  }

  return (
    <div className={`bg-gradient-to-br ${colorClasses[color]} rounded-lg p-6 text-white shadow-lg relative overflow-hidden`}>
      <div className="flex items-start justify-between mb-2">
        <div className="text-3xl opacity-90">{icon}</div>
      </div>
      <div className="text-sm opacity-90 mb-1">{label}</div>
      <div className="text-3xl font-bold mb-1">{value}</div>
      {subtitle && (
        <div className="text-xs opacity-75">{subtitle}</div>
      )}
      {showProgress && progress !== undefined && (
        <div className="mt-3">
          <div className="w-full bg-white bg-opacity-20 rounded-full h-1.5">
            <div
              className="bg-white h-1.5 rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}
    </div>
  )
}

export function QuickStats({ stats }: QuickStatsProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      <StatCard
        icon="⬆️"
        label="Level"
        value={stats.level}
        subtitle={`${stats.xpToNextLevel} XP to next level`}
        color="purple"
        showProgress={true}
        progress={stats.levelProgress}
      />

      <StatCard
        icon="📊"
        label="Total XP"
        value={stats.totalXP.toLocaleString()}
        subtitle="Keep learning!"
        color="blue"
      />

      <StatCard
        icon="🔥"
        label="Current Streak"
        value={`${stats.currentStreak} days`}
        subtitle={stats.streakShields > 0 ? `🛡️ ${stats.streakShields} Shield(s) Active` : 'Keep it up!'}
        color="orange"
      />

      <StatCard
        icon="📝"
        label="Cards Reviewed"
        value={stats.cardsReviewed}
        subtitle="Total reviews completed"
        color="green"
      />

      <StatCard
        icon="✅"
        label="Concepts Mastered"
        value={stats.conceptsMastered}
        subtitle="≥80% mastery"
        color="purple"
      />

      <StatCard
        icon="🎯"
        label="Success Rate"
        value="68%"
        subtitle="Across all cards"
        color="blue"
      />
    </div>
  )
}
