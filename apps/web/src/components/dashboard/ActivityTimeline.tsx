'use client'

import { formatDistanceToNow } from 'date-fns'

export interface Activity {
  id: string
  type: 'session_completed' | 'achievement_unlocked' | 'level_up' | 'concept_mastered' | 'streak_milestone'
  title: string
  description: string
  timestamp: Date
  metadata?: {
    xp_earned?: number
    level?: number
    achievement_name?: string
    concept_name?: string
    streak_days?: number
    cards_reviewed?: number
  }
}

export interface ActivityTimelineProps {
  activities: Activity[]
  title?: string
  maxItems?: number
}

function getActivityIcon(type: Activity['type']): string {
  switch (type) {
    case 'session_completed':
      return '📝'
    case 'achievement_unlocked':
      return '🏆'
    case 'level_up':
      return '⬆️'
    case 'concept_mastered':
      return '✅'
    case 'streak_milestone':
      return '🔥'
    default:
      return '📌'
  }
}

function getActivityColor(type: Activity['type']): string {
  switch (type) {
    case 'session_completed':
      return 'bg-blue-100 text-blue-600'
    case 'achievement_unlocked':
      return 'bg-purple-100 text-purple-600'
    case 'level_up':
      return 'bg-green-100 text-green-600'
    case 'concept_mastered':
      return 'bg-emerald-100 text-emerald-600'
    case 'streak_milestone':
      return 'bg-orange-100 text-orange-600'
    default:
      return 'bg-gray-100 text-gray-600'
  }
}

function ActivityItem({ activity }: { activity: Activity }) {
  const icon = getActivityIcon(activity.type)
  const colorClass = getActivityColor(activity.type)

  return (
    <div className="flex items-start gap-3 group">
      {/* Icon */}
      <div className={`w-10 h-10 rounded-full ${colorClass} flex items-center justify-center flex-shrink-0 text-lg`}>
        {icon}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            <p className="font-medium text-gray-900 group-hover:text-blue-600 transition-colors">
              {activity.title}
            </p>
            <p className="text-sm text-gray-600 mt-0.5">{activity.description}</p>

            {/* Metadata */}
            {activity.metadata && (
              <div className="flex flex-wrap gap-2 mt-2">
                {activity.metadata.xp_earned !== undefined && (
                  <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded">
                    +{activity.metadata.xp_earned} XP
                  </span>
                )}
                {activity.metadata.level !== undefined && (
                  <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded">
                    Level {activity.metadata.level}
                  </span>
                )}
                {activity.metadata.cards_reviewed !== undefined && (
                  <span className="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded">
                    {activity.metadata.cards_reviewed} cards
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Timestamp */}
          <span className="text-xs text-gray-400 flex-shrink-0">
            {formatDistanceToNow(activity.timestamp, { addSuffix: true })}
          </span>
        </div>
      </div>
    </div>
  )
}

export function ActivityTimeline({ activities, title = '📅 Recent Activity', maxItems = 10 }: ActivityTimelineProps) {
  const displayedActivities = activities.slice(0, maxItems)

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">{title}</h3>
        {activities.length > maxItems && (
          <button
            onClick={() => window.location.href = '/activity'}
            className="text-sm text-blue-600 hover:text-blue-700 font-medium"
          >
            View All →
          </button>
        )}
      </div>

      {displayedActivities.length === 0 ? (
        <div className="flex items-center justify-center py-12 text-gray-400">
          <div className="text-center">
            <p className="text-lg mb-2">📅</p>
            <p>No recent activity</p>
            <p className="text-sm">Start learning to see your progress here!</p>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {displayedActivities.map((activity) => (
            <ActivityItem key={activity.id} activity={activity} />
          ))}
        </div>
      )}

      {/* Quick Stats */}
      {activities.length > 0 && (
        <div className="mt-6 pt-6 border-t border-gray-100">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-gray-900">
                {activities.filter(a => a.type === 'session_completed').length}
              </div>
              <div className="text-xs text-gray-500">Sessions</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-600">
                {activities.filter(a => a.type === 'achievement_unlocked').length}
              </div>
              <div className="text-xs text-gray-500">Achievements</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-green-600">
                {activities.filter(a => a.type === 'concept_mastered').length}
              </div>
              <div className="text-xs text-gray-500">Mastered</div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
