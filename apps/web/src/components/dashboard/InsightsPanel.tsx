'use client'

export interface LearningInsights {
  bestTimeOfDay?: string
  mostProductiveDay?: string
  averageSessionLength?: number // minutes
  topConcepts: string[]
  weakConcepts: string[]
  recommendation: string
  streakMessage?: string
  motivationalMessage?: string
}

export interface InsightsPanelProps {
  insights: LearningInsights
  title?: string
}

interface InsightProps {
  icon: string
  title: string
  description: string
  actionButton?: React.ReactNode
  color?: 'blue' | 'green' | 'purple' | 'orange' | 'red'
}

function Insight({ icon, title, description, actionButton, color = 'blue' }: InsightProps) {
  const colorClasses = {
    blue: 'bg-blue-50 border-blue-200',
    green: 'bg-green-50 border-green-200',
    purple: 'bg-purple-50 border-purple-200',
    orange: 'bg-orange-50 border-orange-200',
    red: 'bg-red-50 border-red-200'
  }

  return (
    <div className={`rounded-lg border ${colorClasses[color]} p-4`}>
      <div className="flex items-start gap-3">
        <div className="text-2xl flex-shrink-0">{icon}</div>
        <div className="flex-1 min-w-0">
          <h4 className="font-semibold text-gray-900 mb-1">{title}</h4>
          <p className="text-sm text-gray-700 mb-2">{description}</p>
          {actionButton}
        </div>
      </div>
    </div>
  )
}

export function InsightsPanel({ insights, title = '💡 Insights & Recommendations' }: InsightsPanelProps) {
  return (
    <div className="bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold mb-4">{title}</h3>

      <div className="space-y-4">
        {/* Best Time */}
        {insights.bestTimeOfDay && (
          <Insight
            icon="🌟"
            title="Peak Performance"
            description={`You learn best ${insights.bestTimeOfDay}. Your focus and retention are highest during this time.`}
            color="blue"
          />
        )}

        {/* Most Productive Day */}
        {insights.mostProductiveDay && (
          <Insight
            icon="📅"
            title="Most Productive Day"
            description={`${insights.mostProductiveDay}s are your strongest learning days. Consider scheduling important topics then.`}
            color="green"
          />
        )}

        {/* Average Session Length */}
        {insights.averageSessionLength && (
          <Insight
            icon="⏱️"
            title="Session Length"
            description={`Your average session is ${insights.averageSessionLength} minutes. ${
              insights.averageSessionLength < 15
                ? 'Try extending sessions to 20-30 minutes for better retention.'
                : insights.averageSessionLength > 45
                ? 'Consider shorter, more focused sessions to prevent fatigue.'
                : 'This is an ideal session length for effective learning!'
            }`}
            color={
              insights.averageSessionLength < 15 ? 'orange' :
              insights.averageSessionLength > 45 ? 'purple' :
              'green'
            }
          />
        )}

        {/* Strengths */}
        {insights.topConcepts.length > 0 && (
          <Insight
            icon="💪"
            title="Top Strengths"
            description={`You're excelling at ${insights.topConcepts.slice(0, 2).join(' and ')}. ${
              insights.topConcepts.length > 2 ? `Plus ${insights.topConcepts.length - 2} more!` : ''
            } Great work!`}
            color="green"
          />
        )}

        {/* Areas for Improvement */}
        {insights.weakConcepts.length > 0 && (
          <Insight
            icon="📚"
            title="Focus Areas"
            description={`Consider spending more time on ${insights.weakConcepts.slice(0, 2).join(' and ')}. ${
              insights.weakConcepts.length > 2 ? `And ${insights.weakConcepts.length - 2} other concept${insights.weakConcepts.length - 2 > 1 ? 's' : ''}.` : ''
            } Regular practice will help solidify these concepts.`}
            color="orange"
          />
        )}

        {/* Streak Message */}
        {insights.streakMessage && (
          <Insight
            icon="🔥"
            title="Streak Status"
            description={insights.streakMessage}
            color="orange"
          />
        )}

        {/* Main Recommendation */}
        <Insight
          icon="🎯"
          title="Next Steps"
          description={insights.recommendation}
          color="purple"
          actionButton={
            <button
              onClick={() => window.location.href = '/learn'}
              className="text-sm bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors font-medium mt-2"
            >
              Start Learning →
            </button>
          }
        />

        {/* Motivational Message */}
        {insights.motivationalMessage && (
          <div className="bg-white bg-opacity-60 rounded-lg p-4 border border-purple-200">
            <div className="flex items-center gap-3">
              <div className="text-2xl">✨</div>
              <p className="text-sm italic text-gray-700">"{insights.motivationalMessage}"</p>
            </div>
          </div>
        )}
      </div>

      {/* Learning Progress Summary */}
      <div className="mt-6 pt-6 border-t border-white border-opacity-50">
        <h4 className="text-sm font-semibold text-gray-700 mb-3">Quick Stats</h4>
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-white bg-opacity-60 rounded-lg p-3">
            <div className="text-xs text-gray-600 mb-1">Strengths</div>
            <div className="text-lg font-bold text-green-600">{insights.topConcepts.length}</div>
          </div>
          <div className="bg-white bg-opacity-60 rounded-lg p-3">
            <div className="text-xs text-gray-600 mb-1">To Review</div>
            <div className="text-lg font-bold text-orange-600">{insights.weakConcepts.length}</div>
          </div>
        </div>
      </div>
    </div>
  )
}
