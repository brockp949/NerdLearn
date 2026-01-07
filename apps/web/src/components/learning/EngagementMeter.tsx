'use client'

interface EngagementScore {
  score: number // 0-1
  cognitive_load: 'low' | 'medium' | 'high'
  attention_level: 'low' | 'medium' | 'high'
  timestamp: string
}

interface EngagementMeterProps {
  engagement: EngagementScore | null
  connected: boolean
}

export function EngagementMeter({ engagement, connected }: EngagementMeterProps) {
  if (!connected && !engagement) {
    return null // Don't show if never connected
  }

  const score = engagement?.score || 0
  const percentage = Math.round(score * 100)

  // Color based on engagement level
  const getColor = () => {
    if (score >= 0.7) return 'bg-green-500'
    if (score >= 0.4) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  const getLoadColor = () => {
    if (!engagement) return 'text-gray-500'
    switch (engagement.cognitive_load) {
      case 'low': return 'text-green-600'
      case 'medium': return 'text-yellow-600'
      case 'high': return 'text-red-600'
      default: return 'text-gray-600'
    }
  }

  const getAttentionIcon = () => {
    if (!engagement) return '○'
    switch (engagement.attention_level) {
      case 'high': return '●●●'
      case 'medium': return '●●○'
      case 'low': return '●○○'
      default: return '○○○'
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-4">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <span className="text-xl">🧠</span>
          <h3 className="text-sm font-semibold text-gray-900">Engagement</h3>
        </div>
        <div className="flex items-center space-x-1">
          {connected ? (
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
          ) : (
            <span className="w-2 h-2 rounded-full bg-gray-400"></span>
          )}
          <span className="text-xs text-gray-500">
            {connected ? 'Live' : 'Offline'}
          </span>
        </div>
      </div>

      {/* Engagement Bar */}
      <div className="mb-3">
        <div className="flex justify-between items-center mb-1">
          <span className="text-xs text-gray-600">Level</span>
          <span className="text-sm font-semibold text-gray-900">{percentage}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3">
          <div
            className={`${getColor()} h-3 rounded-full transition-all duration-500 ease-out`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>

      {/* Metrics */}
      {engagement && (
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <p className="text-gray-500">Cognitive Load</p>
            <p className={`font-medium ${getLoadColor()}`}>
              {engagement.cognitive_load.charAt(0).toUpperCase() + engagement.cognitive_load.slice(1)}
            </p>
          </div>
          <div>
            <p className="text-gray-500">Attention</p>
            <p className="font-medium text-gray-700">
              {getAttentionIcon()}
            </p>
          </div>
        </div>
      )}

      {/* Offline Message */}
      {!connected && (
        <div className="mt-2 text-xs text-gray-500 italic">
          Telemetry offline - basic tracking active
        </div>
      )}
    </div>
  )
}
