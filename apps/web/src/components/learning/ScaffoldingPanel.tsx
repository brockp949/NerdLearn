'use client'

interface ScaffoldingData {
  type: string
  content: string
  show: boolean
}

interface ScaffoldingPanelProps {
  scaffolding: ScaffoldingData | null
  zpd_zone: string
  zpd_message: string
}

export function ScaffoldingPanel({ scaffolding, zpd_zone, zpd_message }: ScaffoldingPanelProps) {
  if (!scaffolding?.show && zpd_zone === 'optimal') {
    return null
  }

  const zoneConfig = {
    frustration: {
      color: 'red',
      bgColor: 'bg-red-50',
      borderColor: 'border-red-500',
      textColor: 'text-red-900',
      icon: '⚠️',
      title: 'Need Help?'
    },
    comfort: {
      color: 'yellow',
      bgColor: 'bg-yellow-50',
      borderColor: 'border-yellow-500',
      textColor: 'text-yellow-900',
      icon: '🎉',
      title: 'Great Progress!'
    },
    optimal: {
      color: 'green',
      bgColor: 'bg-green-50',
      borderColor: 'border-green-500',
      textColor: 'text-green-900',
      icon: '✅',
      title: 'Perfect Zone!'
    }
  }

  const config = zoneConfig[zpd_zone as keyof typeof zoneConfig] || zoneConfig.optimal

  return (
    <div className="max-w-3xl mx-auto mb-6">
      {/* ZPD Status Banner */}
      <div className={`p-4 ${config.bgColor} border-l-4 ${config.borderColor} rounded-lg mb-4`}>
        <div className="flex items-start space-x-3">
          <span className="text-2xl">{config.icon}</span>
          <div className="flex-1">
            <p className={`font-semibold ${config.textColor} mb-1`}>{config.title}</p>
            <p className={`text-sm ${config.textColor}`}>{zpd_message}</p>
          </div>
        </div>
      </div>

      {/* Scaffolding Content */}
      {scaffolding?.show && (
        <div className="bg-white rounded-lg shadow-lg p-6 border-2 border-purple-200">
          <div className="flex items-start space-x-3 mb-4">
            <span className="text-3xl">🎓</span>
            <div>
              <h3 className="text-lg font-bold text-gray-900">Learning Support</h3>
              <p className="text-sm text-gray-600">Here's some help to guide you</p>
            </div>
          </div>

          <div className="p-4 bg-purple-50 rounded-lg">
            {scaffolding.type === 'worked_example' && (
              <div>
                <p className="text-sm font-semibold text-purple-900 mb-2">📝 Worked Example:</p>
                <p className="text-gray-800 leading-relaxed">{scaffolding.content}</p>
              </div>
            )}

            {scaffolding.type === 'hint' && (
              <div>
                <p className="text-sm font-semibold text-purple-900 mb-2">💡 Hint:</p>
                <p className="text-gray-800 leading-relaxed">{scaffolding.content}</p>
              </div>
            )}

            {scaffolding.type === 'prerequisite_review' && (
              <div>
                <p className="text-sm font-semibold text-purple-900 mb-2">📚 Review This First:</p>
                <p className="text-gray-800 leading-relaxed">{scaffolding.content}</p>
              </div>
            )}
          </div>

          <div className="mt-4 flex items-center space-x-2 text-sm text-gray-600">
            <span>🌡️</span>
            <p>
              This support will be removed as you improve. You're currently in the{' '}
              <span className="font-semibold">{zpd_zone}</span> zone.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
