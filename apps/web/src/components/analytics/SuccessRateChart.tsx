'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts'
import { format, parseISO } from 'date-fns'

export interface SuccessRateData {
  date: string
  rate: number // 0-1 scale
  cardsReviewed: number
}

export interface SuccessRateChartProps {
  data: SuccessRateData[]
  title?: string
  height?: number
}

export function SuccessRateChart({ data, title = '🎯 Success Rate Trend', height = 300 }: SuccessRateChartProps) {
  // Format data for recharts (convert to percentage)
  const chartData = data.map(item => ({
    ...item,
    formattedDate: format(parseISO(item.date), 'MMM d'),
    percentage: Math.round(item.rate * 100)
  }))

  // Calculate average success rate
  const avgRate = data.length > 0
    ? Math.round((data.reduce((sum, item) => sum + item.rate, 0) / data.length) * 100)
    : 0

  // Determine ZPD zone color
  const getZoneColor = (rate: number) => {
    if (rate < 0.35) return '#ef4444' // red - frustration
    if (rate > 0.70) return '#3b82f6' // blue - comfort
    return '#10b981' // green - optimal
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold mb-4">{title}</h3>

      {data.length === 0 ? (
        <div className="flex items-center justify-center h-64 text-gray-400">
          <div className="text-center">
            <p className="text-lg mb-2">🎯</p>
            <p>No success rate data yet</p>
            <p className="text-sm">Complete learning sessions to see your accuracy!</p>
          </div>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="formattedDate"
              stroke="#6b7280"
              style={{ fontSize: '12px' }}
            />
            <YAxis
              domain={[0, 100]}
              stroke="#6b7280"
              style={{ fontSize: '12px' }}
              label={{ value: 'Success Rate (%)', angle: -90, position: 'insideLeft', style: { fontSize: '12px' } }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                border: '1px solid #e5e7eb',
                borderRadius: '6px',
                fontSize: '12px'
              }}
              formatter={(value: number) => [`${value}%`, 'Success Rate']}
              labelFormatter={(label) => `Date: ${label}`}
            />
            <Legend wrapperStyle={{ fontSize: '12px' }} />

            {/* ZPD Zone Reference Lines */}
            <ReferenceLine
              y={35}
              stroke="#ef4444"
              strokeDasharray="3 3"
              label={{ value: 'Frustration Zone', position: 'insideTopLeft', fontSize: 10 }}
            />
            <ReferenceLine
              y={70}
              stroke="#3b82f6"
              strokeDasharray="3 3"
              label={{ value: 'Comfort Zone', position: 'insideTopLeft', fontSize: 10 }}
            />

            <Line
              type="monotone"
              dataKey="percentage"
              stroke="#10b981"
              strokeWidth={2}
              name="Success Rate"
              dot={(props) => {
                const { cx, cy, payload } = props
                return (
                  <circle
                    cx={cx}
                    cy={cy}
                    r={4}
                    fill={getZoneColor(payload.rate)}
                  />
                )
              }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      )}

      {/* Summary Stats */}
      {data.length > 0 && (
        <div className="mt-4">
          <div className="flex gap-6 text-sm text-gray-600 mb-3">
            <div>
              <span className="font-medium">Average:</span> {avgRate}%
            </div>
            <div>
              <span className="font-medium">Latest:</span> {Math.round(data[data.length - 1].rate * 100)}%
            </div>
            <div>
              <span className="font-medium">Total Cards:</span>{' '}
              {data.reduce((sum, item) => sum + item.cardsReviewed, 0)}
            </div>
          </div>

          {/* ZPD Zone Legend */}
          <div className="flex gap-4 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <span className="text-gray-600">Frustration (&lt;35%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500" />
              <span className="text-gray-600">Optimal (35-70%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-blue-500" />
              <span className="text-gray-600">Comfort (&gt;70%)</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
