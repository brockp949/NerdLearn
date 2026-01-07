'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { format, parseISO } from 'date-fns'

export interface ProgressData {
  date: string
  xp: number
  level?: number
}

export interface ProgressChartProps {
  data: ProgressData[]
  title?: string
  height?: number
}

export function ProgressChart({ data, title = '📈 Learning Progress', height = 300 }: ProgressChartProps) {
  // Format data for recharts
  const chartData = data.map(item => ({
    ...item,
    formattedDate: format(parseISO(item.date), 'MMM d')
  }))

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold mb-4">{title}</h3>

      {data.length === 0 ? (
        <div className="flex items-center justify-center h-64 text-gray-400">
          <div className="text-center">
            <p className="text-lg mb-2">📊</p>
            <p>No progress data yet</p>
            <p className="text-sm">Start learning to see your progress!</p>
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
              stroke="#6b7280"
              style={{ fontSize: '12px' }}
              label={{ value: 'Total XP', angle: -90, position: 'insideLeft', style: { fontSize: '12px' } }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                border: '1px solid #e5e7eb',
                borderRadius: '6px',
                fontSize: '12px'
              }}
              formatter={(value: number) => [`${value} XP`, 'Experience']}
              labelFormatter={(label) => `Date: ${label}`}
            />
            <Legend
              wrapperStyle={{ fontSize: '12px' }}
            />
            <Line
              type="monotone"
              dataKey="xp"
              stroke="#3b82f6"
              strokeWidth={2}
              name="Total XP"
              dot={{ fill: '#3b82f6', r: 4 }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      )}

      {/* Summary Stats */}
      {data.length > 0 && (
        <div className="mt-4 flex gap-6 text-sm text-gray-600">
          <div>
            <span className="font-medium">Total XP:</span> {data[data.length - 1]?.xp || 0}
          </div>
          <div>
            <span className="font-medium">Days Tracked:</span> {data.length}
          </div>
          {data.length > 1 && (
            <div>
              <span className="font-medium">Avg Growth:</span>{' '}
              {Math.round((data[data.length - 1].xp - data[0].xp) / (data.length - 1))} XP/day
            </div>
          )}
        </div>
      )}
    </div>
  )
}
