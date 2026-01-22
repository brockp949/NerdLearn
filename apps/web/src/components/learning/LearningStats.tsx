'use client'

import { useEffect, useState } from 'react'

interface Achievement {
  name: string
  icon: string
  description: string
}

interface LearningStatsProps {
  xp_earned: number
  new_total_xp: number
  level: number
  level_progress: number
  achievement?: Achievement | null
  showAnimation?: boolean
}

export function LearningStats({
  xp_earned,
  new_total_xp,
  level,
  level_progress,
  achievement,
  showAnimation = false
}: LearningStatsProps) {
  const [showAchievement, setShowAchievement] = useState(false)
  const [animatedXP, setAnimatedXP] = useState(new_total_xp - xp_earned)

  useEffect(() => {
    if (showAnimation && xp_earned > 0) {
      // Animate XP counter
      const duration = 1000
      const steps = 20
      const increment = xp_earned / steps
      let current = new_total_xp - xp_earned

      const timer = setInterval(() => {
        current += increment
        if (current >= new_total_xp) {
          setAnimatedXP(new_total_xp)
          clearInterval(timer)
        } else {
          setAnimatedXP(Math.floor(current))
        }
      }, duration / steps)

      return () => clearInterval(timer)
    } else {
      setAnimatedXP(new_total_xp)
    }
  }, [xp_earned, new_total_xp, showAnimation])

  useEffect(() => {
    if (achievement) {
      setShowAchievement(true)
      const timer = setTimeout(() => setShowAchievement(false), 5000)
      return () => clearTimeout(timer)
    }
  }, [achievement])

  return (
    <div className="space-y-4">
      {/* XP Gained Banner */}
      {xp_earned > 0 && (
        <div className="bg-gradient-to-r from-yellow-400 to-orange-500 text-white rounded-lg p-4 shadow-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <span className="text-3xl">⚡</span>
              <div>
                <p className="text-sm font-medium opacity-90">XP Earned</p>
                <p className="text-2xl font-bold">+{xp_earned}</p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm opacity-90">Total XP</p>
              <p className="text-xl font-bold">{animatedXP.toLocaleString()}</p>
            </div>
          </div>
        </div>
      )}

      {/* Level Progress */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-3">
            <span className="text-3xl">🎯</span>
            <div>
              <p className="text-sm text-gray-600">Current Level</p>
              <p className="text-3xl font-bold text-gray-900">{level}</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-600">Progress to Level {level + 1}</p>
            <p className="text-lg font-semibold text-purple-600">{level_progress.toFixed(1)}%</p>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-gray-200 rounded-full h-3">
          <div
            className="bg-gradient-to-r from-purple-500 to-blue-500 h-3 rounded-full transition-all duration-1000 ease-out"
            style={{ width: `${level_progress}%` }}
          />
        </div>
      </div>

      {/* Achievement Notification */}
      {showAchievement && achievement && (
        <div className="bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg p-6 shadow-xl border-4 border-yellow-400 animate-bounce">
          <div className="flex items-center space-x-4">
            <span className="text-5xl">{achievement.icon}</span>
            <div className="flex-1">
              <p className="text-sm font-medium opacity-90 mb-1">🎉 Achievement Unlocked!</p>
              <p className="text-2xl font-bold mb-1">{achievement.name}</p>
              <p className="text-sm opacity-90">{achievement.description}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
