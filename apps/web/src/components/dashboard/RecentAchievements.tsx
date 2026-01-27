'use client'

import { motion } from 'framer-motion'

export interface Achievement {
  id: string
  name: string
  icon: string
  description: string
  unlockedAt: string
  rarity: 'common' | 'rare' | 'epic' | 'legendary'
}

interface RecentAchievementsProps {
  achievements: Achievement[]
}

const RARITY_COLORS = {
  common: 'bg-slate-100 text-slate-600',
  rare: 'bg-blue-100 text-blue-600',
  epic: 'bg-purple-100 text-purple-600',
  legendary: 'bg-amber-100 text-amber-600',
}

export function RecentAchievements({ achievements }: RecentAchievementsProps) {
  if (achievements.length === 0) return null

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
        <span>🏆</span> Recent Achievements
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {achievements.map((achievement, index) => (
          <motion.div
            key={achievement.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="flex items-center p-3 border border-gray-100 rounded-xl bg-gray-50/50 hover:bg-gray-50 transition-colors group"
          >
            <div className="text-3xl mr-3 group-hover:scale-110 transition-transform">
              {achievement.icon}
            </div>
            <div className="flex-1 min-w-0">
              <h4 className="text-sm font-bold text-gray-900 truncate">{achievement.name}</h4>
              <p className="text-xs text-gray-500 truncate">{achievement.description}</p>
              <div className="mt-1 flex items-center gap-2">
                <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wider ${RARITY_COLORS[achievement.rarity]}`}>
                  {achievement.rarity}
                </span>
                <span className="text-[10px] text-gray-400">
                  {new Date(achievement.unlockedAt).toLocaleDateString()}
                </span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
      <button className="w-full mt-4 py-2 text-sm font-medium text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-lg transition-colors">
        View All Achievements
      </button>
    </div>
  )
}
