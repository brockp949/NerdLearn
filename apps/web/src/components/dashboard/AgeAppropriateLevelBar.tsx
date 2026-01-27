'use client'

import { motion } from 'framer-motion'

export type AgeGroup = 'early_childhood' | 'middle_childhood' | 'adolescence' | 'adult'

interface AgeAppropriateLevelBarProps {
  level: number
  progress: number // 0-100
  ageGroup: AgeGroup
}

export function AgeAppropriateLevelBar({ level, progress, ageGroup }: AgeAppropriateLevelBarProps) {
  // Early childhood (3-7): More playful, bouncy, stars
  if (ageGroup === 'early_childhood') {
    return (
      <div className="space-y-2">
        <div className="flex justify-between items-end">
          <motion.div 
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="text-4xl"
          >
            🌟
          </motion.div>
          <span className="text-xl font-bold text-yellow-500">Level {level}</span>
        </div>
        <div className="relative h-6 bg-yellow-100 rounded-full overflow-hidden border-2 border-yellow-200">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 1, ease: 'backOut' }}
            className="h-full bg-gradient-to-r from-yellow-400 to-orange-400"
          />
          {progress > 10 && (
             <motion.div 
               animate={{ x: [0, 5, 0] }}
               transition={{ duration: 1, repeat: Infinity }}
               className="absolute top-1 left-2 text-[10px]"
             >
               ✨
             </motion.div>
          )}
        </div>
        <p className="text-xs text-center text-orange-600 font-medium">You're doing amazing! 🎈</p>
      </div>
    )
  }

  // Middle childhood (8-12): Colorful, energetic, more structured
  if (ageGroup === 'middle_childhood') {
    return (
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <span className="text-sm font-bold text-blue-600 uppercase tracking-wider">Level {level}</span>
          <span className="text-xs font-mono text-blue-400">{progress}% to Next Level</span>
        </div>
        <div className="h-4 bg-blue-50 rounded-lg overflow-hidden border border-blue-100 p-0.5">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 1, ease: 'easeOut' }}
            className="h-full bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 rounded-md"
          />
        </div>
      </div>
    )
  }

  // Adolescence (13-18): Sleek, minimalist, high-tech
  if (ageGroup === 'adolescence') {
    return (
      <div className="space-y-2">
        <div className="flex justify-between items-baseline">
          <h4 className="text-lg font-light text-slate-300">CORE LEVEL <span className="font-bold text-white">{level}</span></h4>
          <span className="text-[10px] font-mono text-cyan-500">{progress}% COMPLETE</span>
        </div>
        <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 1.5, ease: 'circOut' }}
            className="h-full bg-cyan-500 shadow-[0_0_10px_rgba(6,182,212,0.5)]"
          />
        </div>
      </div>
    )
  }

  // Adult (18+): Professional, clean, data-driven
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-xs text-slate-500">
        <span>Level {level}</span>
        <span>{progress}%</span>
      </div>
      <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          className="h-full bg-slate-800"
        />
      </div>
    </div>
  )
}
