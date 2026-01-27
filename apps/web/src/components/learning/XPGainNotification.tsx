'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { useEffect, useState } from 'react'
import confetti from 'canvas-confetti'

export type AgeGroup = 'early_childhood' | 'middle_childhood' | 'adolescence' | 'adult'

interface XPGainNotificationProps {
  xp: number
  levelUp?: boolean
  newLevel?: number
  ageGroup: AgeGroup
  onComplete?: () => void
}

export function XPGainNotification({ xp, levelUp, newLevel, ageGroup, onComplete }: XPGainNotificationProps) {
  const [visible, setVisible] = useState(true)

  useEffect(() => {
    if (xp > 0 || levelUp) {
      if (ageGroup === 'early_childhood' || levelUp) {
        confetti({
          particleCount: levelUp ? 150 : 40,
          spread: levelUp ? 100 : 60,
          colors: levelUp ? ['#FFD700', '#FF69B4', '#00BFFF', '#ADFF2F'] : ['#FFD700', '#FF69B4', '#00BFFF'],
          origin: { y: 0.8 }
        })
      }
      
      const timer = setTimeout(() => {
        setVisible(false)
        if (onComplete) onComplete()
      }, levelUp ? 5000 : 3000)
      return () => clearTimeout(timer)
    }
  }, [xp, levelUp, ageGroup, onComplete])

  return (
    <AnimatePresence>
      {visible && (xp > 0 || levelUp) && (
        <motion.div
          initial={{ opacity: 0, y: 50, scale: 0.5 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -50, scale: 0.5 }}
          className="fixed bottom-24 left-1/2 transform -translate-x-1/2 z-50"
        >
          {levelUp ? (
            <div className={`
              ${ageGroup === 'early_childhood' ? 'bg-gradient-to-r from-yellow-400 via-pink-500 to-purple-600' : 
                ageGroup === 'adolescence' ? 'bg-slate-900 border-cyan-500 shadow-cyan-500/50' : 
                'bg-white'} 
              px-10 py-6 rounded-2xl shadow-2xl border-4 border-white flex flex-col items-center gap-2
            `}>
              <span className="text-6xl">🎊</span>
              <h2 className={`text-3xl font-black ${ageGroup === 'early_childhood' ? 'text-white' : ageGroup === 'adolescence' ? 'text-cyan-400' : 'text-gray-900'}`}>
                LEVEL UP!
              </h2>
              <p className={`text-xl font-bold ${ageGroup === 'early_childhood' ? 'text-white' : ageGroup === 'adolescence' ? 'text-white' : 'text-blue-600'}`}>
                You reached Level {newLevel}
              </p>
            </div>
          ) : ageGroup === 'early_childhood' ? (
             <div className="bg-gradient-to-r from-yellow-400 to-pink-500 text-white px-8 py-4 rounded-full shadow-2xl border-4 border-white flex items-center gap-3">
               <span className="text-4xl animate-bounce">🌟</span>
               <div className="flex flex-col">
                 <span className="text-xl font-black tracking-widest">+{xp} XP</span>
                 <span className="text-xs font-bold uppercase">Super Star!</span>
               </div>
               <span className="text-4xl animate-bounce">✨</span>
             </div>
          ) : ageGroup === 'adolescence' ? (
             <div className="bg-slate-900 text-cyan-400 px-6 py-3 rounded-lg shadow-cyan-500/20 shadow-lg border border-cyan-500/50 flex items-center gap-4 font-mono">
               <div className="h-2 w-2 bg-cyan-500 animate-pulse rounded-full" />
               <span className="text-lg tracking-tighter">NODE_XP_INCREMENT: <span className="text-white">+{xp}</span></span>
               <div className="h-2 w-2 bg-cyan-500 animate-pulse rounded-full" />
             </div>
          ) : (
             <div className="bg-white text-gray-900 px-6 py-3 rounded-xl shadow-xl border border-gray-100 flex items-center gap-3">
               <span className="text-2xl text-yellow-500">⚡</span>
               <span className="text-lg font-bold">+{xp} XP Earned</span>
             </div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  )
}
