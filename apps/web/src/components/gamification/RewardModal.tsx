
import React, { useEffect, useState } from 'react';
import { Reward, RewardFeedback } from '@/lib/gamification-client';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { motion, AnimatePresence } from 'framer-motion';
import confetti from 'canvas-confetti';

interface RewardModalProps {
  isOpen: boolean;
  onClose: () => void;
  reward?: Reward;
  feedback?: RewardFeedback;
}

const RARITY_COLORS = {
  common: 'bg-slate-100 border-slate-300 text-slate-800',
  rare: 'bg-blue-100 border-blue-300 text-blue-800',
  epic: 'bg-purple-100 border-purple-300 text-purple-800',
  legendary: 'bg-amber-100 border-amber-300 text-amber-800',
};

const RARITY_GLOW = {
  common: 'shadow-slate-400/50',
  rare: 'shadow-blue-400/50',
  epic: 'shadow-purple-400/50',
  legendary: 'shadow-amber-400/50',
};

export function RewardModal({ isOpen, onClose, reward, feedback }: RewardModalProps) {
  const [showContent, setShowContent] = useState(false);

  useEffect(() => {
    if (isOpen && reward) {
      // Trigger confetti based on rarity
      const particleCount = reward.rarity === 'legendary' ? 200 : reward.rarity === 'epic' ? 100 : 50;
      confetti({
        particleCount,
        spread: 70,
        origin: { y: 0.6 }
      });
      
      // Delay content slightly for "opening" effect
      const timer = setTimeout(() => setShowContent(true), 300);
      return () => clearTimeout(timer);
    } else {
      setShowContent(false);
    }
  }, [isOpen, reward]);

  if (!reward) return null;

  const rarityColor = RARITY_COLORS[reward.rarity] || RARITY_COLORS.common;
  const glowColor = RARITY_GLOW[reward.rarity] || RARITY_GLOW.common;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-md bg-white/95 backdrop-blur-sm border-2 border-primary/20">
        <DialogHeader>
          <DialogTitle className="text-center text-2xl font-bold bg-gradient-to-r from-primary to-purple-600 bg-clip-text text-transparent">
            {feedback?.message || 'Reward Unlocked!'}
          </DialogTitle>
        </DialogHeader>
        
        <div className="flex flex-col items-center justify-center p-6 space-y-6">
          <AnimatePresence>
            {showContent && (
              <motion.div
                initial={{ scale: 0.5, opacity: 0, rotate: -10 }}
                animate={{ scale: 1, opacity: 1, rotate: 0 }}
                transition={{ type: "spring", damping: 12 }}
                className={`relative w-32 h-32 flex items-center justify-center rounded-2xl border-4 ${rarityColor} ${glowColor} shadow-[0_0_30px_rgba(0,0,0,0.2)]`}
              >
                {/* Visual Icon based on type */}
                <span className="text-6xl filter drop-shadow-lg">
                  {reward.reward_type === 'xp' && '⚡'}
                  {reward.reward_type === 'streak_shield' && '🛡️'}
                  {reward.reward_type === 'badge' && '🏅'}
                  {reward.reward_type === 'cosmetic' && '🎨'}
                </span>
                
                {/* Rarity Badge */}
                <div className="absolute -top-3 bg-black/80 text-white text-xs px-3 py-1 rounded-full uppercase font-bold tracking-wider">
                  {reward.rarity}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <div className="text-center space-y-2">
            <h3 className="text-xl font-bold text-slate-800">{reward.name}</h3>
            <p className="text-slate-600">
              {reward.reward_type === 'xp' && `+${reward.value} XP Boost`}
              {reward.reward_type === 'streak_shield' && "Protects your streak for one missed day."}
              {reward.reward_type === 'badge' && "New badge added to your profile."}
            </p>
          </div>

          <Button 
            onClick={onClose}
            className="w-full bg-gradient-to-r from-primary to-purple-600 hover:from-primary/90 hover:to-purple-700 text-white font-bold py-6 rounded-xl shadow-lg transform transition-transform active:scale-95"
          >
            Claim Reward
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
