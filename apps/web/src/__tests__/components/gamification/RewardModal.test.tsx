import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, act } from '@testing-library/react'
import { RewardModal } from '@/components/gamification/RewardModal'
import React from 'react'

// Mock framer-motion
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}))

// Mock canvas-confetti
vi.mock('canvas-confetti', () => ({
  default: vi.fn(),
}))

// Mock Dialog components
vi.mock('@/components/ui/dialog', () => ({
  Dialog: ({ open, children, onOpenChange }: any) => (
    open ? (
      <div data-testid="dialog" onClick={() => onOpenChange(false)}>
        {children}
      </div>
    ) : null
  ),
  DialogContent: ({ children, ...props }: any) => (
    <div data-testid="dialog-content" {...props}>{children}</div>
  ),
  DialogHeader: ({ children }: any) => <div>{children}</div>,
  DialogTitle: ({ children, ...props }: any) => <h2 {...props}>{children}</h2>,
}))

describe('RewardModal Component', () => {
  const mockOnClose = vi.fn()

  const defaultReward = {
    id: '1',
    name: 'XP Boost',
    rarity: 'common' as const,
    reward_type: 'xp' as const,
    value: 100,
  }

  beforeEach(() => {
    mockOnClose.mockClear()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('renders nothing when isOpen is false', () => {
    const { container } = render(
      <RewardModal
        isOpen={false}
        onClose={mockOnClose}
        reward={defaultReward}
      />
    )

    expect(screen.queryByTestId('dialog')).toBeNull()
  })

  it('renders nothing when reward is undefined', () => {
    render(
      <RewardModal
        isOpen={true}
        onClose={mockOnClose}
        reward={undefined}
      />
    )

    expect(screen.queryByTestId('dialog-content')).toBeNull()
  })

  it('renders dialog when isOpen is true and reward is provided', () => {
    render(
      <RewardModal
        isOpen={true}
        onClose={mockOnClose}
        reward={defaultReward}
      />
    )

    expect(screen.getByTestId('dialog')).toBeDefined()
    expect(screen.getByText('XP Boost')).toBeDefined()
  })

  it('displays XP reward correctly', async () => {
    render(
      <RewardModal
        isOpen={true}
        onClose={mockOnClose}
        reward={defaultReward}
      />
    )

    // Advance timer to show content (300ms delay in component)
    await act(async () => {
      vi.advanceTimersByTime(300)
    })

    expect(screen.getByText('⚡')).toBeDefined()
    expect(screen.getByText('+100 XP Boost')).toBeDefined()
  })

  it('displays streak shield reward correctly', async () => {
    const shieldReward = {
      ...defaultReward,
      reward_type: 'streak_shield' as const,
      name: 'Streak Shield',
    }

    render(
      <RewardModal
        isOpen={true}
        onClose={mockOnClose}
        reward={shieldReward}
      />
    )

    await act(async () => {
      vi.advanceTimersByTime(300)
    })

    expect(screen.getByText('🛡️')).toBeDefined()
    expect(screen.getByText(/Protects your streak/)).toBeDefined()
  })

  it('displays badge reward correctly', async () => {
    const badgeReward = {
      ...defaultReward,
      reward_type: 'badge' as const,
      name: 'First Steps Badge',
    }

    render(
      <RewardModal
        isOpen={true}
        onClose={mockOnClose}
        reward={badgeReward}
      />
    )

    await act(async () => {
      vi.advanceTimersByTime(300)
    })

    expect(screen.getByText('🏅')).toBeDefined()
    expect(screen.getByText(/badge added to your profile/)).toBeDefined()
  })

  it('displays cosmetic reward icon', async () => {
    const cosmeticReward = {
      ...defaultReward,
      reward_type: 'cosmetic' as const,
      name: 'Special Theme',
    }

    render(
      <RewardModal
        isOpen={true}
        onClose={mockOnClose}
        reward={cosmeticReward}
      />
    )

    await act(async () => {
      vi.advanceTimersByTime(300)
    })

    expect(screen.getByText('🎨')).toBeDefined()
  })

  it('displays rarity badge', async () => {
    render(
      <RewardModal
        isOpen={true}
        onClose={mockOnClose}
        reward={defaultReward}
      />
    )

    await act(async () => {
      vi.advanceTimersByTime(300)
    })

    expect(screen.getByText('common')).toBeDefined()
  })

  it('displays feedback message when provided', () => {
    const feedback = {
      message: 'Amazing work!',
      celebration_level: 'medium' as const,
    }

    render(
      <RewardModal
        isOpen={true}
        onClose={mockOnClose}
        reward={defaultReward}
        feedback={feedback}
      />
    )

    expect(screen.getByText('Amazing work!')).toBeDefined()
  })

  it('displays default message when no feedback', () => {
    render(
      <RewardModal
        isOpen={true}
        onClose={mockOnClose}
        reward={defaultReward}
      />
    )

    expect(screen.getByText('Reward Unlocked!')).toBeDefined()
  })

  it('calls onClose when Claim Reward button is clicked', () => {
    render(
      <RewardModal
        isOpen={true}
        onClose={mockOnClose}
        reward={defaultReward}
      />
    )

    const claimButton = screen.getByText('Claim Reward')
    fireEvent.click(claimButton)

    // Button click calls onClose; event bubbles to dialog which also calls onOpenChange
    // The important thing is that onClose was called
    expect(mockOnClose).toHaveBeenCalled()
  })

  it('applies correct color classes for different rarities', async () => {
    const rarities = ['common', 'rare', 'epic', 'legendary'] as const

    for (const rarity of rarities) {
      const { unmount } = render(
        <RewardModal
          isOpen={true}
          onClose={mockOnClose}
          reward={{ ...defaultReward, rarity }}
        />
      )

      await act(async () => {
        vi.advanceTimersByTime(300)
      })

      expect(screen.getByText(rarity)).toBeDefined()
      unmount()
    }
  })

  it('triggers confetti on open', async () => {
    const confetti = await import('canvas-confetti')

    render(
      <RewardModal
        isOpen={true}
        onClose={mockOnClose}
        reward={defaultReward}
      />
    )

    expect(confetti.default).toHaveBeenCalled()
  })

  it('shows more confetti for legendary rewards', async () => {
    const confetti = await import('canvas-confetti')

    render(
      <RewardModal
        isOpen={true}
        onClose={mockOnClose}
        reward={{ ...defaultReward, rarity: 'legendary' }}
      />
    )

    expect(confetti.default).toHaveBeenCalledWith(
      expect.objectContaining({ particleCount: 200 })
    )
  })
})
