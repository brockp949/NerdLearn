import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QuickStats, QuickStatsData } from '@/components/dashboard/QuickStats'
import React from 'react'

// Mock the AgeAppropriateLevelBar component
vi.mock('@/components/dashboard/AgeAppropriateLevelBar', () => ({
  AgeAppropriateLevelBar: ({ level, progress }: { level: number; progress: number }) => (
    <div data-testid="age-appropriate-level-bar">
      Level: {level}, Progress: {progress}%
    </div>
  ),
}))

describe('QuickStats Component', () => {
  const defaultStats: QuickStatsData = {
    level: 10,
    totalXP: 5000,
    xpToNextLevel: 200,
    levelProgress: 75,
    currentStreak: 7,
    streakShields: 1,
    cardsReviewed: 150,
    conceptsMastered: 25,
  }

  it('renders all stat cards', () => {
    render(<QuickStats stats={defaultStats} />)

    expect(screen.getByText('Level')).toBeDefined()
    expect(screen.getByText('Total XP')).toBeDefined()
    expect(screen.getByText('Current Streak')).toBeDefined()
    expect(screen.getByText('Cards Reviewed')).toBeDefined()
    expect(screen.getByText('Concepts Mastered')).toBeDefined()
    expect(screen.getByText('Success Rate')).toBeDefined()
  })

  it('displays correct level', () => {
    render(<QuickStats stats={defaultStats} />)

    expect(screen.getByText('10')).toBeDefined()
    expect(screen.getByText('200 XP to next level')).toBeDefined()
  })

  it('displays formatted total XP', () => {
    render(<QuickStats stats={defaultStats} />)

    expect(screen.getByText('5,000')).toBeDefined()
  })

  it('displays current streak with days', () => {
    render(<QuickStats stats={defaultStats} />)

    expect(screen.getByText('7 days')).toBeDefined()
  })

  it('shows streak shields when available', () => {
    render(<QuickStats stats={defaultStats} />)

    expect(screen.getByText(/1 Shield.*Active/i)).toBeDefined()
  })

  it('shows "Keep it up!" when no streak shields', () => {
    const statsNoShields = { ...defaultStats, streakShields: 0 }
    render(<QuickStats stats={statsNoShields} />)

    expect(screen.getByText('Keep it up!')).toBeDefined()
  })

  it('displays cards reviewed count', () => {
    render(<QuickStats stats={defaultStats} />)

    expect(screen.getByText('150')).toBeDefined()
  })

  it('displays concepts mastered count', () => {
    render(<QuickStats stats={defaultStats} />)

    expect(screen.getByText('25')).toBeDefined()
  })

  it('renders AgeAppropriateLevelBar with correct props', () => {
    render(<QuickStats stats={defaultStats} ageGroup="teen" />)

    const levelBar = screen.getByTestId('age-appropriate-level-bar')
    expect(levelBar).toBeDefined()
    expect(levelBar.textContent).toContain('Level: 10')
    expect(levelBar.textContent).toContain('Progress: 75%')
  })

  it('uses adult ageGroup by default', () => {
    render(<QuickStats stats={defaultStats} />)

    // The component should render without errors with default ageGroup
    expect(screen.getByText('Level')).toBeDefined()
  })

  it('handles large numbers correctly', () => {
    const largeStats: QuickStatsData = {
      level: 100,
      totalXP: 1000000,
      xpToNextLevel: 5000,
      levelProgress: 50,
      currentStreak: 365,
      streakShields: 5,
      cardsReviewed: 10000,
      conceptsMastered: 500,
    }

    render(<QuickStats stats={largeStats} />)

    expect(screen.getByText('1,000,000')).toBeDefined()
    expect(screen.getByText('365 days')).toBeDefined()
  })

  it('handles zero values correctly', () => {
    const zeroStats: QuickStatsData = {
      level: 1,
      totalXP: 0,
      xpToNextLevel: 100,
      levelProgress: 0,
      currentStreak: 0,
      streakShields: 0,
      cardsReviewed: 0,
      conceptsMastered: 0,
    }

    render(<QuickStats stats={zeroStats} />)

    expect(screen.getByText('0 days')).toBeDefined()
  })

  it('renders with responsive grid layout', () => {
    const { container } = render(<QuickStats stats={defaultStats} />)

    const grid = container.querySelector('.grid')
    expect(grid?.className).toContain('grid-cols-1')
    expect(grid?.className).toContain('sm:grid-cols-2')
    expect(grid?.className).toContain('lg:grid-cols-3')
  })
})

describe('StatCard rendering', () => {
  const defaultStats: QuickStatsData = {
    level: 5,
    totalXP: 1000,
    xpToNextLevel: 500,
    levelProgress: 50,
    currentStreak: 3,
    streakShields: 0,
    cardsReviewed: 50,
    conceptsMastered: 10,
  }

  it('renders icons for all stats', () => {
    render(<QuickStats stats={defaultStats} />)

    // Check for emoji icons
    expect(screen.getByText('⬆️')).toBeDefined()
    expect(screen.getByText('📊')).toBeDefined()
    expect(screen.getByText('🔥')).toBeDefined()
    expect(screen.getByText('📝')).toBeDefined()
    expect(screen.getByText('✅')).toBeDefined()
    expect(screen.getByText('🎯')).toBeDefined()
  })

  it('renders subtitles correctly', () => {
    render(<QuickStats stats={defaultStats} />)

    expect(screen.getByText('Keep learning!')).toBeDefined()
    expect(screen.getByText('Total reviews completed')).toBeDefined()
    expect(screen.getByText('≥80% mastery')).toBeDefined()
    expect(screen.getByText('Across all cards')).toBeDefined()
  })
})
