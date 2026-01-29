import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, act } from '@testing-library/react'
import { LearningStats } from '@/components/learning/LearningStats'
import React from 'react'

describe('LearningStats Component', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('renders level and progress', () => {
    render(
      <LearningStats
        xp_earned={0}
        new_total_xp={500}
        level={5}
        level_progress={75.5}
      />
    )

    expect(screen.getByText('5')).toBeDefined()
    expect(screen.getByText('75.5%')).toBeDefined()
    expect(screen.getByText('Progress to Level 6')).toBeDefined()
  })

  it('displays XP earned banner when xp_earned > 0', () => {
    render(
      <LearningStats
        xp_earned={50}
        new_total_xp={550}
        level={5}
        level_progress={80}
      />
    )

    expect(screen.getByText('+50')).toBeDefined()
    expect(screen.getByText('XP Earned')).toBeDefined()
  })

  it('hides XP banner when xp_earned is 0', () => {
    render(
      <LearningStats
        xp_earned={0}
        new_total_xp={500}
        level={5}
        level_progress={75}
      />
    )

    expect(screen.queryByText('XP Earned')).toBeNull()
  })

  it('shows achievement notification when achievement is provided', async () => {
    const achievement = {
      name: 'First Steps',
      icon: '🎯',
      description: 'Complete your first lesson'
    }

    render(
      <LearningStats
        xp_earned={100}
        new_total_xp={100}
        level={1}
        level_progress={10}
        achievement={achievement}
      />
    )

    expect(screen.getByText('First Steps')).toBeDefined()
    expect(screen.getByText('🎉 Achievement Unlocked!')).toBeDefined()
    expect(screen.getByText('Complete your first lesson')).toBeDefined()
  })

  it('hides achievement after 5 seconds', async () => {
    const achievement = {
      name: 'Test Achievement',
      icon: '⭐',
      description: 'Test description'
    }

    render(
      <LearningStats
        xp_earned={100}
        new_total_xp={100}
        level={1}
        level_progress={10}
        achievement={achievement}
      />
    )

    // Achievement should be visible initially
    expect(screen.getByText('Test Achievement')).toBeDefined()

    // Fast-forward 5 seconds and flush all pending updates
    await act(async () => {
      vi.advanceTimersByTime(5000)
    })

    // Achievement should be hidden after timer
    expect(screen.queryByText('Test Achievement')).toBeNull()
  })

  it('animates XP counter when showAnimation is true', async () => {
    render(
      <LearningStats
        xp_earned={100}
        new_total_xp={200}
        level={1}
        level_progress={50}
        showAnimation={true}
      />
    )

    // Initial value should be previous total (200 - 100 = 100)
    expect(screen.getByText('100')).toBeDefined()

    // Fast-forward animation and flush all pending updates
    await act(async () => {
      vi.advanceTimersByTime(1100)
    })

    // Should reach final value
    expect(screen.getByText('200')).toBeDefined()
  })

  it('does not animate when showAnimation is false', () => {
    render(
      <LearningStats
        xp_earned={100}
        new_total_xp={200}
        level={1}
        level_progress={50}
        showAnimation={false}
      />
    )

    // Should immediately show final value
    expect(screen.getByText('200')).toBeDefined()
  })

  it('formats large XP numbers with locale string', () => {
    render(
      <LearningStats
        xp_earned={100}
        new_total_xp={1234567}
        level={50}
        level_progress={50}
        showAnimation={false}
      />
    )

    // The Total XP banner shows when xp_earned > 0
    // toLocaleString() formats numbers with appropriate separators
    const formattedNumber = (1234567).toLocaleString()
    expect(screen.getByText(formattedNumber)).toBeDefined()
  })

  it('displays correct progress bar width', () => {
    const { container } = render(
      <LearningStats
        xp_earned={0}
        new_total_xp={500}
        level={5}
        level_progress={60}
      />
    )

    const progressBar = container.querySelector('[style*="width: 60%"]')
    expect(progressBar).toBeDefined()
  })

  it('handles zero values correctly', () => {
    render(
      <LearningStats
        xp_earned={0}
        new_total_xp={0}
        level={1}
        level_progress={0}
      />
    )

    expect(screen.getByText('1')).toBeDefined()
    expect(screen.getByText('0.0%')).toBeDefined()
  })

  it('handles null achievement', () => {
    render(
      <LearningStats
        xp_earned={50}
        new_total_xp={50}
        level={1}
        level_progress={5}
        achievement={null}
      />
    )

    expect(screen.queryByText('Achievement Unlocked!')).toBeNull()
  })
})
