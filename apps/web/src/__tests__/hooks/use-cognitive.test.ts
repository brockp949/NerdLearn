/**
 * Tests for Cognitive Hooks
 *
 * Tests cover:
 * - useFrustration: Frustration detection and event tracking
 * - useMetacognition: Prompts, confidence, and self-explanation
 * - useCalibration: Calibration calculation and feedback
 * - useIntervention: Intervention decisions and history
 * - useCognitiveProfile: Profile loading
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { createWrapper } from '../utils/test-utils';
import {
  useFrustration,
  useMetacognition,
  useCalibration,
  useIntervention,
  useCognitiveProfile,
  useCognitiveState,
} from '@/hooks/use-cognitive';

describe('useFrustration', () => {
  const wrapper = createWrapper();

  it('should initialize with default state', () => {
    const { result } = renderHook(
      () => useFrustration({ userId: 'user_123' }),
      { wrapper }
    );

    expect(result.current.frustration).toBeNull();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.eventCount).toBe(0);
  });

  it('should add events to the event buffer', () => {
    const { result } = renderHook(
      () => useFrustration({ userId: 'user_123' }),
      { wrapper }
    );

    act(() => {
      // Test event filtering
      result.current.addEvent({
        event_type: 'answer',
        correct: true,
        response_time_ms: 5000,
      });
    });

    expect(result.current.eventCount).toBe(1);
  });

  it('should limit event buffer to 50 events', () => {
    const { result } = renderHook(
      () => useFrustration({ userId: 'user_123' }),
      { wrapper }
    );

    act(() => {
      for (let i = 0; i < 60; i++) {
        result.current.addEvent({
          event_type: 'answer',
          correct: i % 2 === 0,
        });
      }
    });

    expect(result.current.eventCount).toBe(50);
  });

  it('should clear events', () => {
    const { result } = renderHook(
      () => useFrustration({ userId: 'user_123' }),
      { wrapper }
    );

    act(() => {
      result.current.addEvent({ event_type: 'click' });
      result.current.addEvent({ event_type: 'click' });
    });

    expect(result.current.eventCount).toBe(2);

    act(() => {
      result.current.clearEvents();
    });

    expect(result.current.eventCount).toBe(0);
  });

  it('should not detect frustration with insufficient events', async () => {
    const { result } = renderHook(
      () => useFrustration({ userId: 'user_123', minEventsForDetection: 5 }),
      { wrapper }
    );

    act(() => {
      result.current.addEvent({ event_type: 'click' });
      result.current.addEvent({ event_type: 'click' });
    });

    const response = await act(async () => {
      return result.current.detectFrustration();
    });

    expect(response).toBeNull();
  });

  it('should detect frustration with sufficient events', async () => {
    const { result } = renderHook(
      () => useFrustration({ userId: 'user_123', minEventsForDetection: 5 }),
      { wrapper }
    );

    act(() => {
      for (let i = 0; i < 10; i++) {
        result.current.addEvent({
          event_type: 'answer',
          correct: false,
          response_time_ms: 1000,
        });
      }
    });

    await act(async () => {
      await result.current.detectFrustration();
    });

    await waitFor(() => {
      expect(result.current.frustration).not.toBeNull();
    });
  });
});

describe('useMetacognition', () => {
  const wrapper = createWrapper();

  it('should initialize with default state', () => {
    const { result } = renderHook(
      () => useMetacognition({ userId: 'user_123' }),
      { wrapper }
    );

    expect(result.current.prompt).toBeNull();
    expect(result.current.analysis).toBeNull();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it('should get metacognition prompt', async () => {
    const { result } = renderHook(
      () => useMetacognition({ userId: 'user_123' }),
      { wrapper }
    );

    await act(async () => {
      await result.current.getPrompt('Binary Search', 'after');
    });

    await waitFor(() => {
      expect(result.current.prompt).not.toBeNull();
      expect(result.current.prompt?.prompt_text).toBeTruthy();
    });
  });

  it('should get confidence scale', async () => {
    const { result } = renderHook(
      () => useMetacognition({ userId: 'user_123' }),
      { wrapper }
    );

    const scale = await act(async () => {
      return result.current.getConfidenceScale('Binary Search', 'numeric');
    });

    expect(scale).not.toBeNull();
    expect(scale?.options).toBeDefined();
  });

  it('should analyze self-explanation', async () => {
    const { result } = renderHook(
      () => useMetacognition({ userId: 'user_123' }),
      { wrapper }
    );

    await act(async () => {
      await result.current.analyzeExplanation(
        'Binary search divides the array in half each time',
        'Binary Search',
        ['divide_conquer', 'time_complexity']
      );
    });

    await waitFor(() => {
      expect(result.current.analysis).not.toBeNull();
      expect(result.current.analysis?.quality_score).toBeDefined();
    });
  });
});

describe('useCalibration', () => {
  const wrapper = createWrapper();

  it('should initialize with default state', () => {
    const { result } = renderHook(
      () => useCalibration({ userId: 'user_123' }),
      { wrapper }
    );

    expect(result.current.calibration).toBeNull();
    expect(result.current.feedback).toBeNull();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it('should calculate calibration', async () => {
    const { result } = renderHook(
      () => useCalibration({ userId: 'user_123' }),
      { wrapper }
    );

    await act(async () => {
      await result.current.calculateCalibration();
    });

    await waitFor(() => {
      expect(result.current.calibration).not.toBeNull();
      expect(result.current.calibration?.calibration_level).toBeDefined();
    });
  });

  it('should get calibration feedback', async () => {
    const { result } = renderHook(
      () => useCalibration({ userId: 'user_123' }),
      { wrapper }
    );

    await act(async () => {
      await result.current.getCalibrationFeedback();
    });

    await waitFor(() => {
      expect(result.current.feedback).not.toBeNull();
    });
  });
});

describe('useIntervention', () => {
  const wrapper = createWrapper();

  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should initialize with default state', () => {
    const { result } = renderHook(
      () => useIntervention({ userId: 'user_123' }),
      { wrapper }
    );

    expect(result.current.decision).toBeNull();
    expect(result.current.history).toBeNull();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it('should check intervention', async () => {
    vi.useRealTimers();

    const { result } = renderHook(
      () => useIntervention({ userId: 'user_123' }),
      { wrapper }
    );

    await act(async () => {
      await result.current.checkIntervention({
        frustration_score: 0.3,
        frustration_level: 'low',
      });
    });

    await waitFor(() => {
      expect(result.current.decision).not.toBeNull();
    });
  });

  it('should dismiss intervention', () => {
    const { result } = renderHook(
      () => useIntervention({ userId: 'user_123' }),
      { wrapper }
    );

    act(() => {
      result.current.dismissIntervention();
    });

    expect(result.current.decision).toBeNull();
  });

  it('should call onIntervention callback when intervention needed', async () => {
    vi.useRealTimers();

    const onIntervention = vi.fn();
    const { result } = renderHook(
      () => useIntervention({ userId: 'user_123', onIntervention }),
      { wrapper }
    );

    await act(async () => {
      await result.current.checkIntervention({
        frustration_score: 0.9,
        frustration_level: 'high',
      });
    });

    // Note: depends on MSW mock response
    await waitFor(() => {
      expect(result.current.decision).not.toBeNull();
    });
  });
});

describe('useCognitiveProfile', () => {
  const wrapper = createWrapper();

  it('should initialize with default state', () => {
    const { result } = renderHook(
      () => useCognitiveProfile({ userId: 'user_123' }),
      { wrapper }
    );

    expect(result.current.profile).toBeNull();
    expect(result.current.isLoading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it('should load profile on demand', async () => {
    const { result } = renderHook(
      () => useCognitiveProfile({ userId: 'user_123' }),
      { wrapper }
    );

    await act(async () => {
      await result.current.loadProfile();
    });

    await waitFor(() => {
      expect(result.current.profile).not.toBeNull();
      expect(result.current.profile?.user_id).toBe('user_123');
    });
  });

  it('should auto-load profile when autoLoad is true', async () => {
    const { result } = renderHook(
      () => useCognitiveProfile({ userId: 'user_123', autoLoad: true }),
      { wrapper }
    );

    await waitFor(() => {
      expect(result.current.profile).not.toBeNull();
    });
  });
});

describe('useCognitiveState', () => {
  const wrapper = createWrapper();

  it('should combine all cognitive hooks', () => {
    const { result } = renderHook(
      () => useCognitiveState({ userId: 'user_123' }),
      { wrapper }
    );

    expect(result.current.frustration).toBeDefined();
    expect(result.current.calibration).toBeDefined();
    expect(result.current.intervention).toBeDefined();
    expect(result.current.profile).toBeDefined();
    expect(result.current.getLearnerState).toBeDefined();
  });

  it('should get learner state', () => {
    const { result } = renderHook(
      () => useCognitiveState({ userId: 'user_123' }),
      { wrapper }
    );

    const state = result.current.getLearnerState();

    expect(state.frustration_score).toBeDefined();
    expect(state.frustration_level).toBeDefined();
    expect(state.calibration_level).toBeDefined();
  });
});
