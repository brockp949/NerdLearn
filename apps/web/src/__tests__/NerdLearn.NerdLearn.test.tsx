import { describe, it, expect, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import React from 'react';
import { useClassifier, usePACER } from '@/hooks/use-pacer';
import { pacerApi } from '@/lib/api';

// Mock the API client
vi.mock('@/lib/api', () => ({
  pacerApi: {
    classify: vi.fn(),
    classifyTriage: vi.fn(),
    classifyBatch: vi.fn(),
    getProfile: vi.fn(),
    getAnalogy: vi.fn(),
    submitCritique: vi.fn(),
    getEvidenceForConcept: vi.fn(),
    autoLinkEvidence: vi.fn(),
    linkEvidence: vi.fn(),
    startProcedure: vi.fn(),
    completeStep: vi.fn(),
    getProceduralStatus: vi.fn(),
    getActiveProcedures: vi.fn(),
  },
}));

describe('NerdLearn.NerdLearn.tests', () => {
  describe('PACER Classifier Hook', () => {
    it('should classify content successfully', async () => {
      const mockResult = {
        pacerType: 'procedural',
        confidence: 0.95,
        reasoning: 'Contains steps',
        alternatives: [],
        recommendedAction: {
          action: 'Practice',
          description: 'Do it',
          tool: 'Sim',
        },
        contentHash: 'abc12345',
      };

      vi.mocked(pacerApi.classify).mockResolvedValue(mockResult as any);

      const { result } = renderHook(() => useClassifier());

      // Trigger classification
      await result.current.classify('Step 1: Do this');

      expect(pacerApi.classify).toHaveBeenCalledWith({
        content: 'Step 1: Do this',
        context: undefined,
      });
      
      await waitFor(() => {
         expect(result.current.result).toEqual(mockResult);
      });
      expect(result.current.error).toBeNull();
    });

    it('should handle API errors', async () => {
      vi.mocked(pacerApi.classify).mockRejectedValue(new Error('API Error'));

      const { result } = renderHook(() => useClassifier());

      await result.current.classify('Step 1: Do this');

      await waitFor(() => {
        expect(result.current.error).toBe('API Error');
      });
      expect(result.current.result).toBeNull();
    });
  });

  describe('PACER Integration', () => {
    it('should load user profile on mount', async () => {
      const mockProfile = {
        userId: 1,
        proceduralProficiency: 0.8,
        analogousProficiency: 0.5,
        conceptualProficiency: 0.6,
        evidenceProficiency: 0.4,
        referenceProficiency: 0.7,
        totalItemsProcessed: 10,
        preferredTypes: ['procedural'],
      };

      vi.mocked(pacerApi.getProfile).mockResolvedValue(mockProfile as any);

      const { result } = renderHook(() =>
        usePACER({ userId: 1, autoLoadProfile: true })
      );

      await waitFor(() => {
        expect(pacerApi.getProfile).toHaveBeenCalledWith(1);
        expect(result.current.profile.profile).toEqual(mockProfile);
      });
    });
  });
});
