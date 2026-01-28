import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { testingApi } from '@/lib/api';
import { TestSummary, TestSuite, FailedTestsResponse } from '@/types/testing';

export function useTestSummary() {
    const { data, isLoading, error, refetch } = useQuery({
        queryKey: ['tests', 'summary'],
        queryFn: async (): Promise<TestSummary> => {
            return testingApi.getSummary();
        },
        staleTime: 10 * 1000, // 10 seconds - refresh more often for real-time updates
        gcTime: 5 * 60 * 1000, // 5 minutes
        refetchInterval: (query) => {
            // Refetch more frequently if tests are running
            const data = query.state.data as TestSummary | undefined;
            return data?.running ? 2000 : 30000; // 2s while running, 30s otherwise
        },
    });

    return {
        data: data ?? {
            totalSuites: 0,
            totalTests: 0,
            passed: 0,
            failed: 0,
            skipped: 0,
            errors: 0,
            passRate: 0,
            duration: 0,
            lastRun: new Date().toISOString(),
            suites: [],
            running: false,
        },
        loading: isLoading,
        error: error instanceof Error ? error : null,
        refetch,
    };
}

export function useTestSuite(suiteId: string) {
    const { data, isLoading, error } = useQuery({
        queryKey: ['tests', 'suite', suiteId],
        queryFn: async (): Promise<TestSuite> => {
            return testingApi.getSuite(suiteId);
        },
        enabled: !!suiteId,
        staleTime: 30 * 1000,
    });

    return {
        data,
        loading: isLoading,
        error: error instanceof Error ? error : null,
    };
}

export function useFailedTests() {
    const { data, isLoading, error, refetch } = useQuery({
        queryKey: ['tests', 'failed'],
        queryFn: async (): Promise<FailedTestsResponse> => {
            return testingApi.getFailedTests();
        },
        staleTime: 10 * 1000,
    });

    return {
        data: data ?? { tests: [], total: 0, message: '' },
        loading: isLoading,
        error: error instanceof Error ? error : null,
        refetch,
    };
}

export function useRunTests() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: async ({ suiteId, testPath }: { suiteId?: string; testPath?: string } = {}) => {
            return testingApi.runTests(suiteId, testPath);
        },
        onSuccess: () => {
            // Invalidate queries to trigger refresh
            queryClient.invalidateQueries({ queryKey: ['tests'] });
        },
    });
}

export function useClearTestCache() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: async () => {
            return testingApi.clearCache();
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['tests'] });
        },
    });
}
