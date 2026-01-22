/**
 * Test Utilities
 *
 * Custom render function and utilities for testing React components
 */

import React, { ReactElement } from 'react';
import { render, RenderOptions, RenderResult } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Create a custom QueryClient for tests
function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
        staleTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });
}

interface AllTheProvidersProps {
  children: React.ReactNode;
}

function AllTheProviders({ children }: AllTheProvidersProps) {
  const queryClient = createTestQueryClient();

  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
}

// Custom render function that wraps with providers
function customRender(
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
): RenderResult {
  return render(ui, { wrapper: AllTheProviders, ...options });
}

// Create a wrapper for testing hooks with React Query
export function createWrapper() {
  const queryClient = createTestQueryClient();
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    );
  };
}

// Re-export everything from testing-library
export * from '@testing-library/react';
export { userEvent } from '@testing-library/user-event';

// Export custom render
export { customRender as render };

// Utility to wait for async operations
export async function waitForLoadingToFinish() {
  const loadingElements = document.querySelectorAll('[aria-busy="true"]');
  await Promise.all(
    Array.from(loadingElements).map((element) =>
      new Promise<void>((resolve) => {
        const observer = new MutationObserver(() => {
          if (element.getAttribute('aria-busy') !== 'true') {
            observer.disconnect();
            resolve();
          }
        });
        observer.observe(element, { attributes: true });
      })
    )
  );
}

// Mock user data factory
export function createMockUser(overrides = {}) {
  return {
    id: 'user_123',
    email: 'test@example.com',
    name: 'Test User',
    ...overrides,
  };
}

// Mock concept data factory
export function createMockConcept(overrides = {}) {
  return {
    id: 'concept_123',
    name: 'Binary Search',
    difficulty: 5.0,
    mastery_level: 0.5,
    ...overrides,
  };
}

// Mock session data factory
export function createMockTeachingSession(overrides = {}) {
  return {
    session_id: 'session_123',
    user_id: 'user_123',
    concept_id: 'concept_123',
    concept_name: 'Binary Search',
    persona: 'curious',
    comprehension_level: 0.0,
    exchanges: [],
    completed: false,
    ...overrides,
  };
}

// Mock debate session factory
export function createMockDebateSession(overrides = {}) {
  return {
    session_id: 'debate_123',
    topic: 'AI in Education',
    format: 'roundtable',
    agents: [
      { agent_id: 'agent_1', name: 'Alex', role: 'advocate' },
      { agent_id: 'agent_2', name: 'Jordan', role: 'skeptic' },
    ],
    contributions: [],
    current_round: 1,
    max_rounds: 3,
    completed: false,
    ...overrides,
  };
}
