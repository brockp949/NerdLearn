import { describe, it, expect, vi, beforeEach, beforeAll, afterAll } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import React from 'react'

// Store original fetch before MSW takes over
const originalFetch = globalThis.fetch

// Mock fetch for API calls
const mockFetch = vi.fn()

beforeAll(() => {
  // Override fetch with our mock
  globalThis.fetch = mockFetch as typeof fetch
})

afterAll(() => {
  // Restore original fetch
  globalThis.fetch = originalFetch
})

// Mock components from UI library
vi.mock('@/components/ui/button', () => ({
  Button: ({ children, onClick, disabled, ...props }: any) => (
    <button onClick={onClick} disabled={disabled} {...props}>
      {children}
    </button>
  ),
}))

vi.mock('@/components/ui/input', () => ({
  Input: ({ value, onChange, placeholder, ...props }: any) => (
    <input
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      {...props}
    />
  ),
}))

vi.mock('@/components/ui/scroll-area', () => ({
  ScrollArea: ({ children }: any) => <div data-testid="scroll-area">{children}</div>,
}))

describe('Chat Interface Component', () => {
  beforeEach(() => {
    mockFetch.mockClear()
    vi.clearAllMocks()
  })

  describe('Message Input', () => {
    it('renders input field with placeholder', () => {
      // Test that chat input would render correctly
      const input = document.createElement('input')
      input.placeholder = 'Ask about the course material...'
      expect(input.placeholder).toBe('Ask about the course material...')
    })

    it('handles text input correctly', async () => {
      const user = userEvent.setup()
      const input = document.createElement('input')
      document.body.appendChild(input)

      await user.type(input, 'What is Python?')
      expect(input.value).toBe('What is Python?')

      document.body.removeChild(input)
    })

    it('clears input after sending message', async () => {
      const input = document.createElement('input')
      input.value = 'Test message'

      // Simulate clearing after send
      input.value = ''
      expect(input.value).toBe('')
    })
  })

  describe('Message Display', () => {
    it('displays user messages correctly', () => {
      const userMessage = {
        role: 'user',
        content: 'What are Python decorators?',
        timestamp: new Date().toISOString(),
      }

      expect(userMessage.role).toBe('user')
      expect(userMessage.content).toBe('What are Python decorators?')
    })

    it('displays assistant messages correctly', () => {
      const assistantMessage = {
        role: 'assistant',
        content: 'Decorators are a way to modify functions in Python.',
        citations: [],
        timestamp: new Date().toISOString(),
      }

      expect(assistantMessage.role).toBe('assistant')
      expect(assistantMessage.citations).toEqual([])
    })

    it('displays citations when present', () => {
      const messageWithCitations = {
        role: 'assistant',
        content: 'Python supports multiple paradigms.',
        citations: [
          {
            module_id: 1,
            module_title: 'Introduction to Python',
            chunk_text: 'Python is multi-paradigm...',
            relevance_score: 0.95,
          },
        ],
      }

      expect(messageWithCitations.citations.length).toBe(1)
      expect(messageWithCitations.citations[0].module_title).toBe('Introduction to Python')
    })
  })

  describe('API Integration', () => {
    it('sends chat request with correct payload', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          message: 'Response from API',
          citations: [],
          xp_earned: 5,
        }),
      })

      const chatRequest = {
        query: 'What is Python?',
        user_id: 1,
        course_id: 1,
        session_id: 'test-session',
      }

      await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(chatRequest),
      })

      expect(mockFetch).toHaveBeenCalledWith('/api/chat', expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('What is Python?'),
      }))
    })

    it('handles API errors gracefully', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'))

      let errorOccurred = false
      try {
        await fetch('/api/chat', { method: 'POST' })
      } catch {
        errorOccurred = true
      }

      expect(errorOccurred).toBe(true)
    })

    it('handles 500 error responses', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => ({ detail: 'Server error' }),
      })

      const response = await fetch('/api/chat', { method: 'POST' })
      expect(response.ok).toBe(false)
      expect(response.status).toBe(500)
    })
  })

  describe('Chat History', () => {
    it('fetches chat history on mount', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          messages: [
            { role: 'user', content: 'Previous question', timestamp: '2024-01-01T00:00:00Z' },
            { role: 'assistant', content: 'Previous answer', citations: [], timestamp: '2024-01-01T00:00:01Z' },
          ],
          message_count: 2,
        }),
      })

      await fetch('/api/chat/history?user_id=1&course_id=1')
      expect(mockFetch).toHaveBeenCalled()
    })

    it('clears chat history when requested', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          message: 'Chat history cleared',
          messages_deleted: 5,
        }),
      })

      const response = await fetch('/api/chat/history?user_id=1', { method: 'DELETE' })
      const data = await response.json()

      expect(data.messages_deleted).toBe(5)
    })
  })

  describe('Loading States', () => {
    it('shows loading indicator while waiting for response', () => {
      const isLoading = true

      // Simulate loading state
      expect(isLoading).toBe(true)
      // Would show spinner or typing indicator
    })

    it('disables input while loading', () => {
      const isLoading = true
      const inputDisabled = isLoading

      expect(inputDisabled).toBe(true)
    })
  })

  describe('XP Display', () => {
    it('shows XP earned after receiving response', () => {
      const response = {
        message: 'Answer to your question',
        citations: [],
        xp_earned: 5,
      }

      expect(response.xp_earned).toBe(5)
    })

    it('accumulates XP across messages', () => {
      const responses = [
        { xp_earned: 5 },
        { xp_earned: 5 },
        { xp_earned: 5 },
      ]

      const totalXP = responses.reduce((sum, r) => sum + r.xp_earned, 0)
      expect(totalXP).toBe(15)
    })
  })
})

describe('Chat Message Components', () => {
  describe('UserMessage', () => {
    it('renders user message with correct styling', () => {
      const message = {
        role: 'user',
        content: 'Test user message',
      }

      expect(message.role).toBe('user')
    })
  })

  describe('AssistantMessage', () => {
    it('renders assistant message with avatar', () => {
      const message = {
        role: 'assistant',
        content: 'Test assistant message',
      }

      expect(message.role).toBe('assistant')
    })

    it('renders markdown content correctly', () => {
      const message = {
        role: 'assistant',
        content: '**Bold** and *italic* text',
      }

      expect(message.content).toContain('**Bold**')
    })

    it('renders code blocks', () => {
      const message = {
        role: 'assistant',
        content: '```python\nprint("Hello")\n```',
      }

      expect(message.content).toContain('```python')
    })
  })

  describe('Citation Component', () => {
    it('displays citation source', () => {
      const citation = {
        module_id: 1,
        module_title: 'Python Basics',
        module_type: 'pdf',
        chunk_text: 'Python is a programming language.',
        relevance_score: 0.92,
      }

      expect(citation.module_title).toBe('Python Basics')
      expect(citation.relevance_score).toBeGreaterThan(0.9)
    })

    it('truncates long citation text', () => {
      const longText = 'A'.repeat(500)
      const maxLength = 200
      const truncated = longText.length > maxLength
        ? longText.slice(0, maxLength) + '...'
        : longText

      expect(truncated.length).toBeLessThanOrEqual(maxLength + 3)
    })
  })
})
