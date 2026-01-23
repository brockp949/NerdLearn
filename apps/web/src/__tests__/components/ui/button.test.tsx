import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { Button } from '@/components/ui/button'
import React from 'react'

// Mock the utils to avoid issues if path aliases aren't perfect in test env, 
// though usually vitest handles them. For robustness:
vi.mock('@/lib/utils', () => ({
    cn: (...inputs: any[]) => inputs.join(' '),
}))

describe('Button Component', () => {
    it('renders correctly with default props', () => {
        render(<Button>Click me</Button>)
        const button = screen.getByRole('button', { name: /click me/i })
        expect(button).toBeDefined()
        // Check for default variants based on cva string
        expect(button.className).toContain('bg-primary')
    })

    it('handles click events', () => {
        const handleClick = vi.fn()
        render(<Button onClick={handleClick}>Click me</Button>)

        const button = screen.getByRole('button', { name: /click me/i })
        fireEvent.click(button)

        expect(handleClick).toHaveBeenCalledTimes(1)
    })

    it('renders different variants', () => {
        const { rerender } = render(<Button variant="destructive">Destructive</Button>)
        expect(screen.getByRole('button').className).toContain('bg-destructive')

        rerender(<Button variant="outline">Outline</Button>)
        expect(screen.getByRole('button').className).toContain('border-input')
    })

    it('renders as a child element when asChild is true', () => {
        // Requires slightly more complex setup involving Radix Slot if testing "asChild" fully,
        // but here we verify it doesn't crash and renders the content.
        // For simple test, we just check data attribute or similar if applicable, 
        // but the Button component passes ...props.
        render(
            <Button asChild>
                <a href="/test">Link Button</a>
            </Button>
        )
        const link = screen.getByRole('link', { name: /link button/i })
        expect(link).toBeDefined()
        expect(link.getAttribute('href')).toBe('/test')
    })
})
