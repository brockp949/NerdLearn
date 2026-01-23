import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MasteryGraph } from './MasteryGraph';

// Mock the ForceGraph2D component since it uses Canvas/WebGL
vi.mock('react-force-graph-2d', () => ({
    default: ({ graphData, nodeLabel, nodeColor }: any) => (
        <div data-testid="force-graph">
            <span>ForceGraph2D Mock</span>
            <div data-testid="graph-nodes">
                {graphData.nodes.map((node: any) => (
                    <div
                        key={node.id}
                        data-testid={`node-${node.id}`}
                        data-color={typeof nodeColor === 'function' ? nodeColor(node) : nodeColor}
                    >
                        {node[nodeLabel]}
                    </div>
                ))}
            </div>
        </div>
    ),
}));

// Mock the hook
const mockUseGraphData: {
    data: { nodes: any[]; links: any[] };
    loading: boolean;
    error: any;
} = {
    data: {
        nodes: [],
        links: []
    },
    loading: false,
    error: null
};

vi.mock('@/hooks/use-graph-data', () => ({
    useGraphData: () => mockUseGraphData
}));

describe('MasteryGraph', () => {
    beforeEach(() => {
        // Reset mock data
        mockUseGraphData.data = {
            nodes: [],
            links: []
        };
    });

    it('renders the container and legends', async () => {
        render(<MasteryGraph />);

        expect(screen.getByText('Mastered')).toBeDefined();
        expect(screen.getByText('In Progress')).toBeDefined();
        expect(await screen.findByTestId('force-graph')).toBeDefined();
    });

    it('renders nodes with correct colors based on mastery status', () => {
        mockUseGraphData.data = {
            nodes: [
                { id: '1', label: 'Concept A', mastered: true },
                { id: '2', label: 'Concept B', mastered: false }
            ],
            links: []
        };

        render(<MasteryGraph />);

        const nodeA = screen.getByTestId('node-1');
        const nodeB = screen.getByTestId('node-2');

        // Mastered should be emerald (#10b981)
        expect(nodeA.getAttribute('data-color')).toBe('#10b981');

        // In Progress should be blue (#3b82f6)
        expect(nodeB.getAttribute('data-color')).toBe('#3b82f6');
    });

    it('handles empty data gracefully', () => {
        mockUseGraphData.data = { nodes: [], links: [] };
        render(<MasteryGraph />);
        expect(screen.getByTestId('force-graph')).toBeDefined();
    });
});
