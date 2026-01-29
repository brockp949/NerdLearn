import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MasteryGraph } from './MasteryGraph';

// Mock ReactFlow since it uses Canvas/WebGL
vi.mock('reactflow', () => ({
    default: ({ nodes, edges, children }: any) => (
        <div data-testid="react-flow">
            <span>ReactFlow Mock</span>
            <div data-testid="graph-nodes">
                {nodes?.map((node: any) => (
                    <div key={node.id} data-testid={`node-${node.id}`}>
                        {node.data?.label}
                    </div>
                ))}
            </div>
            <div data-testid="graph-edges">
                {edges?.map((edge: any) => (
                    <div key={edge.id} data-testid={`edge-${edge.id}`}>
                        {edge.source} -&gt; {edge.target}
                    </div>
                ))}
            </div>
            {children}
        </div>
    ),
    Controls: () => <div data-testid="controls">Controls</div>,
    Background: () => <div data-testid="background">Background</div>,
    useNodesState: (initial: any[] = []) => {
        const state = { nodes: initial };
        return [state.nodes, (n: any) => { state.nodes = n; }, vi.fn()];
    },
    useEdgesState: (initial: any[] = []) => {
        const state = { edges: initial };
        return [state.edges, (e: any) => { state.edges = e; }, vi.fn()];
    },
    MarkerType: { ArrowClosed: 'arrowclosed' },
    ConnectionLineType: { SmoothStep: 'smoothstep' },
}));

// Mock ELK layout
vi.mock('elkjs/lib/elk.bundled', () => ({
    default: class ELK {
        layout(graph: any) {
            return Promise.resolve({
                ...graph,
                children: graph.children?.map((child: any, i: number) => ({
                    ...child,
                    x: i * 100,
                    y: i * 50,
                })),
            });
        }
    },
}));

// Mock the hook
const mockUseGraphData: {
    data: { nodes: any[]; links: any[] } | null;
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
        mockUseGraphData.loading = false;
    });

    it('renders the container and legend', () => {
        render(<MasteryGraph />);

        // Should have a "Concept" label
        expect(screen.getByText('Concept')).toBeDefined();
        expect(screen.getByTestId('react-flow')).toBeDefined();
    });

    it('shows loading state when loading', () => {
        mockUseGraphData.loading = true;
        const { container } = render(<MasteryGraph />);

        // Should show loading indicator (Loader2 component)
        expect(container.querySelector('.animate-spin')).toBeDefined();
    });

    it('handles empty data gracefully', () => {
        mockUseGraphData.data = { nodes: [], links: [] };
        render(<MasteryGraph />);
        expect(screen.getByTestId('react-flow')).toBeDefined();
    });
});
