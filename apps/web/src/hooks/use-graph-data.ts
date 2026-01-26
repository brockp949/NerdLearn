import { useQuery } from '@tanstack/react-query';
import { graphApi } from '@/lib/api';
import { GraphData } from '@/types/graph';

export function useGraphData(courseId: number = 1) {
    const { data, isLoading, error } = useQuery({
        queryKey: ['graph', courseId],
        queryFn: async (): Promise<GraphData> => {
            // Use the new endpoint which returns 'edges'
            const response = await graphApi.getCourseGraph(courseId);

            // Map API 'edges' to 'links' for frontend compatibility
            const links = (response.edges || []).map((edge: any) => ({
                source: edge.source,
                target: edge.target,
                type: edge.type,
                confidence: edge.confidence
            }));

            // Map API nodes to UI structure
            const nodes = (response.nodes || []).map((node: any) => ({
                ...node,
                name: node.label,
                domain: node.module || 'General',
                mastery: 0,
                cardsReviewed: 0,
                totalCards: 0
            }));

            return { nodes, links };
        },
        staleTime: 10 * 60 * 1000, // 10 minutes - graph data doesn't change often
        gcTime: 30 * 60 * 1000, // 30 minutes garbage collection
    });

    return {
        data: data ?? { nodes: [], links: [] },
        loading: isLoading,
        error: error instanceof Error ? error : null
    };
}
