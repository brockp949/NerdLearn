import { useState, useEffect } from 'react';
import { graphApi } from '@/lib/api';
import { GraphData } from '@/types/graph';

export function useGraphData() {
    const [data, setData] = useState<GraphData>({ nodes: [], links: [] });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<Error | null>(null);

    useEffect(() => {
        const fetchGraph = async () => {
            try {
                // Use the new endpoint which returns 'edges'
                const response = await graphApi.getCourseGraph(1);

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

                setData({
                    nodes: nodes,
                    links: links
                });
            } catch (err) {
                console.error("Failed to fetch graph", err);
                setError(err instanceof Error ? err : new Error('Unknown error'));
            } finally {
                setLoading(false);
            }
        };

        fetchGraph();
    }, []);

    return { data, loading, error };
}
