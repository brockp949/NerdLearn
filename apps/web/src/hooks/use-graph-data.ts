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
                const response = await graphApi.getGraph();
                // Ensure we have a valid structure
                setData({
                    nodes: response.nodes || [],
                    links: response.links || []
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
