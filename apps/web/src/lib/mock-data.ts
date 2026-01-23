import { GraphData } from "@/components/analytics/KnowledgeGraphView";

export const generateMockGraphData = (): GraphData => {
    const concepts = [
        { id: '1', name: 'Variables', domain: 'Python', mastery: 0.85, cardsReviewed: 12, totalCards: 15, bloomLevel: 'REMEMBER' },
        { id: '2', name: 'Functions', domain: 'Python', mastery: 0.72, cardsReviewed: 10, totalCards: 15, bloomLevel: 'UNDERSTAND' },
        { id: '3', name: 'Loops', domain: 'Python', mastery: 0.65, cardsReviewed: 8, totalCards: 12, bloomLevel: 'APPLY' },
        { id: '4', name: 'Lists', domain: 'Python', mastery: 0.78, cardsReviewed: 11, totalCards: 14, bloomLevel: 'UNDERSTAND' },
        { id: '5', name: 'Dictionaries', domain: 'Python', mastery: 0.55, cardsReviewed: 6, totalCards: 12, bloomLevel: 'UNDERSTAND' },
        { id: '6', name: 'Control Flow', domain: 'Python', mastery: 0.80, cardsReviewed: 9, totalCards: 11, bloomLevel: 'APPLY' },
        { id: '7', name: 'Recursion', domain: 'Python', mastery: 0.35, cardsReviewed: 4, totalCards: 10, bloomLevel: 'ANALYZE' },
        { id: '8', name: 'Error Handling', domain: 'Python', mastery: 0.42, cardsReviewed: 5, totalCards: 10, bloomLevel: 'APPLY' },
        { id: '9', name: 'File I/O', domain: 'Python', mastery: 0.0, cardsReviewed: 0, totalCards: 8, bloomLevel: 'APPLY' },
        { id: '10', name: 'Classes', domain: 'Python', mastery: 0.0, cardsReviewed: 0, totalCards: 12, bloomLevel: 'CREATE' }
    ];

    const prerequisites = [
        { source: '2', target: '1', weight: 0.9, type: 'prerequisite' as const }, // Functions → Variables
        { source: '3', target: '1', weight: 0.8, type: 'prerequisite' as const }, // Loops → Variables
        { source: '3', target: '6', weight: 0.7, type: 'prerequisite' as const }, // Loops → Control Flow
        { source: '4', target: '1', weight: 0.8, type: 'prerequisite' as const }, // Lists → Variables
        { source: '5', target: '4', weight: 0.9, type: 'prerequisite' as const }, // Dictionaries → Lists
        { source: '7', target: '2', weight: 0.9, type: 'prerequisite' as const }, // Recursion → Functions
        { source: '8', target: '2', weight: 0.7, type: 'prerequisite' as const }, // Error Handling → Functions
        { source: '9', target: '8', weight: 0.8, type: 'prerequisite' as const }, // File I/O → Error Handling
        { source: '10', target: '2', weight: 0.9, type: 'prerequisite' as const }, // Classes → Functions
        { source: '10', target: '5', weight: 0.8, type: 'prerequisite' as const }  // Classes → Dictionaries
    ];

    return {
        nodes: concepts,
        edges: prerequisites
    };
};
