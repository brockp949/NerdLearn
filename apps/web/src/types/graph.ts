export interface GraphNode {
    id: string;
    label: string;
    name?: string;
    module?: string;
    module_id?: number;
    module_order?: number;
    difficulty?: number;
    importance?: number;
    type?: string;
    domain?: string;
    mastery?: number;
    cardsReviewed?: number;
    totalCards?: number;
    bloomLevel?: string;
    x?: number;
    y?: number;
    z?: number;
}

export interface GraphLink {
    source: string | GraphNode;
    target: string | GraphNode;
    type?: string;
    confidence?: number;
}

export interface GraphData {
    nodes: GraphNode[];
    links: GraphLink[];
}
