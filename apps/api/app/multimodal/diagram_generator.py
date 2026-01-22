"""
Diagram Generation Service - Mermaid → React Flow Conversion

Research alignment:
- React Flow: Interactive diagram rendering
- Mermaid: Declarative diagram syntax
- Knowledge Visualization: Concept maps, flowcharts, hierarchies

This module transforms educational content into interactive diagrams,
enabling visual learning experiences that complement text and audio.

Diagram Types:
1. Flowchart: Process flows, decision trees, algorithms
2. Mindmap: Concept hierarchies, topic exploration
3. Sequence: Interactions, API flows, protocols
4. Entity-Relationship: Data models, system architecture
5. State: State machines, lifecycle diagrams
"""
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import logging
import hashlib
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DiagramType(str, Enum):
    """Supported diagram types"""
    FLOWCHART = "flowchart"
    MINDMAP = "mindmap"
    SEQUENCE = "sequence"
    ENTITY_RELATIONSHIP = "er"
    STATE = "state"
    CLASS = "class"
    CONCEPT_MAP = "concept_map"


class NodeType(str, Enum):
    """React Flow node types"""
    DEFAULT = "default"
    INPUT = "input"
    OUTPUT = "output"
    GROUP = "group"
    CONCEPT = "concept"
    PROCESS = "process"
    DECISION = "decision"
    ANNOTATION = "annotation"


class EdgeType(str, Enum):
    """React Flow edge types"""
    DEFAULT = "default"
    STRAIGHT = "straight"
    STEP = "step"
    SMOOTHSTEP = "smoothstep"
    BEZIER = "bezier"
    ANIMATED = "animated"


@dataclass
class ReactFlowNode:
    """Represents a React Flow node"""
    id: str
    type: NodeType
    position: Dict[str, float]
    data: Dict[str, Any]
    style: Optional[Dict[str, Any]] = None
    parent_node: Optional[str] = None
    extent: Optional[str] = None  # 'parent' for child nodes

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "type": self.type.value,
            "position": self.position,
            "data": self.data,
        }
        if self.style:
            result["style"] = self.style
        if self.parent_node:
            result["parentNode"] = self.parent_node
            result["extent"] = self.extent or "parent"
        return result


@dataclass
class ReactFlowEdge:
    """Represents a React Flow edge"""
    id: str
    source: str
    target: str
    type: EdgeType = EdgeType.SMOOTHSTEP
    label: Optional[str] = None
    animated: bool = False
    style: Optional[Dict[str, Any]] = None
    marker_end: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.type.value,
        }
        if self.label:
            result["label"] = self.label
        if self.animated:
            result["animated"] = True
        if self.style:
            result["style"] = self.style
        if self.marker_end:
            result["markerEnd"] = self.marker_end
        return result


@dataclass
class DiagramData:
    """Complete diagram representation"""
    id: str
    type: DiagramType
    title: str
    nodes: List[ReactFlowNode]
    edges: List[ReactFlowEdge]
    mermaid_source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_react_flow(self) -> Dict[str, Any]:
        """Convert to React Flow compatible format"""
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "mermaidSource": self.mermaid_source,
            "metadata": self.metadata
        }


class MermaidParser:
    """
    Parses Mermaid diagram syntax into React Flow components

    Supports:
    - flowchart (TB, LR, etc.)
    - mindmap
    - sequenceDiagram
    - erDiagram
    - stateDiagram-v2
    """

    # Node shape patterns for flowcharts
    NODE_PATTERNS = {
        r'\[([^\]]+)\]': NodeType.DEFAULT,      # [text] - rectangle
        r'\(([^\)]+)\)': NodeType.PROCESS,       # (text) - rounded
        r'\{([^\}]+)\}': NodeType.DECISION,      # {text} - diamond
        r'\[\[([^\]]+)\]\]': NodeType.GROUP,     # [[text]] - subroutine
        r'\(\(([^\)]+)\)\)': NodeType.INPUT,     # ((text)) - circle
        r'\>([^\]]+)\]': NodeType.ANNOTATION,    # >text] - flag
    }

    # Edge patterns
    EDGE_PATTERNS = [
        (r'-->', EdgeType.DEFAULT, None),           # simple arrow
        (r'---', EdgeType.STRAIGHT, None),          # line no arrow
        (r'-\.->',EdgeType.DEFAULT, 'dotted'),      # dotted arrow
        (r'==>', EdgeType.DEFAULT, 'thick'),        # thick arrow
        (r'--([^-]+)-->', EdgeType.DEFAULT, None),  # labeled arrow
        (r'-->|([^|]+)|', EdgeType.DEFAULT, None),  # labeled arrow alt
    ]

    def __init__(self):
        self.node_counter = 0
        self.layout_config = {
            "horizontal_spacing": 200,
            "vertical_spacing": 100,
            "start_x": 50,
            "start_y": 50
        }

    def _generate_node_id(self, label: str) -> str:
        """Generate unique node ID from label"""
        # Clean the label for ID
        clean = re.sub(r'[^a-zA-Z0-9]', '_', label)[:20]
        return f"node_{clean}_{self.node_counter}"

    def _extract_node_text(self, node_def: str) -> Tuple[str, NodeType, str]:
        """Extract text and type from node definition"""
        # Try each pattern
        for pattern, node_type in self.NODE_PATTERNS.items():
            match = re.search(pattern, node_def)
            if match:
                return match.group(1), node_type, node_def.split('[')[0].split('(')[0].split('{')[0].strip()

        # Default: just text
        return node_def.strip(), NodeType.DEFAULT, node_def.strip()

    def parse_flowchart(self, mermaid: str) -> Tuple[List[ReactFlowNode], List[ReactFlowEdge]]:
        """
        Parse a Mermaid flowchart into React Flow components

        Example:
        flowchart TD
            A[Start] --> B{Decision}
            B -->|Yes| C[Action]
            B -->|No| D[Other]
        """
        nodes: List[ReactFlowNode] = []
        edges: List[ReactFlowEdge] = []
        node_map: Dict[str, ReactFlowNode] = {}

        lines = mermaid.strip().split('\n')
        direction = "TD"  # Top-Down default

        # Parse header
        for line in lines:
            line = line.strip()
            if line.startswith('flowchart') or line.startswith('graph'):
                parts = line.split()
                if len(parts) > 1:
                    direction = parts[1]
                continue

            if not line or line.startswith('%%'):
                continue

            # Parse node definitions and edges
            # Pattern: A[text] --> B[text]
            edge_match = re.search(r'(\w+)(\[.*?\]|\(.*?\)|\{.*?\})?\s*(-->|---|-\.->|==>)\|?([^|]*)\|?\s*(\w+)(\[.*?\]|\(.*?\)|\{.*?\})?', line)

            if edge_match:
                source_id = edge_match.group(1)
                source_def = edge_match.group(2) or ""
                edge_type_str = edge_match.group(3)
                edge_label = edge_match.group(4).strip() if edge_match.group(4) else None
                target_id = edge_match.group(5)
                target_def = edge_match.group(6) or ""

                # Create source node if not exists
                if source_id not in node_map:
                    text, node_type, _ = self._extract_node_text(source_def) if source_def else (source_id, NodeType.DEFAULT, source_id)
                    self.node_counter += 1
                    node = ReactFlowNode(
                        id=source_id,
                        type=node_type,
                        position={"x": 0, "y": 0},  # Will be calculated later
                        data={"label": text or source_id}
                    )
                    nodes.append(node)
                    node_map[source_id] = node

                # Create target node if not exists
                if target_id not in node_map:
                    text, node_type, _ = self._extract_node_text(target_def) if target_def else (target_id, NodeType.DEFAULT, target_id)
                    self.node_counter += 1
                    node = ReactFlowNode(
                        id=target_id,
                        type=node_type,
                        position={"x": 0, "y": 0},
                        data={"label": text or target_id}
                    )
                    nodes.append(node)
                    node_map[target_id] = node

                # Create edge
                edge_type = EdgeType.SMOOTHSTEP
                if edge_type_str == '---':
                    edge_type = EdgeType.STRAIGHT
                elif edge_type_str == '-..->':
                    edge_type = EdgeType.STEP

                edge = ReactFlowEdge(
                    id=f"edge_{source_id}_{target_id}",
                    source=source_id,
                    target=target_id,
                    type=edge_type,
                    label=edge_label,
                    marker_end={"type": "arrowclosed"} if '-->' in edge_type_str or '==>' in edge_type_str else None
                )
                edges.append(edge)
            else:
                # Standalone node definition
                node_match = re.match(r'(\w+)(\[.*?\]|\(.*?\)|\{.*?\})', line)
                if node_match:
                    node_id = node_match.group(1)
                    node_def = node_match.group(2)

                    if node_id not in node_map:
                        text, node_type, _ = self._extract_node_text(node_def)
                        self.node_counter += 1
                        node = ReactFlowNode(
                            id=node_id,
                            type=node_type,
                            position={"x": 0, "y": 0},
                            data={"label": text}
                        )
                        nodes.append(node)
                        node_map[node_id] = node

        # Calculate positions based on direction
        self._calculate_positions(nodes, edges, direction)

        return nodes, edges

    def parse_mindmap(self, mermaid: str) -> Tuple[List[ReactFlowNode], List[ReactFlowEdge]]:
        """
        Parse a Mermaid mindmap into React Flow components

        Example:
        mindmap
          root((Central Idea))
            Branch1
              Leaf1
              Leaf2
            Branch2
        """
        nodes: List[ReactFlowNode] = []
        edges: List[ReactFlowEdge] = []

        lines = mermaid.strip().split('\n')
        indent_stack: List[Tuple[int, str]] = []  # (indent_level, node_id)

        for line in lines:
            if line.strip().startswith('mindmap'):
                continue
            if not line.strip() or line.strip().startswith('%%'):
                continue

            # Calculate indent level
            indent = len(line) - len(line.lstrip())
            text = line.strip()

            # Extract node text (handle special shapes)
            node_type = NodeType.CONCEPT
            if text.startswith('((') and text.endswith('))'):
                text = text[2:-2]
                node_type = NodeType.INPUT  # Circle for root
            elif text.startswith('(') and text.endswith(')'):
                text = text[1:-1]
                node_type = NodeType.PROCESS
            elif text.startswith('[') and text.endswith(']'):
                text = text[1:-1]

            self.node_counter += 1
            node_id = f"mm_{self.node_counter}"

            node = ReactFlowNode(
                id=node_id,
                type=node_type,
                position={"x": 0, "y": 0},
                data={"label": text}
            )
            nodes.append(node)

            # Find parent based on indent
            while indent_stack and indent_stack[-1][0] >= indent:
                indent_stack.pop()

            if indent_stack:
                parent_id = indent_stack[-1][1]
                edge = ReactFlowEdge(
                    id=f"edge_{parent_id}_{node_id}",
                    source=parent_id,
                    target=node_id,
                    type=EdgeType.BEZIER
                )
                edges.append(edge)

            indent_stack.append((indent, node_id))

        # Calculate radial positions for mindmap
        self._calculate_mindmap_positions(nodes, edges)

        return nodes, edges

    def _calculate_positions(
        self,
        nodes: List[ReactFlowNode],
        edges: List[ReactFlowEdge],
        direction: str = "TD"
    ):
        """Calculate node positions using simple layered layout"""
        if not nodes:
            return

        # Build adjacency for topological sort
        adj: Dict[str, List[str]] = {n.id: [] for n in nodes}
        in_degree: Dict[str, int] = {n.id: 0 for n in nodes}

        for edge in edges:
            if edge.source in adj:
                adj[edge.source].append(edge.target)
            if edge.target in in_degree:
                in_degree[edge.target] += 1

        # Topological sort into layers
        layers: List[List[str]] = []
        current_layer = [n_id for n_id, deg in in_degree.items() if deg == 0]

        while current_layer:
            layers.append(current_layer)
            next_layer = []
            for n_id in current_layer:
                for child in adj.get(n_id, []):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_layer.append(child)
            current_layer = next_layer

        # Handle any remaining nodes (cycles or disconnected)
        remaining = [n.id for n in nodes if not any(n.id in layer for layer in layers)]
        if remaining:
            layers.append(remaining)

        # Assign positions
        node_map = {n.id: n for n in nodes}
        h_spacing = self.layout_config["horizontal_spacing"]
        v_spacing = self.layout_config["vertical_spacing"]

        is_horizontal = direction in ["LR", "RL"]

        for layer_idx, layer in enumerate(layers):
            for node_idx, node_id in enumerate(layer):
                if node_id in node_map:
                    if is_horizontal:
                        node_map[node_id].position = {
                            "x": self.layout_config["start_x"] + layer_idx * h_spacing,
                            "y": self.layout_config["start_y"] + node_idx * v_spacing
                        }
                    else:
                        node_map[node_id].position = {
                            "x": self.layout_config["start_x"] + node_idx * h_spacing,
                            "y": self.layout_config["start_y"] + layer_idx * v_spacing
                        }

    def _calculate_mindmap_positions(
        self,
        nodes: List[ReactFlowNode],
        edges: List[ReactFlowEdge]
    ):
        """Calculate radial positions for mindmap layout"""
        import math

        if not nodes:
            return

        # Build tree structure
        children: Dict[str, List[str]] = {n.id: [] for n in nodes}
        has_parent: set = set()

        for edge in edges:
            children[edge.source].append(edge.target)
            has_parent.add(edge.target)

        # Find root (node with no parent)
        root_id = None
        for node in nodes:
            if node.id not in has_parent:
                root_id = node.id
                break

        if not root_id:
            root_id = nodes[0].id

        node_map = {n.id: n for n in nodes}

        # Position root at center
        center_x, center_y = 400, 300
        node_map[root_id].position = {"x": center_x, "y": center_y}

        # BFS to position children in rings
        radius = 150
        queue = [(root_id, 0)]  # (node_id, level)
        level_counts: Dict[int, int] = {}
        level_indices: Dict[int, int] = {}

        # First pass: count nodes per level
        visited = {root_id}
        temp_queue = [root_id]
        level_map = {root_id: 0}

        while temp_queue:
            current = temp_queue.pop(0)
            current_level = level_map[current]
            level_counts[current_level] = level_counts.get(current_level, 0) + 1

            for child in children.get(current, []):
                if child not in visited:
                    visited.add(child)
                    level_map[child] = current_level + 1
                    temp_queue.append(child)

        # Second pass: position nodes
        level_indices = {k: 0 for k in level_counts}
        visited = {root_id}
        queue = [root_id]

        while queue:
            current = queue.pop(0)
            current_level = level_map[current]

            for child in children.get(current, []):
                if child not in visited:
                    visited.add(child)
                    child_level = level_map[child]

                    # Calculate position on ring
                    count = level_counts[child_level]
                    idx = level_indices[child_level]
                    level_indices[child_level] += 1

                    angle = (2 * math.pi * idx / count) - math.pi / 2
                    r = radius * child_level

                    node_map[child].position = {
                        "x": center_x + r * math.cos(angle),
                        "y": center_y + r * math.sin(angle)
                    }

                    queue.append(child)

    def parse(self, mermaid: str, diagram_type: Optional[DiagramType] = None) -> Tuple[List[ReactFlowNode], List[ReactFlowEdge]]:
        """
        Parse Mermaid syntax into React Flow components

        Args:
            mermaid: Mermaid diagram source
            diagram_type: Optional type hint (auto-detected if not provided)

        Returns:
            (nodes, edges) tuple
        """
        self.node_counter = 0

        # Auto-detect type
        if not diagram_type:
            first_line = mermaid.strip().split('\n')[0].lower()
            if 'flowchart' in first_line or 'graph' in first_line:
                diagram_type = DiagramType.FLOWCHART
            elif 'mindmap' in first_line:
                diagram_type = DiagramType.MINDMAP
            elif 'sequencediagram' in first_line:
                diagram_type = DiagramType.SEQUENCE
            elif 'erdiagram' in first_line:
                diagram_type = DiagramType.ENTITY_RELATIONSHIP
            elif 'statediagram' in first_line:
                diagram_type = DiagramType.STATE
            else:
                diagram_type = DiagramType.FLOWCHART

        if diagram_type == DiagramType.FLOWCHART:
            return self.parse_flowchart(mermaid)
        elif diagram_type == DiagramType.MINDMAP:
            return self.parse_mindmap(mermaid)
        elif diagram_type == DiagramType.CONCEPT_MAP:
            return self.parse_flowchart(mermaid)  # Use flowchart parser
        else:
            # Default to flowchart parser
            return self.parse_flowchart(mermaid)


class DiagramGenerator:
    """
    Generates interactive diagrams from educational content

    Pipeline:
    1. Content Analysis: Extract key concepts and relationships
    2. Mermaid Generation: Create appropriate diagram syntax
    3. React Flow Conversion: Parse to interactive components
    4. Layout Optimization: Position nodes for readability
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.parser = MermaidParser()

    def _generate_diagram_id(self, content: str, diagram_type: DiagramType) -> str:
        """Generate unique diagram ID"""
        hash_input = f"{content[:100]}:{diagram_type.value}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    async def generate_mermaid(
        self,
        content: str,
        diagram_type: DiagramType,
        focus_concepts: Optional[List[str]] = None
    ) -> str:
        """
        Generate Mermaid diagram from content using LLM

        Args:
            content: Educational content to visualize
            diagram_type: Type of diagram to generate
            focus_concepts: Optional concepts to highlight

        Returns:
            Mermaid syntax string
        """
        type_instructions = {
            DiagramType.FLOWCHART: """Create a flowchart showing the process or algorithm.
Use flowchart TD (top-down) or LR (left-right).
Syntax: A[Step] --> B{Decision} --> C[Action]""",

            DiagramType.MINDMAP: """Create a mindmap with the main topic at center.
Syntax:
mindmap
  root((Main Topic))
    Branch1
      Leaf1
      Leaf2
    Branch2""",

            DiagramType.CONCEPT_MAP: """Create a concept map showing relationships between ideas.
Use flowchart with labeled edges.
Syntax: Concept1 -->|relationship| Concept2""",

            DiagramType.SEQUENCE: """Create a sequence diagram showing interactions.
Syntax:
sequenceDiagram
    Actor1->>Actor2: Message
    Actor2-->>Actor1: Response""",

            DiagramType.STATE: """Create a state diagram showing states and transitions.
Syntax:
stateDiagram-v2
    [*] --> State1
    State1 --> State2: event
    State2 --> [*]"""
        }

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at creating educational diagrams.
Generate valid Mermaid syntax that clearly visualizes the content.

Rules:
1. Keep diagrams focused and readable (5-15 nodes typically)
2. Use clear, concise labels
3. Show logical flow and relationships
4. Highlight key concepts
5. Use appropriate node shapes for meaning

{type_instructions}

Output ONLY the Mermaid code, no explanation."""),
            ("human", """Content to visualize:
{content}

{focus_section}

Generate a {diagram_type} diagram:""")
        ])

        focus_section = ""
        if focus_concepts:
            focus_section = f"Focus on these concepts: {', '.join(focus_concepts)}"

        try:
            messages = prompt.format_messages(
                type_instructions=type_instructions.get(diagram_type, ""),
                content=content[:2000],
                focus_section=focus_section,
                diagram_type=diagram_type.value
            )

            response = await self.llm.ainvoke(messages)
            mermaid_code = response.content

            # Clean up response
            if "```mermaid" in mermaid_code:
                start = mermaid_code.find("```mermaid") + 10
                end = mermaid_code.find("```", start)
                mermaid_code = mermaid_code[start:end]
            elif "```" in mermaid_code:
                start = mermaid_code.find("```") + 3
                end = mermaid_code.find("```", start)
                mermaid_code = mermaid_code[start:end]

            return mermaid_code.strip()

        except Exception as e:
            logger.error(f"Error generating Mermaid: {e}")
            # Return basic diagram on error
            return f"""flowchart TD
    A[Content Analysis] --> B[Diagram Generation]
    B --> C[Visualization]"""

    async def generate(
        self,
        content: str,
        diagram_type: DiagramType = DiagramType.CONCEPT_MAP,
        title: Optional[str] = None,
        focus_concepts: Optional[List[str]] = None
    ) -> DiagramData:
        """
        Generate a complete interactive diagram from content

        Args:
            content: Educational content to visualize
            diagram_type: Type of diagram
            title: Optional diagram title
            focus_concepts: Optional concepts to highlight

        Returns:
            DiagramData with React Flow components
        """
        logger.info(f"Generating {diagram_type.value} diagram")

        # Generate Mermaid syntax
        mermaid = await self.generate_mermaid(content, diagram_type, focus_concepts)

        # Parse to React Flow
        nodes, edges = self.parser.parse(mermaid, diagram_type)

        # Generate ID and title
        diagram_id = self._generate_diagram_id(content, diagram_type)
        if not title:
            title = f"{diagram_type.value.title()} Diagram"

        return DiagramData(
            id=diagram_id,
            type=diagram_type,
            title=title,
            nodes=nodes,
            edges=edges,
            mermaid_source=mermaid,
            metadata={
                "node_count": len(nodes),
                "edge_count": len(edges),
                "focus_concepts": focus_concepts or []
            }
        )

    async def generate_from_concepts(
        self,
        concepts: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        title: str = "Concept Map"
    ) -> DiagramData:
        """
        Generate diagram directly from concept/relationship data

        Args:
            concepts: List of {"id": str, "name": str, "type": str}
            relationships: List of {"source": str, "target": str, "label": str}
            title: Diagram title

        Returns:
            DiagramData
        """
        nodes = []
        edges = []

        # Create nodes
        for i, concept in enumerate(concepts):
            node_type = NodeType.CONCEPT
            if concept.get("type") == "root":
                node_type = NodeType.INPUT
            elif concept.get("type") == "leaf":
                node_type = NodeType.OUTPUT

            node = ReactFlowNode(
                id=concept.get("id", f"c_{i}"),
                type=node_type,
                position={"x": 0, "y": 0},
                data={
                    "label": concept.get("name", f"Concept {i}"),
                    "description": concept.get("description", "")
                },
                style=concept.get("style")
            )
            nodes.append(node)

        # Create edges
        for i, rel in enumerate(relationships):
            edge = ReactFlowEdge(
                id=f"rel_{i}",
                source=rel.get("source"),
                target=rel.get("target"),
                type=EdgeType.SMOOTHSTEP,
                label=rel.get("label"),
                animated=rel.get("animated", False),
                marker_end={"type": "arrowclosed"}
            )
            edges.append(edge)

        # Calculate positions
        self.parser._calculate_positions(nodes, edges, "TD")

        # Generate Mermaid representation
        mermaid_lines = ["flowchart TD"]
        for node in nodes:
            mermaid_lines.append(f"    {node.id}[{node.data['label']}]")
        for edge in edges:
            label = f"|{edge.label}|" if edge.label else ""
            mermaid_lines.append(f"    {edge.source} -->{label} {edge.target}")

        return DiagramData(
            id=hashlib.md5(title.encode()).hexdigest()[:12],
            type=DiagramType.CONCEPT_MAP,
            title=title,
            nodes=nodes,
            edges=edges,
            mermaid_source="\n".join(mermaid_lines),
            metadata={
                "node_count": len(nodes),
                "edge_count": len(edges),
                "generated_from": "concepts"
            }
        )

    async def update_diagram(
        self,
        diagram: DiagramData,
        updates: Dict[str, Any]
    ) -> DiagramData:
        """
        Update an existing diagram with new data

        Args:
            diagram: Existing diagram
            updates: {"add_nodes": [], "remove_nodes": [], "add_edges": [], ...}

        Returns:
            Updated DiagramData
        """
        nodes = list(diagram.nodes)
        edges = list(diagram.edges)

        # Remove nodes
        remove_ids = set(updates.get("remove_nodes", []))
        nodes = [n for n in nodes if n.id not in remove_ids]
        edges = [e for e in edges if e.source not in remove_ids and e.target not in remove_ids]

        # Add nodes
        for node_data in updates.get("add_nodes", []):
            node = ReactFlowNode(
                id=node_data.get("id", f"new_{len(nodes)}"),
                type=NodeType(node_data.get("type", "default")),
                position=node_data.get("position", {"x": 100, "y": 100}),
                data=node_data.get("data", {"label": "New Node"})
            )
            nodes.append(node)

        # Add edges
        for edge_data in updates.get("add_edges", []):
            edge = ReactFlowEdge(
                id=edge_data.get("id", f"new_edge_{len(edges)}"),
                source=edge_data.get("source"),
                target=edge_data.get("target"),
                label=edge_data.get("label")
            )
            edges.append(edge)

        # Update node positions if provided
        for node_update in updates.get("update_positions", []):
            for node in nodes:
                if node.id == node_update.get("id"):
                    node.position = node_update.get("position", node.position)

        return DiagramData(
            id=diagram.id,
            type=diagram.type,
            title=diagram.title,
            nodes=nodes,
            edges=edges,
            mermaid_source=diagram.mermaid_source,  # May be stale after updates
            metadata={**diagram.metadata, "updated": True}
        )


# Global instance (lazy initialization)
_diagram_generator: Optional[DiagramGenerator] = None


def get_diagram_generator_instance() -> DiagramGenerator:
    """Get or create the diagram generator instance (lazy)"""
    global _diagram_generator
    if _diagram_generator is None:
        _diagram_generator = DiagramGenerator()
    return _diagram_generator


async def get_diagram_generator() -> DiagramGenerator:
    """Dependency injection"""
    return get_diagram_generator_instance()
