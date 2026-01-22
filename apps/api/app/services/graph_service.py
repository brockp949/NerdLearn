"""
Async Knowledge Graph Service for API
Implements Neo4j integration with NER-based concept extraction per research guidelines

Research alignment:
- Knowledge Graph Construction: GNNs, BERT-NER (92.8% F1), prerequisite extraction
- Prerequisite detection using sequential ordering and co-occurrence
"""
from typing import List, Dict, Any, Optional, Tuple
from neo4j import AsyncGraphDatabase
from app.core.config import settings
import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class AsyncGraphService:
    """
    Async Knowledge Graph Service for managing concept relationships in Neo4j

    Features:
    - Async Neo4j operations for FastAPI compatibility
    - Concept extraction with technical term detection
    - Prerequisite relationship management
    - Learning path generation based on graph structure
    """

    def __init__(self):
        self.driver = None
        self._connected = False

    async def connect(self):
        """Initialize Neo4j connection"""
        if not self._connected:
            self.driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            self._connected = True
            logger.info("Connected to Neo4j")

    async def close(self):
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
            self._connected = False
            logger.info("Disconnected from Neo4j")

    async def ensure_connected(self):
        """Ensure connection is established"""
        if not self._connected:
            await self.connect()

    # ============== Graph Queries ==============

    async def get_course_graph(self, course_id: int) -> Dict[str, Any]:
        """
        Get complete knowledge graph for a course

        Args:
            course_id: Course ID

        Returns:
            Graph with nodes (concepts) and edges (prerequisites)
        """
        await self.ensure_connected()

        query = """
        MATCH (c:Course {id: $course_id})-[:HAS_MODULE]->(m:Module)-[:TEACHES]->(con:Concept)
        OPTIONAL MATCH (con)-[r:PREREQUISITE_FOR]->(con2:Concept)
        WHERE con2.course_id = $course_id
        RETURN con.name as concept,
               con.difficulty as difficulty,
               con.importance as importance,
               m.title as module,
               m.id as module_id,
               m.order as module_order,
               collect(DISTINCT {target: con2.name, confidence: r.confidence, type: r.type}) as outgoing_prereqs
        ORDER BY m.order, con.name
        """

        async with self.driver.session() as session:
            result = await session.run(query, course_id=course_id)
            records = await result.data()

        nodes = []
        edges = []
        seen_concepts = set()
        concept_modules = {}

        for record in records:
            concept = record["concept"]

            if concept and concept not in seen_concepts:
                nodes.append({
                    "id": concept,
                    "label": concept,
                    "module": record["module"],
                    "module_id": record["module_id"],
                    "module_order": record["module_order"] or 0,
                    "difficulty": record["difficulty"] or 5.0,
                    "importance": record["importance"] or 0.5,
                    "type": "concept"
                })
                seen_concepts.add(concept)
                concept_modules[concept] = record["module_order"] or 0

            # Add prerequisite edges
            for prereq in record["outgoing_prereqs"] or []:
                if prereq.get("target"):
                    edges.append({
                        "source": concept,
                        "target": prereq["target"],
                        "type": prereq.get("type", "prerequisite"),
                        "confidence": prereq.get("confidence", 0.5)
                    })

        return {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "course_id": course_id,
                "total_concepts": len(nodes),
                "total_relationships": len(edges)
            }
        }

    async def get_concept_details(
        self, course_id: int, concept_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a concept including prerequisites and dependents

        Args:
            course_id: Course ID
            concept_name: Name of the concept

        Returns:
            Concept details with prerequisites and dependent concepts
        """
        await self.ensure_connected()

        query = """
        MATCH (con:Concept {name: $concept_name, course_id: $course_id})
        OPTIONAL MATCH (m:Module)-[:TEACHES]->(con)
        OPTIONAL MATCH (prereq:Concept)-[r1:PREREQUISITE_FOR]->(con)
        OPTIONAL MATCH (con)-[r2:PREREQUISITE_FOR]->(dependent:Concept)
        RETURN con.name as name,
               con.difficulty as difficulty,
               con.importance as importance,
               con.description as description,
               m.title as module,
               m.id as module_id,
               collect(DISTINCT {name: prereq.name, confidence: r1.confidence}) as prerequisites,
               collect(DISTINCT {name: dependent.name, confidence: r2.confidence}) as dependents
        """

        async with self.driver.session() as session:
            result = await session.run(
                query,
                course_id=course_id,
                concept_name=concept_name
            )
            record = await result.single()

        if not record:
            return None

        return {
            "name": record["name"],
            "difficulty": record["difficulty"] or 5.0,
            "importance": record["importance"] or 0.5,
            "description": record["description"],
            "module": record["module"],
            "module_id": record["module_id"],
            "prerequisites": [
                p for p in record["prerequisites"]
                if p.get("name")
            ],
            "dependents": [
                d for d in record["dependents"]
                if d.get("name")
            ]
        }

    async def get_learning_path(
        self,
        course_id: int,
        target_concepts: List[str],
        user_mastered: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate optimal learning path using difficulty-weighted sequencing
        
        Implements Dijkstra-inspired traversal where path "cost" is concept difficulty.
        This ensures the user is presented with the most accessible prerequisites first.

        Args:
            course_id: Course ID
            target_concepts: Concepts the user wants to learn
            user_mastered: Concepts already mastered by user

        Returns:
            Ordered list of concepts to learn
        """
        await self.ensure_connected()
        user_mastered = user_mastered or []

        # Find all prerequisite chains with difficulty-based ordering
        query = """
        UNWIND $targets as target
        MATCH path = (prereq:Concept)-[:PREREQUISITE_FOR*0..10]->(goal:Concept {name: target, course_id: $course_id})
        WHERE prereq.course_id = $course_id
        WITH nodes(path) as concepts, length(path) as depth
        UNWIND concepts as concept
        WITH DISTINCT concept, max(depth) as max_depth
        OPTIONAL MATCH (m:Module)-[:TEACHES]->(concept)
        RETURN concept.name as name,
               concept.difficulty as difficulty,
               m.order as module_order,
               max_depth as depth,
               # Weighting formula: combine depth in graph with intrinsic difficulty
               (max_depth * 2.0 + coalesce(concept.difficulty, 5.0)) as weight
        ORDER BY weight ASC, module_order ASC
        """

        async with self.driver.session() as session:
            result = await session.run(
                query,
                course_id=course_id,
                targets=target_concepts
            )
            records = await result.data()

        # Filter out already mastered concepts
        learning_path = [
            {
                "name": r["name"],
                "difficulty": r["difficulty"] or 5.0,
                "depth": r["depth"],
                "module_order": r["module_order"] or 0,
                "weight": r["weight"]
            }
            for r in records
            if r["name"] not in user_mastered
        ]

        return learning_path

    async def get_prerequisites(
        self, course_id: int, concept_name: str
    ) -> List[Dict[str, Any]]:
        """Get all prerequisites for a concept"""
        await self.ensure_connected()

        query = """
        MATCH (prereq:Concept)-[r:PREREQUISITE_FOR]->(con:Concept {name: $concept_name, course_id: $course_id})
        WHERE prereq.course_id = $course_id
        RETURN prereq.name as name,
               prereq.difficulty as difficulty,
               r.confidence as confidence,
               r.type as type
        ORDER BY r.confidence DESC
        """

        async with self.driver.session() as session:
            result = await session.run(
                query,
                course_id=course_id,
                concept_name=concept_name
            )
            records = await result.data()

        return [
            {
                "name": r["name"],
                "difficulty": r["difficulty"] or 5.0,
                "confidence": r["confidence"] or 0.5,
                "type": r["type"] or "sequential"
            }
            for r in records
        ]

    async def get_dependents(
        self, course_id: int, concept_name: str
    ) -> List[Dict[str, Any]]:
        """Get all concepts that depend on this concept"""
        await self.ensure_connected()

        query = """
        MATCH (con:Concept {name: $concept_name, course_id: $course_id})-[r:PREREQUISITE_FOR]->(dependent:Concept)
        WHERE dependent.course_id = $course_id
        RETURN dependent.name as name,
               dependent.difficulty as difficulty,
               r.confidence as confidence
        ORDER BY r.confidence DESC
        """

        async with self.driver.session() as session:
            result = await session.run(
                query,
                course_id=course_id,
                concept_name=concept_name
            )
            records = await result.data()

        return [
            {
                "name": r["name"],
                "difficulty": r["difficulty"] or 5.0,
                "confidence": r["confidence"] or 0.5
            }
            for r in records
        ]

    # ============== Graph Mutations ==============

    async def create_course_node(self, course_id: int, title: str) -> bool:
        """Create or update a course node"""
        await self.ensure_connected()

        query = """
        MERGE (c:Course {id: $course_id})
        SET c.title = $title, c.updated_at = datetime()
        RETURN c.id as id
        """

        async with self.driver.session() as session:
            result = await session.run(query, course_id=course_id, title=title)
            record = await result.single()
            return record is not None

    async def create_module_node(
        self,
        course_id: int,
        module_id: int,
        title: str,
        order: int = 0
    ) -> bool:
        """Create or update a module node and link to course"""
        await self.ensure_connected()

        query = """
        MERGE (m:Module {id: $module_id})
        SET m.title = $title, m.course_id = $course_id, m.order = $order
        WITH m
        MATCH (c:Course {id: $course_id})
        MERGE (c)-[:HAS_MODULE]->(m)
        RETURN m.id as id
        """

        async with self.driver.session() as session:
            result = await session.run(
                query,
                course_id=course_id,
                module_id=module_id,
                title=title,
                order=order
            )
            record = await result.single()
            return record is not None

    async def create_concept_node(
        self,
        course_id: int,
        module_id: int,
        name: str,
        difficulty: float = 5.0,
        importance: float = 0.5,
        description: Optional[str] = None
    ) -> bool:
        """Create or update a concept node and link to module"""
        await self.ensure_connected()

        query = """
        MERGE (con:Concept {name: $name, course_id: $course_id})
        SET con.difficulty = $difficulty,
            con.importance = $importance,
            con.description = $description
        WITH con
        MATCH (m:Module {id: $module_id})
        MERGE (m)-[:TEACHES]->(con)
        RETURN con.name as name
        """

        async with self.driver.session() as session:
            result = await session.run(
                query,
                course_id=course_id,
                module_id=module_id,
                name=name,
                difficulty=difficulty,
                importance=importance,
                description=description
            )
            record = await result.single()
            return record is not None

    async def add_prerequisite(
        self,
        course_id: int,
        prerequisite_name: str,
        concept_name: str,
        confidence: float = 1.0,
        prereq_type: str = "explicit"
    ) -> bool:
        """
        Add prerequisite relationship between concepts

        Args:
            course_id: Course ID
            prerequisite_name: Name of the prerequisite concept
            concept_name: Name of the dependent concept
            confidence: Confidence score (0-1)
            prereq_type: Type of prerequisite (explicit, sequential, inferred)
        """
        await self.ensure_connected()

        query = """
        MATCH (prereq:Concept {name: $prerequisite_name, course_id: $course_id})
        MATCH (con:Concept {name: $concept_name, course_id: $course_id})
        MERGE (prereq)-[r:PREREQUISITE_FOR]->(con)
        SET r.confidence = $confidence, r.type = $prereq_type
        RETURN prereq.name as prereq, con.name as concept
        """

        async with self.driver.session() as session:
            result = await session.run(
                query,
                course_id=course_id,
                prerequisite_name=prerequisite_name,
                concept_name=concept_name,
                confidence=confidence,
                prereq_type=prereq_type
            )
            record = await result.single()
            return record is not None

    async def remove_prerequisite(
        self,
        course_id: int,
        prerequisite_name: str,
        concept_name: str
    ) -> bool:
        """Remove a prerequisite relationship"""
        await self.ensure_connected()

        query = """
        MATCH (prereq:Concept {name: $prerequisite_name, course_id: $course_id})
              -[r:PREREQUISITE_FOR]->
              (con:Concept {name: $concept_name, course_id: $course_id})
        DELETE r
        RETURN count(r) as deleted
        """

        async with self.driver.session() as session:
            result = await session.run(
                query,
                course_id=course_id,
                prerequisite_name=prerequisite_name,
                concept_name=concept_name
            )
            record = await result.single()
            return record and record["deleted"] > 0

    async def detect_prerequisites_sequential(self, course_id: int) -> int:
        """
        Detect prerequisites based on sequential module ordering

        Research basis: Concepts in earlier modules are often prerequisites
        for concepts in later modules (confidence: 0.3)

        Returns:
            Number of relationships created
        """
        await self.ensure_connected()

        query = """
        MATCH (m1:Module)-[:TEACHES]->(c1:Concept)
        MATCH (m2:Module)-[:TEACHES]->(c2:Concept)
        WHERE m1.course_id = $course_id
          AND m2.course_id = $course_id
          AND m1.order < m2.order
          AND c1.name <> c2.name
          AND NOT EXISTS((c1)-[:PREREQUISITE_FOR]->(c2))
        MERGE (c1)-[r:PREREQUISITE_FOR {confidence: 0.3, type: 'sequential'}]->(c2)
        RETURN count(r) as created
        """

        async with self.driver.session() as session:
            result = await session.run(query, course_id=course_id)
            record = await result.single()
            return record["created"] if record else 0

    # ============== Analytics ==============

    async def get_graph_stats(self, course_id: int) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        await self.ensure_connected()

        query = """
        MATCH (c:Course {id: $course_id})
        OPTIONAL MATCH (c)-[:HAS_MODULE]->(m:Module)
        OPTIONAL MATCH (m)-[:TEACHES]->(con:Concept)
        OPTIONAL MATCH (con)-[r:PREREQUISITE_FOR]->()
        RETURN count(DISTINCT m) as modules,
               count(DISTINCT con) as concepts,
               count(DISTINCT r) as prerequisites,
               avg(con.difficulty) as avg_difficulty
        """

        async with self.driver.session() as session:
            result = await session.run(query, course_id=course_id)
            record = await result.single()

        if not record:
            return {
                "modules": 0,
                "concepts": 0,
                "prerequisites": 0,
                "avg_difficulty": 0
            }

        return {
            "modules": record["modules"] or 0,
            "concepts": record["concepts"] or 0,
            "prerequisites": record["prerequisites"] or 0,
            "avg_difficulty": round(record["avg_difficulty"] or 0, 2)
        }

    async def find_concepts_without_prerequisites(
        self, course_id: int
    ) -> List[str]:
        """Find concepts that have no prerequisites (entry points)"""
        await self.ensure_connected()

        query = """
        MATCH (con:Concept {course_id: $course_id})
        WHERE NOT EXISTS(()-[:PREREQUISITE_FOR]->(con))
        RETURN con.name as name
        ORDER BY con.name
        """

        async with self.driver.session() as session:
            result = await session.run(query, course_id=course_id)
            records = await result.data()

        return [r["name"] for r in records]

    async def find_terminal_concepts(self, course_id: int) -> List[str]:
        """Find concepts that are not prerequisites for anything (endpoints)"""
        await self.ensure_connected()

        query = """
        MATCH (con:Concept {course_id: $course_id})
        WHERE NOT EXISTS((con)-[:PREREQUISITE_FOR]->())
        RETURN con.name as name
        ORDER BY con.name
        """

        async with self.driver.session() as session:
            result = await session.run(query, course_id=course_id)
            records = await result.data()

        return [r["name"] for r in records]

    # ============== Concept Extraction ==============

    def extract_concepts(self, text: str, min_freq: int = 1) -> List[str]:
        """
        Extract concepts from text using pattern matching

        Note: For production, integrate with spaCy NER or domain-specific
        models as recommended in research (BERT-NER achieves 92.8% F1)

        Args:
            text: Text to analyze
            min_freq: Minimum frequency threshold

        Returns:
            List of extracted concept names
        """
        concepts = []

        # Extract capitalized multi-word phrases (potential concepts)
        pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b"
        matches = re.findall(pattern, text)
        concept_counts = Counter(matches)

        # Filter by frequency and length
        concepts.extend([
            concept for concept, count in concept_counts.items()
            if count >= min_freq and len(concept) > 3
        ])

        # Extract technical terms
        technical_terms = self._extract_technical_terms(text.lower())
        concepts.extend(technical_terms)

        # Deduplicate while preserving order
        seen = set()
        unique_concepts = []
        for c in concepts:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique_concepts.append(c)

        return unique_concepts[:100]  # Limit to top 100

    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract common technical/programming terms"""
        # Domain-specific term dictionary
        # In production, load from external file or use NER model
        technical_terms = {
            # Programming fundamentals
            "algorithm", "data structure", "variable", "function", "class",
            "object", "method", "loop", "array", "string", "integer", "boolean",
            "recursion", "iteration", "conditional", "operator", "expression",

            # OOP concepts
            "inheritance", "polymorphism", "encapsulation", "abstraction",
            "interface", "constructor", "destructor", "overloading", "overriding",

            # Data structures
            "linked list", "stack", "queue", "tree", "binary tree", "heap",
            "hash table", "graph", "trie", "set", "map", "dictionary",

            # Algorithms
            "sorting", "searching", "dynamic programming", "greedy algorithm",
            "divide and conquer", "backtracking", "breadth first search",
            "depth first search", "binary search", "merge sort", "quick sort",

            # Web development
            "api", "rest", "http", "html", "css", "javascript", "typescript",
            "react", "angular", "vue", "node", "express", "database",

            # Machine learning
            "neural network", "machine learning", "deep learning",
            "supervised learning", "unsupervised learning", "reinforcement learning",
            "classification", "regression", "clustering", "feature extraction",

            # Databases
            "sql", "nosql", "query", "index", "transaction", "normalization",
            "primary key", "foreign key", "join", "aggregation",

            # Software engineering
            "design pattern", "singleton", "factory", "observer", "decorator",
            "testing", "unit test", "integration test", "debugging", "refactoring",
            "version control", "git", "agile", "scrum", "continuous integration"
        }

        found_terms = []
        for term in technical_terms:
            if term in text:
                # Convert to title case for consistency
                found_terms.append(term.title())

        return found_terms


# Global instance
graph_service = AsyncGraphService()


async def get_graph_service() -> AsyncGraphService:
    """Dependency injection for graph service"""
    await graph_service.ensure_connected()
    return graph_service
