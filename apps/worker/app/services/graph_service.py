"""
Knowledge Graph Service
Constructs and manages concept relationships in Neo4j
"""
from neo4j import GraphDatabase
from typing import List, Dict, Any, Set
import re
from collections import Counter
from ..config import config


class GraphService:
    """Manages knowledge graph in Neo4j"""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )

    def close(self):
        """Close the database connection"""
        self.driver.close()

    def extract_concepts(self, text: str, min_freq: int = 2) -> List[str]:
        """
        Extract potential concepts from text using simple NLP heuristics

        Args:
            text: Text to analyze
            min_freq: Minimum frequency for a concept to be included

        Returns:
            List of concept names
        """
        # Simple concept extraction: look for capitalized phrases (potential terms)
        # This is a simplified version - in production, use NER or domain-specific extraction

        # Find capitalized phrases (2-4 words)
        pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b"
        matches = re.findall(pattern, text)

        # Count frequencies
        concept_counts = Counter(matches)

        # Filter by frequency
        concepts = [
            concept
            for concept, count in concept_counts.items()
            if count >= min_freq and len(concept) > 3
        ]

        # Also look for common technical terms (simple dictionary approach)
        technical_terms = self._extract_technical_terms(text.lower())

        # Combine and deduplicate
        all_concepts = list(set(concepts + technical_terms))

        return all_concepts[:50]  # Limit to top 50 concepts

    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract common technical/educational terms"""
        # Simplified term extraction - in production, use domain-specific dictionaries
        common_terms = [
            "algorithm",
            "data structure",
            "neural network",
            "machine learning",
            "deep learning",
            "python",
            "javascript",
            "database",
            "api",
            "function",
            "class",
            "object",
            "variable",
            "loop",
            "array",
            "string",
            "integer",
            "boolean",
            "framework",
            "library",
            "module",
            "package",
            "interface",
            "inheritance",
            "polymorphism",
            "encapsulation",
            "abstraction",
        ]

        found_terms = []
        for term in common_terms:
            if term in text:
                found_terms.append(term.title())

        return found_terms

    def create_course_graph(
        self, course_id: int, course_title: str, modules: List[Dict[str, Any]]
    ):
        """
        Create a knowledge graph for a course

        Args:
            course_id: Course ID
            course_title: Course title
            modules: List of modules with their extracted concepts
        """
        with self.driver.session() as session:
            # Create course node
            session.run(
                """
                MERGE (c:Course {id: $course_id})
                SET c.title = $title
                """,
                course_id=course_id,
                title=course_title,
            )

            # Process each module
            for module in modules:
                module_id = module["id"]
                module_title = module["title"]
                concepts = module.get("concepts", [])

                # Create module node
                session.run(
                    """
                    MERGE (m:Module {id: $module_id})
                    SET m.title = $title, m.course_id = $course_id
                    WITH m
                    MATCH (c:Course {id: $course_id})
                    MERGE (c)-[:HAS_MODULE]->(m)
                    """,
                    module_id=module_id,
                    title=module_title,
                    course_id=course_id,
                )

                # Create concept nodes and relationships
                for concept in concepts:
                    session.run(
                        """
                        MERGE (con:Concept {name: $concept, course_id: $course_id})
                        WITH con
                        MATCH (m:Module {id: $module_id})
                        MERGE (m)-[:TEACHES]->(con)
                        """,
                        concept=concept,
                        course_id=course_id,
                        module_id=module_id,
                    )

            # Detect prerequisite relationships (simplified heuristic)
            self._detect_prerequisites(session, course_id)

    def _detect_prerequisites(self, session, course_id: int):
        """
        Detect potential prerequisite relationships between concepts

        This is a simplified heuristic. In production, use:
        - Sequential ordering in modules
        - Co-occurrence analysis
        - Explicit prerequisite declarations
        - ML-based prerequisite detection
        """
        # Simple heuristic: concepts in earlier modules are prerequisites for later ones
        session.run(
            """
            MATCH (m1:Module)-[:TEACHES]->(c1:Concept)
            MATCH (m2:Module)-[:TEACHES]->(c2:Concept)
            WHERE m1.course_id = $course_id
              AND m2.course_id = $course_id
              AND m1.id < m2.id
              AND c1.name <> c2.name
            MERGE (c1)-[r:PREREQUISITE_FOR {confidence: 0.3}]->(c2)
            """,
            course_id=course_id,
        )

    def get_course_graph(self, course_id: int) -> Dict[str, Any]:
        """
        Get the knowledge graph for a course

        Args:
            course_id: Course ID

        Returns:
            Graph data with nodes and edges
        """
        with self.driver.session() as session:
            # Get all concepts and their relationships
            result = session.run(
                """
                MATCH (c:Course {id: $course_id})-[:HAS_MODULE]->(m:Module)-[:TEACHES]->(con:Concept)
                OPTIONAL MATCH (con)-[r:PREREQUISITE_FOR]->(con2:Concept)
                WHERE con2.course_id = $course_id
                RETURN con.name as concept, m.title as module,
                       collect({target: con2.name, confidence: r.confidence}) as prerequisites
                """,
                course_id=course_id,
            )

            nodes = []
            edges = []
            seen_concepts = set()

            for record in result:
                concept = record["concept"]

                if concept not in seen_concepts:
                    nodes.append(
                        {
                            "id": concept,
                            "label": concept,
                            "module": record["module"],
                            "type": "concept",
                        }
                    )
                    seen_concepts.add(concept)

                # Add prerequisite edges
                for prereq in record["prerequisites"]:
                    if prereq["target"]:
                        edges.append(
                            {
                                "source": concept,
                                "target": prereq["target"],
                                "type": "prerequisite",
                                "confidence": prereq.get("confidence", 0.5),
                            }
                        )

            return {"nodes": nodes, "edges": edges}

    def get_concept_prerequisites(self, course_id: int, concept: str) -> List[str]:
        """Get all prerequisites for a concept"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c1:Concept)-[:PREREQUISITE_FOR]->(c2:Concept {name: $concept, course_id: $course_id})
                RETURN c1.name as prerequisite
                ORDER BY prerequisite
                """,
                concept=concept,
                course_id=course_id,
            )

            return [record["prerequisite"] for record in result]

    def add_prerequisite(
        self, course_id: int, prerequisite: str, concept: str, confidence: float = 1.0
    ):
        """Manually add a prerequisite relationship"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (c1:Concept {name: $prerequisite, course_id: $course_id})
                MATCH (c2:Concept {name: $concept, course_id: $course_id})
                MERGE (c1)-[r:PREREQUISITE_FOR]->(c2)
                SET r.confidence = $confidence
                """,
                prerequisite=prerequisite,
                concept=concept,
                course_id=course_id,
                confidence=confidence,
            )
