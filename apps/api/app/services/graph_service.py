"""
Async Knowledge Graph Service for API
Implements Apache AGE integration with PostgreSQL
"""
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.core.config import settings
import re
from collections import Counter
import logging
import json
from fastapi import Depends
from app.core.database import get_db

import spacy

logger = logging.getLogger(__name__)

# Global NLP instance to avoid reloading per request
_nlp = None

import os

def get_nlp_model():
    global _nlp
    
    # Check if heavy NLP is enabled (default to False for API safety)
    if os.getenv("ENABLE_HEAVY_NLP", "false").lower() != "true":
        return None

    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Spacy model not found. Downloading...")
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.error(f"Failed to load Spacy: {e}")
            return None
    return _nlp


class AsyncGraphService:
    """
    Async Knowledge Graph Service using Apache AGE (PostgreSQL Extension)
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.graph_name = "nerdlearn_graph"
        self._initialized = False

    async def _init_age(self):
        """Initialize AGE extension and graph if needed"""
        if not self._initialized:
            try:
                # Load extension
                await self.db.execute(text("LOAD 'age'"))
                await self.db.execute(text("SET search_path = ag_catalog, \"$user\", public"))
                
                # Check if graph exists
                res = await self.db.execute(text("SELECT count(*) FROM ag_graph WHERE name = :name"), {"name": self.graph_name})
                count = res.scalar()
                
                if count == 0:
                    await self.db.execute(text(f"SELECT create_graph('{self.graph_name}')"))
                
                self._initialized = True
            except Exception as e:
                logger.warning(f"AGE initialization warning: {e}")

    async def run_cypher(self, query: str, columns_def: str, params: dict = None) -> List[Any]:
        """
        Run a cypher query using AGE.
        """
        await self._init_age()
        
        # Parameter substitution
        formatted_query = query
        if params:
            for k, v in params.items():
                if isinstance(v, str):
                    safe_v = v.replace("'", "\\'")
                    formatted_query = formatted_query.replace(f"${k}", f"'{safe_v}'")
                elif isinstance(v, (int, float)):
                    formatted_query = formatted_query.replace(f"${k}", str(v))
                elif isinstance(v, list):
                    safe_v = json.dumps(v).replace("'", "\\'")
                    formatted_query = formatted_query.replace(f"${k}", f"{safe_v}")
        
        # Prepare SQL with dollar-quoted string for AGE
        # We use explicit $$ quoting as required by AGE's cypher() function
        # This avoids issues with single-quote escaping in the SQL layer
        sql = f"SELECT * FROM cypher('{self.graph_name}', $$ {formatted_query} $$) as ({columns_def})"
        
        try:
            # Escape colons for SQLAlchemy text() parsing
            # This is necessary because SQLAlchemy interprets :word as a bind parameter
            stmt = text(sql.replace(":", "\\:"))
            
            result = await self.db.execute(stmt)
            return result.all()
        except Exception as e:
            logger.error(f"Cypher Query Failed: {sql} | Error: {e}")
            raise

    # ============== Graph Queries ==============

    async def get_course_graph(self, course_id: int) -> Dict[str, Any]:
        # Return a single JSON object column to avoid driver issues with multiple agtype columns
        columns = "res agtype"
        query = """
        MATCH (c:Course {id: $course_id})-[:HAS_MODULE]->(m:Module)-[:TEACHES]->(con:Concept)
        OPTIONAL MATCH (con)-[r:PREREQUISITE_FOR]->(con2:Concept)
        WHERE con2.course_id = $course_id
        RETURN {
            c_name: con.name,
            c_diff: con.difficulty,
            c_imp: con.importance,
            m_title: m.title,
            m_id: m.id,
            m_order: m.module_order,
            p_target: con2.name,
            p_conf: r.confidence,
            p_type: r.type
        }
        """
        
        records = await self.run_cypher(query, columns, {"course_id": course_id})
        
        nodes_map = {}
        edges = []

        for record in records:
            # record.res is the map
            data = self._parse_agtype(record.res)
            if not data:
                continue
                
            # Handle potential string serialization of agtype
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except:
                    pass
            
            # AGE sometimes returns keys without quotes in string representation if not valid JSON? 
            # Ideally asyncpg + age returns dict or valid JSON string.
            
            concept = data.get('c_name')
            if not concept:
                continue

            # Aggregate nodes
            if concept not in nodes_map:
                nodes_map[concept] = {
                    "id": concept,
                    "label": concept,
                    "module": data.get('m_title'),
                    "module_id": data.get('m_id'),
                    "module_order": data.get('m_order') or 0,
                    "difficulty": data.get('c_diff') or 5.0,
                    "importance": data.get('c_imp') or 0.5,
                    "type": "concept"
                }

            # Collect edge
            target = data.get('p_target')
            if target:
                edges.append({
                    "source": concept,
                    "target": target,
                    "type": data.get('p_type') or "prerequisite",
                    "confidence": data.get('p_conf') or 0.5
                })

        return {
            "nodes": list(nodes_map.values()),
            "edges": edges,
            "meta": {
                "course_id": course_id,
                "total_concepts": len(nodes_map),
                "total_relationships": len(edges)
            }
        }

    async def get_concept_details(self, course_id: int, concept_name: str) -> Optional[Dict[str, Any]]:
        columns = "name agtype, difficulty agtype, importance agtype, description agtype, module agtype, module_id agtype, prerequisites agtype, dependents agtype"
        query = """
        MATCH (con:Concept {name: $concept_name, course_id: $course_id})
        OPTIONAL MATCH (m:Module)-[:TEACHES]->(con)
        OPTIONAL MATCH (prereq:Concept)-[r1:PREREQUISITE_FOR]->(con)
        OPTIONAL MATCH (con)-[r2:PREREQUISITE_FOR]->(dependent:Concept)
        RETURN con.name, con.difficulty, con.importance, con.description, m.title, m.id, 
               collect({"name": prereq.name, "confidence": r1.confidence}), 
               collect({"name": dependent.name, "confidence": r2.confidence})
        """
        
        records = await self.run_cypher(query, columns, {"course_id": course_id, "concept_name": concept_name})
        if not records:
            return None
        
        record = records[0]
        name = self._parse_agtype(record.name)
        if not name:
            return None
            
        return {
            "name": name,
            "difficulty": self._parse_agtype(record.difficulty) or 5.0,
            "importance": self._parse_agtype(record.importance) or 0.5,
            "description": self._parse_agtype(record.description),
            "module": self._parse_agtype(record.module),
            "module_id": self._parse_agtype(record.module_id),
            "prerequisites": [p for p in (self._parse_agtype(record.prerequisites) or []) if p.get("name")],
            "dependents": [d for d in (self._parse_agtype(record.dependents) or []) if d.get("name")]
        }

    async def get_learning_path(self, course_id: int, target_concepts: List[str], user_mastered: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        user_mastered = user_mastered or []
        columns = "name agtype, difficulty agtype, module_order agtype, depth agtype, weight agtype"
        
        query = """
        UNWIND $targets as target
        MATCH path = (prereq:Concept)-[:PREREQUISITE_FOR*0..10]->(goal:Concept {name: target, course_id: $course_id})
        WHERE prereq.course_id = $course_id
        WITH nodes(path) as concepts, length(path) as depth
        UNWIND concepts as concept
        WITH DISTINCT concept, max(depth) as max_depth
        OPTIONAL MATCH (m:Module)-[:TEACHES]->(concept)
        RETURN concept.name,
               concept.difficulty,
               m.module_order,
               max_depth,
               (max_depth * 2.0 + coalesce(concept.difficulty, 5.0))
        ORDER BY (max_depth * 2.0 + coalesce(concept.difficulty, 5.0)) ASC, m.module_order ASC
        """
        
        records = await self.run_cypher(query, columns, {"course_id": course_id, "targets": target_concepts})
        
        learning_path = []
        for r in records:
             name = self._parse_agtype(r.name)
             if name and name not in user_mastered:
                 learning_path.append({
                     "name": name,
                     "difficulty": self._parse_agtype(r.difficulty) or 5.0,
                     "depth": self._parse_agtype(r.depth),
                     "module_order": self._parse_agtype(r.module_order) or 0,
                     "weight": self._parse_agtype(r.weight)
                 })
        return learning_path

    # ============== Mutations ==============

    async def create_course_node(self, course_id: int, title: str) -> bool:
        cols = "v agtype"
        query = """
        MERGE (c:Course {id: $course_id})
        SET c.title = $title
        RETURN c
        """
        records = await self.run_cypher(query, cols, {"course_id": course_id, "title": title})
        return len(records) > 0

    async def create_module_node(self, course_id: int, module_id: int, title: str, order: int = 0) -> bool:
        cols = "v agtype"
        query = """
        MERGE (m:Module {id: $module_id})
        SET m.title = $title, m.course_id = $course_id, m.module_order = $order
        WITH m
        MATCH (c:Course {id: $course_id})
        MERGE (c)-[:HAS_MODULE]->(m)
        RETURN m
        """
        records = await self.run_cypher(query, cols, {"course_id": course_id, "module_id": module_id, "title": title, "order": order})
        return len(records) > 0

    async def create_concept_node(self, course_id: int, module_id: int, name: str, difficulty: float = 5.0, importance: float = 0.5, description: str = None) -> bool:
        cols = "name agtype"
        query = """
        MERGE (con:Concept {name: $name, course_id: $course_id})
        SET con.difficulty = $difficulty,
            con.importance = $importance,
            con.description = $description
        WITH con
        MATCH (m:Module {id: $module_id})
        MERGE (m)-[:TEACHES]->(con)
        RETURN con.name
        """
        records = await self.run_cypher(query, cols, {
            "course_id": course_id, "module_id": module_id, "name": name, 
            "difficulty": difficulty, "importance": importance, "description": description or ""
        })
        return len(records) > 0

    async def add_prerequisite(self, course_id: int, prerequisite_name: str, concept_name: str, confidence: float = 1.0, prereq_type: str = "explicit") -> bool:
        cols = "a agtype, b agtype"
        query = """
        MATCH (prereq:Concept {name: $prerequisite_name, course_id: $course_id})
        MATCH (con:Concept {name: $concept_name, course_id: $course_id})
        MERGE (prereq)-[r:PREREQUISITE_FOR]->(con)
        SET r.confidence = $confidence, r.type = $prereq_type
        RETURN prereq.name, con.name
        """
        records = await self.run_cypher(query, cols, {
            "course_id": course_id, "prerequisite_name": prerequisite_name, 
            "concept_name": concept_name, "confidence": confidence, "prereq_type": prereq_type
        })
        return len(records) > 0

    async def remove_prerequisite(self, course_id: int, prerequisite_name: str, concept_name: str) -> bool:
        cols = "a agtype, b agtype"
        query = """
        MATCH (prereq:Concept {name: $prerequisite_name, course_id: $course_id})-[r:PREREQUISITE_FOR]->(con:Concept {name: $concept_name, course_id: $course_id})
        DELETE r
        RETURN prereq.name, con.name
        """
        records = await self.run_cypher(query, cols, {
            "course_id": course_id, "prerequisite_name": prerequisite_name, 
            "concept_name": concept_name
        })
        return len(records) > 0
        
    async def detect_prerequisites_sequential(self, course_id: int) -> int:
        cols = "count agtype"
        # AGE doesn't support < or > for string properties automatically, assuming explicit casting
        query = """
        MATCH (m1:Module)-[:TEACHES]->(c1:Concept)
        MATCH (m2:Module)-[:TEACHES]->(c2:Concept)
        WHERE m1.course_id = $course_id 
          AND m2.course_id = $course_id
          AND toInteger(m1.module_order) < toInteger(m2.module_order)
          AND c1.name <> c2.name
        MERGE (c1)-[r:PREREQUISITE_FOR]->(c2)
        SET r.confidence = 0.3
        RETURN count(r)
        """
        try:
            records = await self.run_cypher(query, cols, {"course_id": course_id})
            if records:
                return self._parse_agtype(records[0].count)
            return 0
        except Exception as e:
            logger.error(f"Sequential detection failed: {e}")
            return 0

    # ============== Stats & Extras ==============

    async def get_graph_stats(self, course_id: int) -> Dict[str, Any]:
        cols = "module_count agtype, concept_count agtype, relation_count agtype"
        query = """
        MATCH (c:Course {id: $course_id})
        OPTIONAL MATCH (c)-[:HAS_MODULE]->(m:Module)
        OPTIONAL MATCH (m)-[:TEACHES]->(con:Concept)
        OPTIONAL MATCH (con)-[r:PREREQUISITE_FOR]->(con2:Concept)
        RETURN count(DISTINCT m), count(DISTINCT con), count(DISTINCT r)
        """
        records = await self.run_cypher(query, cols, {"course_id": course_id})
        if records:
            return {
                "modules": self._parse_agtype(records[0].module_count),
                "concepts": self._parse_agtype(records[0].concept_count),
                "prerequisites": self._parse_agtype(records[0].relation_count),
                "avg_difficulty": 5.0
            }
        return {"modules":0, "concepts":0, "prerequisites":0, "avg_difficulty":0}
        
    async def find_concepts_without_prerequisites(self, course_id: int) -> List[str]:
        cols = "name agtype"
        query = """
        MATCH (con:Concept {course_id: $course_id})
        WHERE NOT (()-[:PREREQUISITE_FOR]->(con))
        RETURN con.name
        """
        records = await self.run_cypher(query, cols, {"course_id": course_id})
        return [self._parse_agtype(r.name) for r in records if r.name]
        
    async def find_terminal_concepts(self, course_id: int) -> List[str]:
        cols = "name agtype"
        query = """
        MATCH (con:Concept {course_id: $course_id})
        WHERE NOT ((con)-[:PREREQUISITE_FOR]->())
        RETURN con.name
        """
        records = await self.run_cypher(query, cols, {"course_id": course_id})
        return [self._parse_agtype(r.name) for r in records if r.name]
        
    async def get_prerequisites(self, course_id: int, concept_name: str) -> List[str]:
        cols = "name agtype"
        query = """
        MATCH (p:Concept {course_id: $course_id})-[r:PREREQUISITE_FOR]->(c:Concept {name: $concept_name, course_id: $course_id})
        RETURN p.name
        """
        records = await self.run_cypher(query, cols, {"course_id": course_id, "concept_name": concept_name})
        return [self._parse_agtype(r.name) for r in records if r.name]

    async def get_dependents(self, course_id: int, concept_name: str) -> List[str]:
        cols = "name agtype"
        query = """
        MATCH (c:Concept {name: $concept_name, course_id: $course_id})-[r:PREREQUISITE_FOR]->(d:Concept {course_id: $course_id})
        RETURN d.name
        """
        records = await self.run_cypher(query, cols, {"course_id": course_id, "concept_name": concept_name})
        return [self._parse_agtype(r.name) for r in records if r.name]

    # ============== Community Detection ==============

    async def update_community_structure(self, course_id: int, community_map: Dict[str, int]) -> int:
        cols = "count agtype"
        count = 0
        for name, cid in community_map.items():
            query = """
            MATCH (c:Concept {name: $name, course_id: $course_id})
            SET c.community_id = $community_id
            RETURN count(c)
            """
            records = await self.run_cypher(query, cols, {"name": name, "course_id": course_id, "community_id": cid})
            if records:
                try:
                    count += int(self._parse_agtype(records[0].count))
                except (ValueError, TypeError):
                    pass
        return count

    async def get_community_members(self, course_id: int, community_id: int) -> List[Dict[str, Any]]:
        cols = "name agtype, description agtype, importance agtype"
        query = """
        MATCH (c:Concept {course_id: $course_id, community_id: $community_id})
        RETURN c.name, c.description, c.importance
        """
        records = await self.run_cypher(query, cols, {"course_id": course_id, "community_id": community_id})
        return [{
            "name": self._parse_agtype(r.name),
            "description": self._parse_agtype(r.description) or "",
            "importance": self._parse_agtype(r.importance) or 0.0
        } for r in records if r.name]
        
    async def get_all_communities(self, course_id: int) -> List[int]:
        cols = "id agtype"
        query = """
        MATCH (c:Concept {course_id: $course_id})
        WHERE c.community_id IS NOT NULL
        RETURN DISTINCT c.community_id
        """
        records = await self.run_cypher(query, cols, {"course_id": course_id})
        return sorted([int(self._parse_agtype(r.id)) for r in records if r.id is not None])

    def extract_concepts(self, text: str, min_freq: int = 1) -> List[str]:
        """
        Extract educational concepts from text using NER and noun chunking.
        """
        nlp = get_nlp_model()
        
        if not nlp:
            # Fallback regex logic
            pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b"
            matches = re.findall(pattern, text)
            concept_counts = Counter(matches)
            concepts = [c for c, count in concept_counts.items() if count >= min_freq and len(c) > 3]
            return concepts[:50]
            
        # Limit text length to avoid excessive processing time
        doc = nlp(text[:100000])
        
        candidates = []
        
        # 1. Named Entities
        valid_labels = {'ORG', 'PRODUCT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'EVENT', 'FAC'}
        for ent in doc.ents:
            if ent.label_ in valid_labels:
                clean_ent = ent.text.strip()
                if len(clean_ent) > 2 and not clean_ent.isnumeric():
                    candidates.append(clean_ent)

        # 2. Capitalized Noun Chunks
        for chunk in doc.noun_chunks:
            clean_chunk = chunk.text.strip()

            # Remove leading determiners
            words = clean_chunk.split()
            if words[0].lower() in {'the', 'a', 'an', 'this', 'that', 'these', 'those'}:
                if len(words) > 1:
                    clean_chunk = " ".join(words[1:])
                else:
                    continue

            # Check for capitalization (heuristic for concept)
            if len(clean_chunk) > 3 and clean_chunk[0].isupper():
                candidates.append(clean_chunk)

        # Count and filter
        concept_counts = Counter(candidates)
        valid_concepts = [c for c, count in concept_counts.items() if count >= min_freq]
        
        # Sort by frequency then length
        valid_concepts.sort(key=lambda x: (concept_counts[x], len(x)), reverse=True)
        
        return valid_concepts[:50]

    def _parse_agtype(self, value):
        if value is None:
            return None
        return value

# Dependency Injection
async def get_graph_service(db: AsyncSession = Depends(get_db)):
    return AsyncGraphService(db)
