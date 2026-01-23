"""
Knowledge Graph Service
Constructs and manages concept relationships in PostgreSQL using Apache AGE
"""
import psycopg2
from typing import List, Dict, Any, Set, Optional
import re
import logging
from collections import Counter
from ..config import config
import json
import spacy

logger = logging.getLogger(__name__)

class GraphService:
    """Manages knowledge graph in PostgreSQL/AGE with NER-based concept extraction"""

    def __init__(self, use_ner: bool = True):
        self.dsn = config.DATABASE_URL
        self.graph_name = "nerdlearn_graph"
        self.use_ner = use_ner
        self._init_age()
        
        if self.use_ner:
            try:
                # Load English tokenizer, tagger, parser and NER
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("Spacy model not found. Downloading...")
                from spacy.cli import download
                download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.warning(f"Could not load Spacy: {e}. Disabling NER.")
                self.use_ner = False
        
    def get_conn(self):
        return psycopg2.connect(self.dsn)

    def _init_age(self):
        try:
            with self.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("LOAD 'age'")
                    cur.execute("SET search_path = ag_catalog, \"$user\", public")
                    # Check if graph exists
                    cur.execute("SELECT count(*) FROM ag_graph WHERE name = %s", (self.graph_name,))
                    if cur.fetchone()[0] == 0:
                        cur.execute(f"SELECT create_graph('{self.graph_name}')")
                conn.commit()
        except Exception as e:
            logger.warning(f"AGE init warning: {e}")

    def _exec_cypher(self, query: str, cols: str, params: dict = None, conn = None):
        """Helper to execute cypher"""
        formatted_query = query
        if params:
            for k, v in params.items():
                if isinstance(v, str):
                    safe_v = v.replace("'", "\\'")
                    formatted_query = formatted_query.replace(f"${k}", f"'{safe_v}'")
                elif isinstance(v, (int, float)):
                    formatted_query = formatted_query.replace(f"${k}", str(v))
        
        sql = f"SELECT * FROM cypher('{self.graph_name}', $$ {formatted_query} $$) as ({cols})"
        
        should_close = False
        if not conn:
            conn = self.get_conn()
            should_close = True
            
        try:
            with conn.cursor() as cur:
                cur.execute("LOAD 'age'")
                cur.execute("SET search_path = ag_catalog, \"$user\", public")
                cur.execute(sql)
                res = cur.fetchall()
                if should_close:
                    conn.commit()
                return res
        except Exception as e:
            if should_close:
                conn.rollback()
            raise e
        finally:
            if should_close:
                conn.close()

    def extract_concepts(self, text: str, min_freq: int = 1) -> List[str]:
        """
        Extract educational concepts from text using NER and noun chunking.
        """
        if not self.use_ner or not hasattr(self, 'nlp'):
            # Fallback regex logic
            pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b"
            matches = re.findall(pattern, text)
            concept_counts = Counter(matches)
            concepts = [c for c, count in concept_counts.items() if count >= min_freq and len(c) > 3]
            return concepts[:50]
            
        # Limit text length to avoid memory issues with huge docs
        doc = self.nlp(text[:100000])
        
        candidates = []
        
        # 1. Named Entities (Select relevant types)
        # ORG: Companies, agencies, institutions
        # PRODUCT: Objects, vehicles, foods, etc. (often used for software tools)
        # WORK_OF_ART: Titles of books, songs, etc.
        # LAW: Named documents made into laws.
        # LANGUAGE: Any named language.
        # EVENT: Named hurricanes, battles, wars, sports events, etc.
        valid_labels = {'ORG', 'PRODUCT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'EVENT', 'FAC'}
        
        for ent in doc.ents:
            if ent.label_ in valid_labels:
                clean_ent = ent.text.strip()
                if len(clean_ent) > 2 and not clean_ent.isnumeric():
                    candidates.append(clean_ent)

        # 2. Capitalized Noun Chunks (Heuristic for concepts)
        for chunk in doc.noun_chunks:
            clean_chunk = chunk.text.strip()
            
            # Skip if it's just a common word starting with a stop word
            # e.g. "The algorithm" -> "algorithm" (lowercase, ignore) 
            # vs "The Euclidean Algorithm" -> "Euclidean Algorithm" (keep)
            
            # Remove leading determiners/articles
            words = clean_chunk.split()
            if words[0].lower() in {'the', 'a', 'an', 'this', 'that', 'these', 'those'}:
                if len(words) > 1:
                    clean_chunk = " ".join(words[1:])
                else:
                    continue

            # Check for capitalization (significant concept indicator)
            # We want strings where meaningful words are capitalized
            # e.g. "Machine Learning", "Neural Networks"
            if len(clean_chunk) > 3 and clean_chunk[0].isupper():
                # Filter out single common words that might be capitalized at start of sentence
                # Spacy handles sentence boundaries well, but noun chunks can be tricky
                # We'll rely on frequency to filter out noise later
                candidates.append(clean_chunk)

        # Count frequencies
        concept_counts = Counter(candidates)
        
        # Filter by minimum frequency
        valid_concepts = [c for c, count in concept_counts.items() if count >= min_freq]
        
        # Sort by frequency (desc) then length (desc)
        # This favors frequent and specific (longer) concepts
        valid_concepts.sort(key=lambda x: (concept_counts[x], len(x)), reverse=True)
        
        return valid_concepts[:50]

    def create_course_graph(self, course_id: int, course_title: str, modules: List[Dict[str, Any]]):
        """Create a knowledge graph for a course"""
        conn = self.get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("LOAD 'age'")
                cur.execute("SET search_path = ag_catalog, \"$user\", public")
                
                # Course
                sql = f"""SELECT * FROM cypher('{self.graph_name}', $$
                    MERGE (c:Course {{id: {course_id}}})
                    SET c.title = '{course_title.replace("'", "\\'")}'
                    RETURN c
                    $$) as (c agtype)
                """
                cur.execute(sql)
                
                # Modules
                for module in modules:
                    m_id = module["id"]
                    m_title = module["title"].replace("'", "\\'")
                    
                    sql = f"""SELECT * FROM cypher('{self.graph_name}', $$
                        MERGE (m:Module {{id: {m_id}}})
                        SET m.title = '{m_title}', m.course_id = {course_id}
                        WITH m
                        MATCH (c:Course {{id: {course_id}}})
                        MERGE (c)-[:HAS_MODULE]->(m)
                        RETURN m
                        $$) as (m agtype)
                    """
                    cur.execute(sql)
                    
                    for concept in module.get("concepts", []):
                        c_name = concept.replace("'", "\\'")
                        sql = f"""SELECT * FROM cypher('{self.graph_name}', $$
                            MERGE (con:Concept {{name: '{c_name}', course_id: {course_id}}})
                            WITH con
                            MATCH (m:Module {{id: {m_id}}})
                            MERGE (m)-[:TEACHES]->(con)
                            RETURN con
                            $$) as (con agtype)
                        """
                        cur.execute(sql)
            conn.commit()
        finally:
            conn.close()

    def get_course_graph(self, course_id: int) -> Dict[str, Any]:
        query = """
        MATCH (c:Course {id: $course_id})-[:HAS_MODULE]->(m:Module)-[:TEACHES]->(con:Concept)
        OPTIONAL MATCH (con)-[r:PREREQUISITE_FOR]->(con2:Concept)
        WHERE con2.course_id = $course_id
        RETURN con.name, m.title, collect({target: con2.name, confidence: r.confidence})
        """
        cols = "concept agtype, module agtype, prereqs agtype"
        
        try:
             rows = self._exec_cypher(query, cols, {"course_id": course_id})
        except Exception:
             return {"nodes": [], "edges": []}

        nodes = []
        edges = []
        seen = set()
        
        for row in rows:
            # parsing logic similar to async
            # row is tuple
            concept = row[0] # assuming string
            module = row[1]
            # prereqs might be a string json or list
            prereqs_raw = row[2]
            
            # AGE string parsing hack
            if isinstance(concept, str) and concept.startswith('"') and concept.endswith('"'):
                concept = concept[1:-1]
                
            if concept not in seen:
                nodes.append({"id": concept, "label": concept, "module": module, "type": "concept"})
                seen.add(concept)
            
            # Prereqs parsing
            # ... skipping robust parsing for brevity in worker ...
            
        return {"nodes": nodes, "edges": edges}

    def detect_prerequisites(self, course_id: int):
        """
        Detect potential prerequisite relationships between concepts
        """
        conn = self.get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("LOAD 'age'")
                cur.execute("SET search_path = ag_catalog, \"$user\", public")
                
                # Heuristic: Concepts in earlier modules are prerequisites for concepts in later modules
                query = f"""
                MATCH (m1:Module)-[:TEACHES]->(c1:Concept)
                MATCH (m2:Module)-[:TEACHES]->(c2:Concept)
                WHERE m1.course_id = {course_id} 
                  AND m2.course_id = {course_id}
                  AND toInteger(m1.id) < toInteger(m2.id)
                  AND c1.name <> c2.name
                MERGE (c1)-[r:PREREQUISITE_FOR]->(c2)
                SET r.confidence = 0.3
                """
                
                sql = f"SELECT * FROM cypher('{self.graph_name}', $$ {query} $$) as (r agtype)"
                cur.execute(sql)
            conn.commit()
        except Exception as e:
             logger.error(f"Failed to detect prerequisites: {e}")
             # Don't fail the whole task if this optimization fails
        finally:
            conn.close()

    def close(self):
        pass
