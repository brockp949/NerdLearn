"""
Embedding similarity utilities for goal alignment computation.

Implements the 'Antigravity' gravitational bias using semantic embeddings.
"""

from typing import List, Optional, Union
import numpy as np
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Returns value between -1 and 1, where:
    - 1 = identical vectors
    - 0 = orthogonal (no similarity)
    - -1 = opposite vectors
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
    
    Returns:
        Cosine similarity score
    """
    # Normalize vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    vec1_normalized = vec1 / norm1
    vec2_normalized = vec2 / norm2
    
    similarity = np.dot(vec1_normalized, vec2_normalized)
    
    # Clip to valid range due to floating point errors
    return float(np.clip(similarity, -1.0, 1.0))


def gravitational_bias(
    query_embedding: np.ndarray,
    candidates: List[np.ndarray],
    goal_vector: np.ndarray,
    bias_strength: float = 0.5
) -> List[float]:
    """
    Apply gravitational biasing to candidate scores.
    
    This implements the core 'Antigravity' concept from the PDF:
    Standard search ranks by similarity to query.
    Antigravity search ranks by similarity to query AND goal vector.
    
    The goal vector acts as a 'gravitational well' that pulls results
    toward the intended outcome, preventing 'ADHD' behavior.
    
    Args:
        query_embedding: The search query embedding
        candidates: List of candidate embeddings
        goal_vector: The goal vector to bias toward
        bias_strength: How much to weight the goal (0=no bias, 1=full bias)
    
    Returns:
        List of biased similarity scores
    """
    scores = []
    
    for candidate in candidates:
        # Base similarity to query
        query_sim = cosine_similarity(query_embedding, candidate)
        
        # Similarity to goal vector (gravitational pull)
        goal_sim = cosine_similarity(candidate, goal_vector)
        
        # Combine with bias weighting
        biased_score = (1 - bias_strength) * query_sim + bias_strength * goal_sim
        
        scores.append(biased_score)
    
    return scores


def semantic_distance(text1: str, text2: str, embedding_fn=None) -> float:
    """
    Compute semantic distance between two text strings.
    
    This is a placeholder that will be integrated with NerdLearn's
    existing embedding service.
    
    Args:
        text1: First text
        text2: Second text
        embedding_fn: Optional embedding function
    
    Returns:
        Distance score (0=identical, higher=more different)
    """
    if embedding_fn is None:
        # Placeholder: would integrate with NerdLearn's embedding service
        # For now, use simple string similarity
        return 1.0 - simple_text_similarity(text1, text2)
    
    # Generate embeddings and compute distance
    emb1 = embedding_fn(text1)
    emb2 = embedding_fn(text2)
    
    similarity = cosine_similarity(emb1, emb2)
    
    # Convert similarity to distance
    return 1.0 - similarity


def simple_text_similarity(text1: str, text2: str) -> float:
    """
    Simple text similarity using character overlap.
    
    This is a fallback when embeddings are not available.
    Should be replaced with proper embedding-based similarity.
    """
    if not text1 or not text2:
        return 0.0
    
    text1 = text1.lower()
    text2 = text2.lower()
    
    # Jaccard similarity of character sets
    set1 = set(text1.split())
    set2 = set(text2.split())
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def batch_similarity(
    query: np.ndarray,
    candidates: List[np.ndarray],
    top_k: Optional[int] = None
) -> List[tuple[int, float]]:
    """
    Compute similarity scores for a batch of candidates.
    
    Args:
        query: Query embedding
        candidates: List of candidate embeddings
        top_k: Optional, return only top K results
    
    Returns:
        List of (index, score) tuples, sorted by score descending
    """
    scores = [(i, cosine_similarity(query, candidate))
              for i, candidate in enumerate(candidates)]
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    if top_k:
        scores = scores[:top_k]
    
    return scores


def compute_alignment_score(
    output_text: str,
    goal_description: str,
    goal_embedding: Optional[np.ndarray] = None,
    embedding_fn=None
) -> float:
    """
    Compute how well output aligns with goal.
    
    This is the main function used by VerifierAgent for goal alignment.
    
    Args:
        output_text: The output to score
        goal_description: Description of the goal
        goal_embedding: Pre-computed goal embedding (optional)
        embedding_fn: Function to generate embeddings
    
    Returns:
        Alignment score from 0.0 to 1.0
    """
    if embedding_fn is None:
        # Fallback to simple text similarity
        similarity = simple_text_similarity(output_text, goal_description)
        # Normalize to 0-1 range
        return max(0.0, min(1.0, similarity))
    
    # Generate embedding for output
    output_embedding = embedding_fn(output_text)
    
    # Use provided goal embedding or generate it
    if goal_embedding is None:
        goal_embedding = embedding_fn(goal_description)
    
    # Compute similarity
    similarity = cosine_similarity(output_embedding, goal_embedding)
    
    # Convert from [-1, 1] to [0, 1] range
    score = (similarity + 1.0) / 2.0
    
    return score
