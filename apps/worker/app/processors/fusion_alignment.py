"""
Multi-modal Fusion Alignment
Synchronizes video transcripts with PDF sections using semantic similarity
"""
from typing import List, Dict, Any, Optional
import numpy as np
from app.services.vector_store import VectorStoreService

class FusionAligner:
    """
    Aligns video segments with PDF pages/sections
    """

    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store

    async def align_video_to_pdf(
        self, 
        video_chunks: List[Dict[str, Any]], 
        pdf_chunks: List[Dict[str, Any]],
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Create alignment links between video chunks and PDF chunks
        
        Args:
            video_chunks: List of chunks from a video
            pdf_chunks: List of chunks from a PDF
            threshold: Similarity threshold for a valid link
            
        Returns:
            List of alignment objects {video_chunk_id, pdf_chunk_id, score}
        """
        alignments = []
        
        for v_chunk in video_chunks:
            # In a real implementation, we would use embeddings here
            # For this optimization, we assume we have access to the vector store
            # or pre-calculated embeddings.
            
            v_text = v_chunk["text"]
            
            # Simple semantic search proxy: find best matching PDF chunk
            # Performance note: In production, batch these similarity calculations
            best_match = None
            highest_score = 0.0
            
            for p_chunk in pdf_chunks:
                score = self._calculate_similarity(v_text, p_chunk["text"])
                if score > highest_score:
                    highest_score = score
                    best_match = p_chunk
            
            if highest_score >= threshold:
                alignments.append({
                    "video_chunk_id": v_chunk.get("chunk_id"),
                    "pdf_chunk_id": best_match.get("chunk_id"),
                    "score": highest_score,
                    "metadata": {
                        "video_timestamp": v_chunk.get("timestamp_start"),
                        "pdf_page": best_match.get("metadata", {}).get("page_number")
                    }
                })
                
        return alignments

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity (placeholder for embedding-based cosine similarity)
        """
        # In production, use sentence-transformers or OpenAI embeddings
        # For now, implemented as a simple word-overlap token similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union)
