"""
Bootstrap Stability Selection for Causal Discovery

Per "Causal Discovery for Educational Graphs" PDF Section 6:
- Generate B bootstrap subsamples of size N/2 (without replacement)
- Run causal discovery (NOTEARS) on each subsample
- Calculate selection probability for each edge: count / B
- Classify edges by confidence threshold:
  - >0.85 = 'verified' (stable edges)
  - 0.5-0.85 = 'hypothetical' (candidate edges)
  - <0.5 = discarded (noise)

Research basis:
- Stability Selection (Meinshausen & Buhlmann, 2010)
- Bootstrap aggregating for structure learning
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

from app.adaptive.causal_discovery.algorithms.notears import NotearsAlgorithm

logger = logging.getLogger(__name__)


class BootstrapStabilitySelector:
    """
    Implements Bootstrap Stability Selection for robust edge confidence estimation
    in causal discovery.

    This class provides a principled way to quantify uncertainty in discovered
    causal edges by running the discovery algorithm on multiple resampled datasets.
    """

    def __init__(
        self,
        n_bootstrap: int = 100,
        subsample_ratio: float = 0.5,
        high_confidence_threshold: float = 0.85,
        low_confidence_threshold: float = 0.5,
        notears_lambda: float = 0.1,
        edge_threshold: float = 0.3
    ):
        """
        Initialize bootstrap selector.

        Args:
            n_bootstrap: Number of bootstrap samples (B in PDF, default 100)
            subsample_ratio: Fraction of data per subsample (N/2 recommended, default 0.5)
            high_confidence_threshold: Threshold for 'verified' status (default 0.85)
            low_confidence_threshold: Threshold for 'hypothetical' vs discard (default 0.5)
            notears_lambda: L1 sparsity penalty for NOTEARS
            edge_threshold: Minimum edge weight threshold for NOTEARS
        """
        self.n_bootstrap = n_bootstrap
        self.subsample_ratio = subsample_ratio
        self.high_threshold = high_confidence_threshold
        self.low_threshold = low_confidence_threshold
        self.edge_threshold = edge_threshold

        self.notears = NotearsAlgorithm(lambda1=notears_lambda)

        # Track results for inspection
        self._last_stability_scores: Dict[Tuple[str, str], float] = {}
        self._last_classification: Dict[str, List[Dict[str, Any]]] = {}

    def compute_stability_scores(
        self,
        data: pd.DataFrame,
        seed: int = 42
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute bootstrap stability scores for all potential edges.

        Per PDF Section 6.1:
        - Generate B subsamples of size N/2 without replacement
        - Run NOTEARS on each subsample
        - Calculate selection probability as count / B

        Args:
            data: User x Concept mastery DataFrame
            seed: Random seed for reproducibility

        Returns:
            Dict mapping (source, target) tuples to selection probabilities
        """
        np.random.seed(seed)
        n_samples = len(data)
        subsample_size = int(n_samples * self.subsample_ratio)

        if subsample_size < 10:
            logger.warning(f"Subsample size too small ({subsample_size}), using minimum of 10")
            subsample_size = min(10, n_samples)

        if n_samples < subsample_size:
            logger.warning("Not enough samples for subsampling, using full data")
            subsample_size = n_samples

        # Track edge occurrences across bootstrap samples
        edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        edge_weights: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        successful_samples = 0

        logger.info(f"Running {self.n_bootstrap} bootstrap samples with {subsample_size}/{n_samples} rows each")

        for b in range(self.n_bootstrap):
            try:
                # Subsample without replacement (per PDF Section 6.1)
                indices = np.random.choice(n_samples, size=subsample_size, replace=False)
                subsample = data.iloc[indices]

                # Run NOTEARS on subsample
                edges = self.notears.run(subsample, threshold=self.edge_threshold)

                for source, target, weight in edges:
                    key = (source, target)
                    edge_counts[key] += 1
                    edge_weights[key].append(weight)

                successful_samples += 1

            except Exception as e:
                logger.warning(f"Bootstrap sample {b} failed: {e}")
                continue

            if (b + 1) % 20 == 0:
                logger.info(f"Completed {b + 1}/{self.n_bootstrap} bootstrap samples")

        if successful_samples == 0:
            logger.error("All bootstrap samples failed")
            return {}

        # Calculate selection probabilities
        stability_scores = {}
        for edge, count in edge_counts.items():
            # Selection probability
            prob = count / successful_samples
            stability_scores[edge] = prob

        self._last_stability_scores = stability_scores
        logger.info(f"Computed stability scores for {len(stability_scores)} edges from {successful_samples} samples")

        return stability_scores

    def classify_edges(
        self,
        stability_scores: Dict[Tuple[str, str], float],
        edge_weights: Dict[Tuple[str, str], List[float]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Classify edges into verified, hypothetical, and discarded categories.

        Per PDF Section 6.2:
        - High Confidence (>0.85): verified - stable edges for production graph
        - Medium Confidence (0.5-0.85): hypothetical - candidate edges needing validation
        - Low Confidence (<0.5): discarded - noise to be filtered out

        Args:
            stability_scores: Dict mapping edges to selection probabilities
            edge_weights: Optional dict of edge weight samples for averaging

        Returns:
            Dict with 'verified', 'hypothetical', and 'discarded' lists
        """
        classified = {
            "verified": [],
            "hypothetical": [],
            "discarded": []
        }

        for (source, target), confidence in stability_scores.items():
            # Calculate average weight if available
            avg_weight = 1.0
            if edge_weights and (source, target) in edge_weights:
                weights = edge_weights[(source, target)]
                avg_weight = float(np.mean(weights)) if weights else 1.0

            edge_info = {
                "source": source,
                "target": target,
                "confidence": round(confidence, 4),
                "weight": round(avg_weight, 4),
                "type": "directed",
                "source_algo": "notears_bootstrap"
            }

            if confidence > self.high_threshold:
                edge_info["status"] = "verified"
                classified["verified"].append(edge_info)
            elif confidence >= self.low_threshold:
                edge_info["status"] = "hypothetical"
                classified["hypothetical"].append(edge_info)
            else:
                edge_info["status"] = "discarded"
                classified["discarded"].append(edge_info)

        self._last_classification = classified

        logger.info(
            f"Edge classification: {len(classified['verified'])} verified, "
            f"{len(classified['hypothetical'])} hypothetical, "
            f"{len(classified['discarded'])} discarded"
        )

        return classified

    def run(
        self,
        data: pd.DataFrame,
        seed: int = 42
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Run full bootstrap stability selection pipeline.

        Args:
            data: User x Concept mastery DataFrame
            seed: Random seed for reproducibility

        Returns:
            Tuple of (edges_to_persist, confidence_distribution)
            - edges_to_persist: List of edge dicts with confidence >= low_threshold
            - confidence_distribution: Dict with counts by status
        """
        if data.empty or data.shape[1] < 2:
            logger.warning("Insufficient data for bootstrap stability selection")
            return [], {"verified": 0, "hypothetical": 0, "discarded": 0}

        # Compute stability scores
        stability_scores = self.compute_stability_scores(data, seed)

        if not stability_scores:
            return [], {"verified": 0, "hypothetical": 0, "discarded": 0}

        # Classify edges
        classified = self.classify_edges(stability_scores)

        # Only return verified + hypothetical for persistence
        edges_to_persist = classified["verified"] + classified["hypothetical"]

        confidence_distribution = {
            "verified": len(classified["verified"]),
            "hypothetical": len(classified["hypothetical"]),
            "discarded": len(classified["discarded"])
        }

        return edges_to_persist, confidence_distribution

    def run_quick(
        self,
        data: pd.DataFrame,
        n_samples: int = 20,
        seed: int = 42
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Run a quicker version with fewer bootstrap samples.

        Useful for development/testing or when computational resources are limited.

        Args:
            data: User x Concept mastery DataFrame
            n_samples: Number of bootstrap samples (default 20)
            seed: Random seed

        Returns:
            Same as run()
        """
        original_n = self.n_bootstrap
        self.n_bootstrap = n_samples

        try:
            result = self.run(data, seed)
        finally:
            self.n_bootstrap = original_n

        return result


# Singleton instance with default configuration
bootstrap_selector = BootstrapStabilitySelector()
