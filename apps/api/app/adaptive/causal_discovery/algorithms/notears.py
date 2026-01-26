import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

class NotearsAlgorithm:
    """
    Implementation of the NOTEARS (Non-combinatorial Optimization via Trace Exponential
    and Augmented lagRangian Structure Learning) algorithm.
    
    Ref: Zheng et al. "DAGs with NO TEARS: Continuous Optimization for Structure Learning" (NeurIPS 2018)
    """
    
    def __init__(self, lambda1: float = 0.1, loss_type: str = 'l2', max_iter: int = 100, h_tol: float = 1e-8, w_threshold: float = 0.3):
        """
        Initialize NOTEARS.
        
        Args:
            lambda1: L1 penalty parameter (sparsity)
            loss_type: Loss function type ('l2', 'logistic', 'poisson')
            max_iter: Maximum iterations for augmented Lagrangian
            h_tol: Tolerance for acyclicity constraint
            w_threshold: Threshold for edge weights
        """
        self.lambda1 = lambda1
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.w_threshold = w_threshold

    def run(self, data: pd.DataFrame, threshold: float = None) -> List[Tuple[str, str, float]]:
        """
        Run NOTEARS on the provided data.
        
        Args:
            data: DataFrame with concepts as columns and user mastery as rows.
            threshold: Minimum edge weight to consider a causal link (overrides init).
            
        Returns:
            List of (source, target, weight) tuples.
        """
        if data.empty:
            logger.warning("Empty data provided to NOTEARS")
            return []

        # Convert to numpy array
        X = data.values
        labels = data.columns.tolist()
        d = X.shape[1]
        
        logger.info(f"Running NOTEARS on {X.shape[0]} samples with {d} variables")
        
        if threshold is None:
            threshold = self.w_threshold
            
        # Run optimization
        W_est = self._notears_linear(X, lambda1=self.lambda1, loss_type=self.loss_type, max_iter=self.max_iter, h_tol=self.h_tol, w_threshold=threshold)
        
        # Extract edges
        edges = []
        for i in range(d):
            for j in range(d):
                weight = W_est[i, j]
                if abs(weight) >= threshold and i != j:
                    source = labels[i]
                    target = labels[j]
                    edges.append((source, target, float(weight)))
                    
        logger.info(f"NOTEARS discovered {len(edges)} edges")
        return edges

    def _notears_linear(self, X, lambda1, loss_type, max_iter, h_tol, w_threshold):
        """
        Solve min_W L(W; X) + lambda1 * ||W||_1 s.t. h(W) = 0
        """
        def _loss(W):
            """Evaluate value and gradient of loss function"""
            M = X @ W
            if loss_type == 'l2':
                R = X - M
                loss = 0.5 / X.shape[0] * (R ** 2).sum()
                G_loss = - 1.0 / X.shape[0] * X.T @ R
            elif loss_type == 'logistic':
                loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
                G_loss = 1.0 / X.shape[0] * X.T @ (1.0 / (1.0 + np.exp(-M)) - X)
            elif loss_type == 'poisson':
                S = np.exp(M)
                loss = 1.0 / X.shape[0] * (S - X * M).sum()
                G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
            else:
                raise ValueError('unknown loss type')
            return loss, G_loss

        def _h(W):
            """Evaluate value and gradient of acyclicity constraint"""
            d = W.shape[0]
            # E = e^(W * W)
            M = np.eye(d) + W * W / d  # (Yu et al. 2019)
            E = np.linalg.matrix_power(M, d - 1)
            h = (E.T * M).sum() - d
            G_h = E.T * W * 2
            return h, G_h

        def _adj(w):
            """Convert doubled variables (w_pos, w_neg) back to W"""
            d = int(np.sqrt(w.size // 2))
            return (w[:d*d] - w[d*d:]).reshape([d, d])

        def _func(w):
            """Evaluate augmented Lagrangian"""
            W = _adj(w)
            loss, G_loss = _loss(W)
            h, G_h = _h(W)
            obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
            G_smooth = G_loss + (rho * h + alpha) * G_h
            g_obj = np.concatenate((G_smooth + lambda1, -G_smooth + lambda1), axis=None)
            return obj, g_obj

        n, d = X.shape
        w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
        bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
        
        for _ in range(max_iter):
            w_new, h_new = None, None
            while rho < 1e+20:
                sol = minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
                w_new = sol.x
                h_new, _ = _h(_adj(w_new))
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            w_est, h = w_new, h_new
            alpha += rho * h
            if h <= h_tol:
                break
                
        W_est = _adj(w_est)
        W_est[np.abs(W_est) < w_threshold] = 0
        return W_est