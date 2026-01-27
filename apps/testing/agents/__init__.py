"""
Agentic Testing Framework for NerdLearn

This module implements advanced AI-powered testing using concepts from
"Building Agents for Codebase Testing" and "Building Code Testing Agents":

Core Testing Concepts:
- Gravitational Biasing: Goal-vector focused testing
- Antigravity Vector Search: Semantic relevance filtering
- Follow the Cable: Causal chain validation
- Adversarial Peer: Chaos-monkey style robustness testing

Agent Role Matrix:
- The Architect: Topological Continuity, DAG validation
- The Verifier: Semantic Truth, Knowledge Graph alignment
- The Peer: System Resilience, adversarial testing
- The Refiner: Coverage Optimization, TDFlow test generation

Key Features:
- TDFlow: Test-Driven Flow - generate tests before content
- Fuel Meter: Token/step limits to prevent runaway costs
- Topological Audit: DAG cycle/orphan detection
"""

from .base_verifier import VerifierAgent, GoalVector
from .adversarial_peer import AdversarialPeer, PoisonType
from .architect_agent import ArchitectAgent, PRReviewResult
from .refiner_agent import RefinerAgent, TDFlowPlan, GeneratedTest
from .fuel_meter import FuelMeter, FuelBudget, FuelUsageReport
from .topological_auditor import TopologicalAuditor, TopologyAuditResult
from .database_verifier import (
    DatabaseVerifier,
    DatabaseTestCategory,
    DatabaseTestResult,
    DatabaseAuditReport,
    create_database_verifier,
    DATABASE_GOAL_VECTORS
)
from .llm_client import (
    create_llm_client,
    get_default_client,
    LLMConfig,
    LLMProvider,
    BaseLLMClient,
    ClaudeClient,
    GeminiClient,
    MockLLMClient
)

__all__ = [
    # Verifier (Semantic Truth)
    'VerifierAgent',
    'GoalVector',

    # Peer (Resilience)
    'AdversarialPeer',
    'PoisonType',

    # Architect (Topological)
    'ArchitectAgent',
    'PRReviewResult',
    'TopologicalAuditor',
    'TopologyAuditResult',

    # Refiner (TDFlow)
    'RefinerAgent',
    'TDFlowPlan',
    'GeneratedTest',

    # Fuel Meter (Cost Control)
    'FuelMeter',
    'FuelBudget',
    'FuelUsageReport',

    # Database Verifier (Data Integrity)
    'DatabaseVerifier',
    'DatabaseTestCategory',
    'DatabaseTestResult',
    'DatabaseAuditReport',
    'create_database_verifier',
    'DATABASE_GOAL_VECTORS',

    # LLM Client (Multi-Provider)
    'create_llm_client',
    'get_default_client',
    'LLMConfig',
    'LLMProvider',
    'BaseLLMClient',
    'ClaudeClient',
    'GeminiClient',
    'MockLLMClient',
]

