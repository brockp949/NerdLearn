"""
Agent Package - LangGraph Multi-Agent Curriculum Generation System

This package implements the HiPlan-inspired hierarchical planning architecture
for autonomous curriculum generation (Generative Cognitive Architecture).

Core Agents (Curriculum Generation):
- BaseAgent: Abstract base class for all agents
- AgentGraph: LangGraph workflow orchestrator
- ArchitectAgent: Global planner (Arc of Learning)
- RefinerAgent: Local planner (Learning Outcomes)
- VerifierAgent: Quality auditor (Validation)
- HiPlanOrchestrator: Hierarchical planning workflow coordinator

GraphRAG (Knowledge Graph Augmentation):
- GraphRAGService: Community detection and global summarization
- ConceptCommunity: Concept cluster representation
- PrerequisiteExtractionAgent: Zero-shot dependency discovery

Content Aggregation:
- YouTubeTranscriptAgent: Video content extraction
- TranscriptQualityFilter: Educational content scoring
- ContentAggregationPipeline: Full content pipeline

Living Syllabus:
- WatchtowerAgent: Real-time domain monitoring
- SyllabusUpdateModule: Dynamic curriculum updates

Multi-Modal Transformation:
- StyleTransferPipeline: ELI5/Academic/Socratic transformations
- EducationalStyle: Supported style types

Agentic Social Layer (Phase 4):
- TeachableAgent: Feynman Protocol (learn by teaching)
- SimClassDebate: Multi-agent perspective debates
- CodeEvaluator: Agentic code review and evaluation
- TDDChallengeGenerator: Test-driven coding challenge generation

Research alignment:
- HiPlan: Hierarchical Planning for LLM Agents
- GraphRAG: Knowledge Graph-augmented retrieval (Microsoft)
- Bloom's Taxonomy: Cognitive level progression
- DSPy: Programmatic prompt optimization
- Louvain: Community detection algorithm
- Feynman Technique: Learning through teaching
- Social Constructivism: Knowledge built through interaction
- TDD: Test-Driven Development for verified assessments
"""
from .base_agent import BaseAgent, AgentState, AgentGraph, CurriculumConstraints
from .architect_agent import ArchitectAgent
from .refiner_agent import RefinerAgent, BLOOMS_TAXONOMY
from .verifier_agent import VerifierAgent, VerificationIssue
from .graphrag import GraphRAGService, GraphRAGResult, ConceptCommunity, get_graphrag_service
from .content_aggregator import (
    YouTubeTranscriptAgent,
    TranscriptQualityFilter,
    ContentAggregationPipeline,
    ContentQualityScore,
    AggregatedContent,
    ContentType,
    get_content_pipeline
)
from .watchtower_agent import (
    WatchtowerAgent,
    WatchtowerConfig,
    DomainUpdate,
    SyllabusUpdateModule,
    NewsSource,
    UpdatePriority,
    get_watchtower_agent
)
from .style_transfer import (
    StyleTransferPipeline,
    StyleTransferResult,
    EducationalStyle,
    STYLE_CONFIGS,
    get_style_transfer_pipeline
)
from .social import (
    # Teachable Agent (Feynman Protocol)
    TeachableAgent,
    StudentPersona,
    ComprehensionLevel,
    QuestionType,
    TeachingSession,
    get_teachable_agent,
    # SimClass Debates
    SimClassDebate,
    DebateRole,
    DebateFormat,
    DebateAgent,
    DebateSession,
    get_simclass_debate,
    # Code Evaluator
    CodeEvaluator,
    DifficultyLevel,
    EvaluationDimension,
    HintLevel,
    CodingChallenge,
    EvaluationResult,
    get_code_evaluator,
)
from .hiplan_orchestrator import (
    HiPlanOrchestrator,
    PlanningContext,
    PlanningPhase,
    get_hiplan_orchestrator,
)
from .prerequisite_extraction import (
    PrerequisiteExtractionAgent,
    PrerequisiteRelation,
    DependencyType,
    ConfidenceLevel,
    BridgeModule,
    ExtractionResult,
    get_prerequisite_agent,
)
from .tdd_challenge_generator import (
    TDDChallengeGenerator,
    CodingChallenge as TDDChallenge,
    TestCase,
    ExecutionResult,
    ProgrammingLanguage,
    DifficultyLevel as TDDDifficulty,
    ChallengeCategory,
    get_tdd_generator,
)

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentState",
    "AgentGraph",
    "CurriculumConstraints",
    # Core Agents
    "ArchitectAgent",
    "RefinerAgent",
    "VerifierAgent",
    # HiPlan Orchestrator
    "HiPlanOrchestrator",
    "PlanningContext",
    "PlanningPhase",
    "get_hiplan_orchestrator",
    # Prerequisite Extraction
    "PrerequisiteExtractionAgent",
    "PrerequisiteRelation",
    "DependencyType",
    "ConfidenceLevel",
    "BridgeModule",
    "ExtractionResult",
    "get_prerequisite_agent",
    # GraphRAG
    "GraphRAGService",
    "GraphRAGResult",
    "ConceptCommunity",
    "get_graphrag_service",
    # Content Aggregation
    "YouTubeTranscriptAgent",
    "TranscriptQualityFilter",
    "ContentAggregationPipeline",
    "ContentQualityScore",
    "AggregatedContent",
    "ContentType",
    "get_content_pipeline",
    # Watchtower (Living Syllabus)
    "WatchtowerAgent",
    "WatchtowerConfig",
    "DomainUpdate",
    "SyllabusUpdateModule",
    "NewsSource",
    "UpdatePriority",
    "get_watchtower_agent",
    # Style Transfer
    "StyleTransferPipeline",
    "StyleTransferResult",
    "EducationalStyle",
    "STYLE_CONFIGS",
    "get_style_transfer_pipeline",
    # Utilities
    "BLOOMS_TAXONOMY",
    "VerificationIssue",
    # Agentic Social Layer (Phase 4)
    # Teachable Agent
    "TeachableAgent",
    "StudentPersona",
    "ComprehensionLevel",
    "QuestionType",
    "TeachingSession",
    "get_teachable_agent",
    # SimClass Debates
    "SimClassDebate",
    "DebateRole",
    "DebateFormat",
    "DebateAgent",
    "DebateSession",
    "get_simclass_debate",
    # Code Evaluator
    "CodeEvaluator",
    "DifficultyLevel",
    "EvaluationDimension",
    "HintLevel",
    "CodingChallenge",
    "EvaluationResult",
    "get_code_evaluator",
    # TDD Challenge Generator
    "TDDChallengeGenerator",
    "TDDChallenge",
    "TestCase",
    "ExecutionResult",
    "ProgrammingLanguage",
    "TDDDifficulty",
    "ChallengeCategory",
    "get_tdd_generator",
]
