"""
Educational Style Transfer Module - Universal Translator for Complexity

Research alignment:
- DSPy-inspired: Programmatic prompt optimization
- Persona-based Transformation: ELI5, Academic, Socratic modes
- Multi-step Pipeline: Extraction → Transformation → Verification
- Fact Retention: Ensure no information distortion during style transfer

Key Features:
1. Extract propositional logic from source content
2. Transform to target persona while preserving facts
3. Verify no information was lost or distorted
4. Support multiple educational styles
"""
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EducationalStyle(str, Enum):
    """Supported educational styles for content transformation"""
    ELI5 = "eli5"                    # Explain Like I'm 5
    ACADEMIC = "academic"            # Formal academic style
    SOCRATIC = "socratic"            # Question-based teaching
    CONVERSATIONAL = "conversational"  # Casual, podcast-style
    TECHNICAL = "technical"          # Precise technical documentation
    NARRATIVE = "narrative"          # Story-based learning
    VISUAL_GUIDE = "visual_guide"    # Instructions for visual learners


@dataclass
class StyleTransferResult:
    """Result of a style transfer operation"""
    original_text: str
    transformed_text: str
    target_style: EducationalStyle
    propositions_extracted: List[str]
    facts_preserved: bool
    fact_preservation_score: float  # 0-1
    style_adherence_score: float    # 0-1
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StyleConfig:
    """Configuration for a specific educational style"""
    name: str
    description: str
    system_prompt: str
    transformation_instructions: str
    example_output: str
    suitable_for: List[str]  # e.g., ["beginners", "visual learners"]


# Style configurations
STYLE_CONFIGS: Dict[EducationalStyle, StyleConfig] = {
    EducationalStyle.ELI5: StyleConfig(
        name="Explain Like I'm 5",
        description="Simple explanations using everyday analogies",
        system_prompt="""You explain complex topics using simple words and everyday analogies.
Imagine explaining to a curious 5-year-old who asks lots of "why" questions.
Use concrete, tangible examples from daily life.""",
        transformation_instructions="""Transform this content:
1. Replace jargon with simple words
2. Use analogies involving household objects, toys, or playground activities
3. Break complex processes into simple steps
4. Use "like" and "imagine" to create mental pictures
5. Keep sentences short (under 15 words)""",
        example_output="""Imagine your brain is like a big library. When you learn something new,
it's like putting a new book on the shelf. Neural networks are like tiny librarians
that help you find the right book when you need it!""",
        suitable_for=["beginners", "visual learners", "non-technical audiences"]
    ),

    EducationalStyle.ACADEMIC: StyleConfig(
        name="Academic",
        description="Formal academic writing with precise terminology",
        system_prompt="""You write in formal academic style suitable for scholarly publications.
Use precise technical terminology, cite theoretical frameworks, and maintain
objectivity. Structure content with clear logical progression.""",
        transformation_instructions="""Transform this content:
1. Use precise technical vocabulary
2. Reference established frameworks and theories
3. Employ formal sentence structures
4. Use passive voice where appropriate
5. Include qualifications and hedging language""",
        example_output="""The neural network architecture demonstrates significant performance
improvements in classification tasks. According to the universal approximation theorem,
multilayer perceptrons can approximate any continuous function given sufficient neurons.""",
        suitable_for=["graduate students", "researchers", "professionals"]
    ),

    EducationalStyle.SOCRATIC: StyleConfig(
        name="Socratic",
        description="Question-based teaching that guides discovery",
        system_prompt="""You teach using the Socratic method - guiding learners to discover
answers through strategic questions rather than direct explanation.
Each question should build on previous understanding.""",
        transformation_instructions="""Transform this content:
1. Identify the core concept to be discovered
2. Formulate a sequence of questions leading to insight
3. Start with what the learner already knows
4. Use "What if..." and "Why do you think..." questions
5. Never directly state the answer - guide toward it""",
        example_output="""Let's think about this together. What do you think happens when
you try to predict something based on past experience?
...Interesting! And what if you had thousands of examples instead of just a few?
...Exactly! Now, what might happen if those examples had patterns you could learn from?""",
        suitable_for=["active learners", "critical thinkers", "deep understanding"]
    ),

    EducationalStyle.CONVERSATIONAL: StyleConfig(
        name="Conversational",
        description="Casual, engaging podcast-style discussion",
        system_prompt="""You explain topics in a casual, conversational manner like
a friendly expert chatting over coffee. Use natural speech patterns,
occasional humor, and relatable examples.""",
        transformation_instructions="""Transform this content:
1. Use "you" and "we" to create connection
2. Include natural speech patterns and transitions
3. Add occasional light humor or relatable observations
4. Break up dense content with reactions ("Pretty cool, right?")
5. Use contractions and informal language""",
        example_output="""So here's the thing about neural networks - they're basically
pattern-recognition machines on steroids. You know how you can recognize your friend's
face even in a crowd? That's kind of what these do, but for data. Pretty wild, actually.""",
        suitable_for=["podcast listeners", "casual learners", "commuters"]
    ),

    EducationalStyle.TECHNICAL: StyleConfig(
        name="Technical",
        description="Precise technical documentation with specifications",
        system_prompt="""You write precise technical documentation suitable for
implementation. Include specifications, parameters, and edge cases.
Prioritize accuracy and completeness over readability.""",
        transformation_instructions="""Transform this content:
1. Use precise technical terminology
2. Include specifications and parameters
3. Document edge cases and exceptions
4. Use bullet points and structured formats
5. Provide implementation details where relevant""",
        example_output="""Neural Network Layer Configuration:
- Type: Dense (fully connected)
- Units: 128
- Activation: ReLU (Rectified Linear Unit)
- Input Shape: (batch_size, 784)
- Weight Initialization: He normal
- Regularization: L2 (lambda=0.001)""",
        suitable_for=["developers", "engineers", "implementers"]
    ),

    EducationalStyle.NARRATIVE: StyleConfig(
        name="Narrative",
        description="Story-based learning with characters and plot",
        system_prompt="""You teach through storytelling. Create characters, scenarios,
and narrative arcs that naturally introduce educational concepts.
Make abstract ideas concrete through story events.""",
        transformation_instructions="""Transform this content:
1. Create a relatable character or scenario
2. Introduce concepts through story events
3. Use conflict/problem → solution structure
4. Make abstract ideas concrete through actions
5. Include emotional elements for engagement""",
        example_output="""Meet Maya, a young data scientist facing her biggest challenge yet.
Her team's model was predicting customer behavior incorrectly, and the deadline was tomorrow.
"What if," she wondered, "we're not looking at enough examples?" That night, she discovered
the power of larger training datasets, and everything changed...""",
        suitable_for=["storytelling learners", "engagement seekers", "memory retention"]
    ),

    EducationalStyle.VISUAL_GUIDE: StyleConfig(
        name="Visual Guide",
        description="Instructions optimized for visual learning",
        system_prompt="""You create content optimized for visual representation.
Describe concepts in terms of diagrams, flows, and spatial relationships.
Use visual language and reference shapes, colors, and arrangements.""",
        transformation_instructions="""Transform this content:
1. Describe in terms of visual elements (boxes, arrows, flows)
2. Use spatial language (above, below, connects to)
3. Reference colors or visual indicators
4. Structure as step-by-step visual instructions
5. Include suggestions for diagram creation""",
        example_output="""[DIAGRAM DESCRIPTION]
Picture three stacked layers, each as a horizontal row of circles:
- TOP (Output): 3 green circles - these are your predictions
- MIDDLE (Hidden): 8 blue circles - these find patterns
- BOTTOM (Input): 10 yellow circles - this is your data
Draw arrows connecting EVERY circle in one layer to ALL circles in the next layer.""",
        suitable_for=["visual learners", "diagram creators", "spatial thinkers"]
    )
}


class PropositionalExtractor:
    """
    Extracts propositional logic (raw facts) from educational content

    This is the first step in style transfer - separating facts from presentation.
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    async def extract(self, text: str) -> List[str]:
        """
        Extract propositions (atomic facts) from text

        Args:
            text: Educational content

        Returns:
            List of proposition strings
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You extract atomic propositions (facts) from educational text.

A proposition is a single, verifiable statement of fact.

Rules:
1. One fact per proposition
2. Remove all style, tone, and pedagogical framing
3. Preserve technical accuracy exactly
4. Use neutral, declarative language
5. Include all facts - do not summarize

Output as JSON array of strings."""),
            ("human", """Extract propositions from:

{text}

Output JSON array:""")
        ])

        try:
            messages = prompt.format_messages(text=text[:3000])
            response = await self.llm.ainvoke(messages)

            # Parse JSON
            response_text = response.content
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end]
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end]

            propositions = json.loads(response_text.strip())
            return propositions if isinstance(propositions, list) else []

        except Exception as e:
            logger.error(f"Error extracting propositions: {e}")
            return []


class StyleTransformer:
    """
    Transforms propositional content into target educational style

    This is the second step - applying style while preserving facts.
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0.7)

    async def transform(
        self,
        propositions: List[str],
        target_style: EducationalStyle,
        context: Optional[str] = None
    ) -> str:
        """
        Transform propositions into target style

        Args:
            propositions: List of extracted facts
            target_style: Target educational style
            context: Optional context about the learner/topic

        Returns:
            Transformed content in target style
        """
        config = STYLE_CONFIGS.get(target_style)
        if not config:
            logger.error(f"Unknown style: {target_style}")
            return "\n".join(propositions)

        prompt = ChatPromptTemplate.from_messages([
            ("system", config.system_prompt),
            ("human", """Transform these facts into {style_name} style:

FACTS TO INCLUDE:
{propositions}

{transformation_instructions}

Context: {context}

Example of target style:
{example}

Write the transformed content:""")
        ])

        try:
            messages = prompt.format_messages(
                style_name=config.name,
                propositions="\n".join(f"- {p}" for p in propositions),
                transformation_instructions=config.transformation_instructions,
                context=context or "General educational content",
                example=config.example_output
            )

            response = await self.llm.ainvoke(messages)
            return response.content

        except Exception as e:
            logger.error(f"Error transforming to {target_style}: {e}")
            return "\n".join(propositions)


class FactVerifier:
    """
    Verifies that no facts were lost or distorted during transformation

    This is the third step - quality assurance for accuracy.
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    async def verify(
        self,
        original_propositions: List[str],
        transformed_text: str
    ) -> Tuple[bool, float, List[str]]:
        """
        Verify facts are preserved in transformed text

        Args:
            original_propositions: Original facts
            transformed_text: Transformed content

        Returns:
            (all_facts_preserved, preservation_score, issues_found)
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You verify that educational content transformations preserve all facts.

For each original fact, check if it's accurately represented in the transformed text.

Scoring:
- Present and accurate: 1.0
- Present but slightly modified: 0.8
- Partially present/implied: 0.5
- Missing or distorted: 0.0

Output JSON:
{{
    "fact_scores": {{"fact_text": score, ...}},
    "overall_preservation": 0.0-1.0,
    "issues": ["list of problems found"],
    "suggestions": ["list of fixes needed"]
}}"""),
            ("human", """Original Facts:
{facts}

Transformed Text:
{transformed}

Verify fact preservation:""")
        ])

        try:
            messages = prompt.format_messages(
                facts="\n".join(f"- {p}" for p in original_propositions),
                transformed=transformed_text[:3000]
            )

            response = await self.llm.ainvoke(messages)

            # Parse JSON
            response_text = response.content
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end]

            result = json.loads(response_text.strip())

            preservation_score = result.get("overall_preservation", 0.5)
            all_preserved = preservation_score >= 0.9
            issues = result.get("issues", [])

            return (all_preserved, preservation_score, issues)

        except Exception as e:
            logger.error(f"Error verifying facts: {e}")
            return (False, 0.5, [f"Verification error: {str(e)}"])


class StyleTransferPipeline:
    """
    Complete style transfer pipeline

    Implements the multi-step process:
    1. Extraction: Extract propositional logic from source
    2. Transformation: Rewrite in target style
    3. Verification: Ensure fact preservation
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0.5)
        self.extractor = PropositionalExtractor(self.llm)
        self.transformer = StyleTransformer(self.llm)
        self.verifier = FactVerifier(self.llm)

    async def transfer(
        self,
        text: str,
        target_style: EducationalStyle,
        context: Optional[str] = None,
        max_retries: int = 2
    ) -> StyleTransferResult:
        """
        Execute complete style transfer

        Args:
            text: Original educational content
            target_style: Target style
            context: Optional learner context
            max_retries: Max retry attempts for verification failures

        Returns:
            StyleTransferResult with transformed content
        """
        logger.info(f"Starting style transfer to {target_style.value}")

        # Step 1: Extract propositions
        logger.debug("Extracting propositions...")
        propositions = await self.extractor.extract(text)

        if not propositions:
            logger.warning("No propositions extracted")
            return StyleTransferResult(
                original_text=text,
                transformed_text=text,
                target_style=target_style,
                propositions_extracted=[],
                facts_preserved=False,
                fact_preservation_score=0.0,
                style_adherence_score=0.0,
                issues=["Could not extract propositions from source text"]
            )

        logger.debug(f"Extracted {len(propositions)} propositions")

        # Step 2 & 3: Transform and verify (with retries)
        best_result = None
        best_score = 0.0

        for attempt in range(max_retries + 1):
            # Transform
            logger.debug(f"Transformation attempt {attempt + 1}")
            transformed_text = await self.transformer.transform(
                propositions,
                target_style,
                context
            )

            # Verify
            logger.debug("Verifying fact preservation...")
            facts_preserved, preservation_score, issues = await self.verifier.verify(
                propositions,
                transformed_text
            )

            # Track best result
            if preservation_score > best_score:
                best_score = preservation_score
                best_result = (transformed_text, facts_preserved, preservation_score, issues)

            # If good enough, return
            if facts_preserved:
                logger.info("Fact verification passed")
                break

            logger.debug(f"Fact verification failed (score: {preservation_score}), retrying...")

        # Use best result
        transformed_text, facts_preserved, preservation_score, issues = best_result

        # Assess style adherence
        style_score = await self._assess_style_adherence(transformed_text, target_style)

        return StyleTransferResult(
            original_text=text,
            transformed_text=transformed_text,
            target_style=target_style,
            propositions_extracted=propositions,
            facts_preserved=facts_preserved,
            fact_preservation_score=preservation_score,
            style_adherence_score=style_score,
            issues=issues,
            metadata={
                "attempts": attempt + 1,
                "proposition_count": len(propositions)
            }
        )

    async def _assess_style_adherence(
        self,
        text: str,
        target_style: EducationalStyle
    ) -> float:
        """Assess how well the text adheres to the target style"""
        config = STYLE_CONFIGS.get(target_style)
        if not config:
            return 0.5

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You assess how well educational content adheres to a specific style.

Score from 0.0 to 1.0:
- 1.0: Perfect adherence to style
- 0.7: Good adherence with minor deviations
- 0.5: Moderate adherence
- 0.3: Poor adherence
- 0.0: Completely wrong style"""),
            ("human", """Target Style: {style_name}
Style Description: {style_description}

Content to assess:
{content}

Provide only a single number (0.0-1.0):""")
        ])

        try:
            messages = prompt.format_messages(
                style_name=config.name,
                style_description=config.description,
                content=text[:1500]
            )

            response = await self.llm.ainvoke(messages)
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.error(f"Error assessing style: {e}")
            return 0.5

    def get_available_styles(self) -> List[Dict[str, Any]]:
        """Get information about available styles"""
        return [
            {
                "id": style.value,
                "name": config.name,
                "description": config.description,
                "suitable_for": config.suitable_for
            }
            for style, config in STYLE_CONFIGS.items()
        ]


# Lazy-initialized global instance
_style_transfer_pipeline: Optional[StyleTransferPipeline] = None


def get_style_transfer_pipeline() -> StyleTransferPipeline:
    """Get or create the style transfer pipeline singleton (lazy initialization)"""
    global _style_transfer_pipeline
    if _style_transfer_pipeline is None:
        _style_transfer_pipeline = StyleTransferPipeline()
    return _style_transfer_pipeline
