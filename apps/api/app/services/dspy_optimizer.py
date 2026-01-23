"""
DSPy Optimizer - Content Transformation Pipeline

Implements programmatic optimization for content style transfer.
Research alignment:
- DSPy Signatures: Declarative prompt definitions
- Metric Definitions: Logic for evaluating quality
- Teleprompter: Automatic prompt optimization
"""
import dspy
from typing import List, Dict, Any
import logging

from app.core.dspy_config import get_dspy_lm

logger = logging.getLogger(__name__)


# ================== Signatures ==================

class FactExtraction(dspy.Signature):
    """Extract core propositional logic and facts from educational content."""
    
    content = dspy.InputField(desc="The source educational content")
    facts = dspy.OutputField(desc="List of atomic facts and logical propositions")


class StyleTransfer(dspy.Signature):
    """Rewrite content into a specific educational persona while preserving facts."""
    
    content = dspy.InputField(desc="The original content")
    facts = dspy.InputField(desc="The core facts that must be preserved")
    persona = dspy.InputField(desc="Target persona (e.g., 'ELI5', 'Academic', 'Socratic')")
    target_audience = dspy.InputField(desc="Description of the learner")
    
    transformed_content = dspy.OutputField(desc="The rewritten content in the target persona")


class FactVerification(dspy.Signature):
    """Verify that the transformed content contains all original facts."""
    
    original_facts = dspy.InputField(desc="List of facts from original content")
    transformed_content = dspy.InputField(desc="The rewritten content")
    
    missing_facts = dspy.OutputField(desc="List of facts missing or distorted in the new content")
    score = dspy.OutputField(desc="float score 0.0-1.0 representing factual fidelity")


# ================== Modules ==================

class StyleTransferModule(dspy.Module):
    """
    Multi-step pipeline for style transfer:
    1. Extract facts
    2. Rewrite content
    3. Verify and refine (optional)
    """
    
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(FactExtraction)
        self.rewrite = dspy.ChainOfThought(StyleTransfer)
        self.verify = dspy.ChainOfThought(FactVerification)

    def forward(self, content: str, persona: str, target_audience: str = "general learner", verify: bool = False):
        # 1. Extract facts
        extraction = self.extract(content=content)
        facts = extraction.facts
        
        # 2. Rewrite
        rewrite = self.rewrite(
            content=content,
            facts=facts,
            persona=persona,
            target_audience=target_audience
        )

        # 3. Verify (Optional)
        score = None
        missing_facts = []
        if verify:
            verification = self.verify(
                original_facts=facts,
                transformed_content=rewrite.transformed_content
            )
            try:
                score = float(verification.score)
            except:
                score = 0.0
            missing_facts = verification.missing_facts
        
        return dspy.Prediction(
            transformed_content=rewrite.transformed_content,
            facts=facts,
            score=score,
            missing_facts=missing_facts
        )



class SocraticEvaluation(dspy.Signature):
    """Evaluate if the content follows the Socratic method."""
    
    content = dspy.InputField(desc="The educational content to evaluate")
    
    is_socratic = dspy.OutputField(desc="Boolean, whether it asks guiding questions instead of giving answers")
    question_quality = dspy.OutputField(desc="Critique of the questions asked")
    score = dspy.OutputField(desc="float score 0.0-1.0 representing Socratic quality")


# ================== Metrics ==================

def socratic_score_metric(gold, pred, trace=None):
    """
    Metric to evaluate Socratic quality.
    Checks if the response uses questions to guide learning.
    """
    # Only evaluate if the intended persona was Socratic
    # (We assume the input example had 'persona'='Socratic')
    # However, 'pred' is a Prediction object which might not have the input 'persona' easily accessible 
    # unless we pass it through. 
    # In DSPy, 'gold' usually has the inputs.
    
    # Check if this example targets Socratic
    target_persona = getattr(gold, 'persona', '')
    if 'socratic' not in target_persona.lower():
        return 1.0 # Pass through for non-Socratic personas (don't penalize)

    evaluator = dspy.ChainOfThought(SocraticEvaluation)
    result = evaluator(content=pred.transformed_content)
    
    try:
        return float(result.score)
    except:
        return 0.0


def fact_retention_metric(gold, pred, trace=None):
    """
    Metric to evaluate if facts are preserved.
    Uses an LLM to judge semantic equivalence.
    """
    verifier = dspy.ChainOfThought(FactVerification)
    
    # Check original facts against predicted content
    result = verifier(
        original_facts=pred.facts,
        transformed_content=pred.transformed_content
    )
    
    try:
        score = float(result.score)
        return score
    except:
        return 0.0


# ================== Optimization ==================

class DSPyOptimizer:
    """Manager for optimizing DSPy modules"""
    
    def __init__(self):
        get_dspy_lm()  # Ensure configured
        self.pipeline = StyleTransferModule()
        
    def optimize(self, trainset: List[dspy.Example], valset: List[dspy.Example]):
        """
        Run the teleprompter optimization
        """
        from dspy.teleprompt import BootstrapFewShot
        
        teleprompter = BootstrapFewShot(
            metric=fact_retention_metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4
        )
        
        logger.info("Starting DSPy optimization...")
        optimized_program = teleprompter.compile(
            self.pipeline,
            trainset=trainset,
            valset=valset
        )
        
        return optimized_program

