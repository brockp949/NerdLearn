
import dspy
import os
import sys
import logging

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.config import settings
from app.core.dspy_config import configure_dspy
from app.services.dspy_optimizer import DSPyOptimizer, StyleTransferModule, fact_retention_metric, socratic_score_metric

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dataset():
    """
    Create a small synthetic dataset for style transfer optimization.
    """
    dataset = [
        # ELI5 Examples
        dspy.Example(
            content="Mitochondria are membrane-bound cell organelles (mitochondrion, singular) that generate most of the chemical energy needed to power the cell's biochemical reactions. Chemical energy produced by the mitochondria is stored in a small molecule called adenosine triphosphate (ATP).",
            persona="ELI5",
            target_audience="Elementary School Student",
            facts=["Mitochondria are organelles", "Generate energy", "Energy stored as ATP"]
        ).with_inputs('content', 'persona', 'target_audience'),

        dspy.Example(
            content="Quantum entanglement is a physical phenomenon that occurs when a group of particles are generated, interact, or share spatial proximity in a way such that the quantum state of each particle of the group cannot be described independently of the state of the others, including when the particles are separated by a large distance.",
            persona="ELI5",
            target_audience="Curious Kid",
            facts=["Particles are linked", "State depends on others", "Distance doesn't matter"]
        ).with_inputs('content', 'persona', 'target_audience'),

        # Socratic Examples
        dspy.Example(
            content="The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse side is equal to the sum of squares of the other two sides.",
            persona="Socratic",
            target_audience="High School Student",
            facts=["Right-angled triangle", "hypotenuse^2 = side1^2 + side2^2"]
        ).with_inputs('content', 'persona', 'target_audience'),

         dspy.Example(
            content="Supply and demand is an economic model of price determination in a market. It postulates that, holding all else equal, in a competitive market, the unit price for a particular good, or other traded item such as labor or liquid financial assets, will vary until it settles at a point where the quantity demanded (at the current price) will equal the quantity supplied (at the current price), resulting in an economic equilibrium for price and quantity transacted.",
            persona="Socratic",
            target_audience="Economics Student",
            facts=["Price determination model", "Supply equals demand at equilibrium"]
        ).with_inputs('content', 'persona', 'target_audience'),
    ]
    return dataset

def combined_metric(gold, pred, trace=None):
    """
    Weighted metric: 60% Fact Retention, 40% Socratic (if applicable)
    """
    fact_score = fact_retention_metric(gold, pred, trace)
    socratic_score = socratic_score_metric(gold, pred, trace)
    
    # If Socratic, weight both. If not, just use fact score.
    target_persona = getattr(gold, 'persona', '').lower()
    
    if 'socratic' in target_persona:
        return (fact_score * 0.6) + (socratic_score * 0.4)
    else:
        return fact_score

def main():
    logger.info("Initializing DSPy...")
    configure_dspy()
    
    logger.info("Creating dataset...")
    trainset = create_dataset()
    
    logger.info("Initializing Optimizer...")
    optimizer_manager = DSPyOptimizer()
    
    # Custom optimization call to use our combined metric
    from dspy.teleprompt import BootstrapFewShot
    
    teleprompter = BootstrapFewShot(
        metric=combined_metric,
        max_bootstrapped_demos=2,
        max_labeled_demos=0 # We rely on the metric to select best generations
    )
    
    logger.info("Compiling Style Transfer Module...")
    optimized_program = teleprompter.compile(
        optimizer_manager.pipeline,
        trainset=trainset,
    )
    
    logger.info("Optimization Complete!")
    
    # Save the optimized program
    output_path = os.path.join(os.path.dirname(__file__), '../app/services/optimized_style_transfer.json')
    optimized_program.save(output_path)
    logger.info(f"Saved optimized program to {output_path}")

    # Test it
    test_ex = trainset[0]
    pred = optimized_program(content=test_ex.content, persona=test_ex.persona, target_audience=test_ex.target_audience)
    print(f"\nOriginal: {test_ex.content}")
    print(f"Persona: {test_ex.persona}")
    print(f"Transformed: {pred.transformed_content}")

if __name__ == "__main__":
    main()
