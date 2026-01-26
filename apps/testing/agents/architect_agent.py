"""
Architect Agent for CI/CD Integration

Reviews Pull Requests for architectural integrity and goal alignment.

From PDF:
"The NerdLearn 'Architect' role should be formalized in the SymbioGen
CI/CD pipeline. Every Pull Request should trigger not just a lint check,
but a visit from the 'Architect Agent'—a Gemini-powered process that
reads the diff, assesses its impact on the 'Opportunity Graph' topology,
and rejects changes that violate the 'Follow the Cable' mental model."
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PRReviewResult:
    """Result of PR architectural review"""
    approved: bool
    confidence: float
    concerns: List[str] = field(default_factory=list)
    architectural_impact: str = ""
    follow_cable_violations: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    risk_level: str = "low"  # low, medium, high, critical
    requires_human_review: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'approved': self.approved,
            'confidence': self.confidence,
            'concerns': self.concerns,
            'architectural_impact': self.architectural_impact,
            'follow_cable_violations': self.follow_cable_violations,
            'suggestions': self.suggestions,
            'risk_level': self.risk_level,
            'requires_human_review': self.requires_human_review,
            'timestamp': self.timestamp.isoformat()
        }
    
    def to_markdown_comment(self) -> str:
        """Format as GitHub PR comment"""
        emoji = "✅" if self.approved else "⚠️"
        
        comment = f"## {emoji} Architect Agent Review\n\n"
        comment += f"**Status:** {'Approved' if self.approved else 'Needs Attention'}\n"
        comment += f"**Confidence:** {self.confidence:.0%}\n"
        comment += f"**Risk Level:** {self.risk_level.upper()}\n\n"
        
        if self.architectural_impact:
            comment += f"### 🏗️ Architectural Impact\n{self.architectural_impact}\n\n"
        
        if self.concerns:
            comment += "### ⚠️ Concerns\n"
            for concern in self.concerns:
                comment += f"- {concern}\n"
            comment += "\n"
        
        if self.follow_cable_violations:
            comment += "### 🔗 Follow the Cable Violations\n"
            for violation in self.follow_cable_violations:
                comment += f"- ❌ {violation}\n"
            comment += "\n"
        
        if self.suggestions:
            comment += "### 💡 Suggestions\n"
            for suggestion in self.suggestions:
                comment += f"- {suggestion}\n"
            comment += "\n"
        
        if self.requires_human_review:
            comment += "---\n"
            comment += "🔔 **Human review recommended** due to complexity or risk level.\n"
        
        return comment


class ArchitectAgent:
    """
    Reviews Pull Requests for architectural integrity.
    
    Implements 'Follow the Cable' mental model enforcement.
    
    Checks:
    1. Does it violate 'Follow the Cable' (causal reasoning)?
    2. Does it introduce hallucination risks?
    3. Does it maintain goal-vector alignment?
    4. What is the architectural impact on the system?
    """
    
    def __init__(self, llm_client, config: Optional[Dict] = None):
        """
        Initialize Architect Agent.
        
        Args:
            llm_client: LLM interface for PR analysis
            config: Optional configuration for review criteria
        """
        self.llm = llm_client
        self.config = config or {}
        self.review_history: List[PRReviewResult] = []
    
    async def review_pr(
        self,
        diff: str,
        pr_description: str,
        pr_title: str = "",
        files_changed: Optional[List[str]] = None
    ) -> PRReviewResult:
        """
        Analyze PR diff for architectural impact and integrity.
        
        Args:
            diff: Git diff of the changes
            pr_description: PR description text
            pr_title: PR title
            files_changed: List of files modified
        
        Returns:
            PRReviewResult with approval status and feedback
        """
        logger.info(f"Reviewing PR: {pr_title}")
        
        # Build comprehensive review prompt
        prompt = self._build_review_prompt(
            diff, pr_description, pr_title, files_changed
        )
        
        try:
            # Get LLM review
            response = await self.llm.generate(prompt)
            
            # Parse response
            result = self._parse_review_response(response)
            
            # Apply risk assessment
            result = self._assess_risk_level(result, diff, files_changed)
            
            # Store in history
            self.review_history.append(result)
            
            logger.info(
                f"Review complete: {'Approved' if result.approved else 'Flagged'} "
                f"(confidence: {result.confidence:.2f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Review failed: {e}")
            return PRReviewResult(
                approved=False,
                confidence=0.0,
                concerns=[f"Review error: {str(e)}"],
                requires_human_review=True,
                risk_level="critical"
            )
    
    def _build_review_prompt(
        self,
        diff: str,
        pr_description: str,
        pr_title: str,
        files_changed: Optional[List[str]]
    ) -> str:
        """Build the architect review prompt"""
        
        files_list = "\n".join(files_changed) if files_changed else "Not provided"
        
        return f"""You are an Architect Agent reviewing a Pull Request for the NerdLearn platform.

PR TITLE: {pr_title}

PR DESCRIPTION:
{pr_description}

FILES CHANGED:
{files_list}

DIFF:
{diff[:5000]}  # Truncate very long diffs

REVIEW CRITERIA:

1. **Follow the Cable (Causal Reasoning)**
   - Does this change maintain clear causal chains?
   - Are dependencies properly sequenced (prerequisites before dependents)?
   - Does it break any existing learning paths or inference chains?

2. **Goal Alignment**
   - Does it serve the system's core objectives (adaptive learning, knowledge mastery)?
   - Is it aligned with NerdLearn's pedagogical principles?
   - Does it maintain or improve learner outcomes?

3. **Semantic Integrity**
   - Could this introduce hallucinations or incorrect inferences?
   - Does it maintain semantic correctness in LLM interactions?
   - Are embeddings and vector operations sound?

4. **Architectural Impact**
   - How does this affect the overall system topology?
   - Does it introduce new dependencies or coupling?
   - Are microservice boundaries respected?
   - Database schema changes handled correctly?

5. **Testing & Verification**
   - Are appropriate tests included?
   - Is the change verifiable?

Respond in JSON format:
{{
    "approved": true/false,
    "confidence": 0.0-1.0,
    "concerns": ["list of concerns, empty if none"],
    "architectural_impact": "description of how this affects the system",
    "follow_cable_violations": ["violations of causal reasoning, empty if none"],
    "suggestions": ["improvement suggestions"]
}}

Be thorough but fair. Focus on architectural soundness, not style preferences.
"""
    
    def _parse_review_response(self, response: str) -> PRReviewResult:
        """Parse LLM review response into structured result"""
        import json
        
        try:
            # Try direct JSON parse
            data = json.loads(response)
        except json.JSONDecodeError:
            # Extract from markdown code block
            if '```json' in response:
                json_start = response.find('```json') + 7
                json_end = response.find('```', json_start)
                json_str = response[json_start:json_end].strip()
                data = json.loads(json_str)
            elif '```' in response:
                json_start = response.find('```') + 3
                json_end = response.find('```', json_start)
                json_str = response[json_start:json_end].strip()
                data = json.loads(json_str)
            else:
                raise ValueError("Could not parse review response")
        
        return PRReviewResult(
            approved=data.get('approved', False),
            confidence=float(data.get('confidence', 0.0)),
            concerns=data.get('concerns', []),
            architectural_impact=data.get('architectural_impact', ''),
            follow_cable_violations=data.get('follow_cable_violations', []),
            suggestions=data.get('suggestions', [])
        )
    
    def _assess_risk_level(
        self,
        result: PRReviewResult,
        diff: str,
        files_changed: Optional[List[str]]
    ) -> PRReviewResult:
        """
        Assess risk level based on various factors.
        
        Risk indicators:
        - Database migrations
        - Core algorithm changes
        - API contract changes
        - Many files changed
        - Low confidence
        - Follow the Cable violations
        """
        
        risk_score = 0
        
        # Low confidence is risky
        if result.confidence < 0.5:
            risk_score += 3
        elif result.confidence < 0.7:
            risk_score += 1
        
        # Follow the Cable violations
        if result.follow_cable_violations:
            risk_score += len(result.follow_cable_violations) * 2
        
        # Concerns raised
        if result.concerns:
            risk_score += len(result.concerns)
        
        # Critical file patterns
        critical_patterns = [
            'migration', 'schema', 'models.py', 'database',
            'bkt', 'fsrs', 'bayesian', 'llm_client', 'core'
        ]
        
        if files_changed:
            for file in files_changed:
                if any(pattern in file.lower() for pattern in critical_patterns):
                    risk_score += 2
            
            # Many files changed
            if len(files_changed) > 10:
                risk_score += 2
        
        # Determine risk level
        if risk_score >= 8:
            result.risk_level = "critical"
            result.requires_human_review = True
        elif risk_score >= 5:
            result.risk_level = "high"
            result.requires_human_review = True
        elif risk_score >= 3:
            result.risk_level = "medium"
        else:
            result.risk_level = "low"
        
        return result
    
    def get_review_statistics(self) -> Dict:
        """Get statistics on review history"""
        if not self.review_history:
            return {'total_reviews': 0}
        
        total = len(self.review_history)
        approved = sum(1 for r in self.review_history if r.approved)
        
        return {
            'total_reviews': total,
            'approval_rate': approved / total,
            'avg_confidence': sum(r.confidence for r in self.review_history) / total,
            'risk_distribution': self._get_risk_distribution()
        }
    
    def _get_risk_distribution(self) -> Dict[str, int]:
        """Get distribution of risk levels"""
        distribution = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for review in self.review_history:
            distribution[review.risk_level] += 1
        return distribution
