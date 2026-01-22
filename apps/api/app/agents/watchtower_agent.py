"""
Living Syllabus Watchtower Agent - Real-Time Curriculum Updates

Research alignment:
- "Living Syllabus" Blue Ocean Feature
- News-Driven Injection: Monitor data streams for breakthroughs
- Real-Time Grafting: Autonomously create update modules
- Graph-Based Prerequisite Healing: Detect and bridge knowledge gaps

Key Features:
1. Monitor relevant sources (arXiv, tech news, GitHub)
2. Detect significant domain developments
3. Match developments to syllabus modules
4. Generate update micro-modules
5. Notify users of curriculum updates
"""
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import hashlib

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NewsSource(str, Enum):
    """Supported news/update sources"""
    ARXIV = "arxiv"
    GITHUB_TRENDING = "github_trending"
    HACKER_NEWS = "hacker_news"
    TECH_NEWS = "tech_news"
    DOCUMENTATION = "documentation"


class UpdatePriority(str, Enum):
    """Priority levels for syllabus updates"""
    CRITICAL = "critical"  # Breaking changes, security issues
    HIGH = "high"         # Major developments, new paradigms
    MEDIUM = "medium"     # Notable advances, new tools
    LOW = "low"           # Minor updates, enhancements


@dataclass
class DomainUpdate:
    """Represents a detected update in the domain"""
    id: str
    source: NewsSource
    title: str
    summary: str
    url: str
    priority: UpdatePriority
    related_concepts: List[str]
    relevance_score: float  # 0-1
    detected_at: datetime
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyllabusUpdateModule:
    """A micro-module to graft into the syllabus"""
    id: str
    title: str
    description: str
    concepts_covered: List[str]
    estimated_minutes: int
    parent_module_id: Optional[str]  # Module to attach to
    prerequisite_concepts: List[str]
    content_type: str  # "briefing", "deep_dive", "practical"
    priority: UpdatePriority
    source_update: DomainUpdate


@dataclass
class WatchtowerConfig:
    """Configuration for the Watchtower agent"""
    poll_interval_minutes: int = 60
    min_relevance_score: float = 0.6
    max_updates_per_day: int = 5
    enabled_sources: List[NewsSource] = field(default_factory=lambda: [
        NewsSource.ARXIV,
        NewsSource.GITHUB_TRENDING,
        NewsSource.TECH_NEWS
    ])
    domains: List[str] = field(default_factory=list)  # e.g., ["machine learning", "python"]


class WatchtowerAgent:
    """
    Watchtower Agent - Monitors domain developments and updates syllabi

    This agent creates the "Living Syllabus" experience:
    1. Continuously monitors configured sources
    2. Detects developments relevant to user's courses
    3. Generates update modules when significant changes occur
    4. Maintains curriculum freshness without overwhelming learners

    Example Flow:
    1. User is taking "Quantum Computing" course
    2. arXiv paper announces room-temperature superconductor
    3. Watchtower detects this is relevant to Module 3 (High-Temperature Conductivity)
    4. Generates 5-minute "News Briefing" update module
    5. Notifies user: "New discovery relevant to your course!"
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        config: Optional[WatchtowerConfig] = None
    ):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.config = config or WatchtowerConfig()
        self._update_cache: Dict[str, DomainUpdate] = {}
        self._last_poll: Dict[NewsSource, datetime] = {}

    def _generate_update_id(self, title: str, url: str) -> str:
        """Generate unique ID for an update"""
        content = f"{title}:{url}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    async def fetch_arxiv_updates(
        self,
        domains: List[str],
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent arXiv papers for given domains

        Note: In production, use arxiv API or feed
        """
        # Simulated arXiv fetch - replace with actual API call
        try:
            import httpx

            # Search arXiv for recent papers
            search_query = " OR ".join([f'all:"{d}"' for d in domains[:5]])

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "http://export.arxiv.org/api/query",
                    params={
                        "search_query": search_query,
                        "start": 0,
                        "max_results": max_results,
                        "sortBy": "submittedDate",
                        "sortOrder": "descending"
                    },
                    timeout=30.0
                )

                if response.status_code == 200:
                    # Parse XML response
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(response.text)

                    ns = {'atom': 'http://www.w3.org/2005/Atom'}
                    entries = root.findall('.//atom:entry', ns)

                    results = []
                    for entry in entries:
                        title = entry.find('atom:title', ns)
                        summary = entry.find('atom:summary', ns)
                        link = entry.find('atom:id', ns)
                        published = entry.find('atom:published', ns)

                        results.append({
                            "title": title.text.strip() if title is not None else "Unknown",
                            "summary": summary.text.strip()[:500] if summary is not None else "",
                            "url": link.text if link is not None else "",
                            "published": published.text if published is not None else "",
                            "source": "arxiv"
                        })

                    return results

        except ImportError:
            logger.warning("httpx not installed, returning mock arXiv results")
        except Exception as e:
            logger.error(f"Error fetching arXiv: {e}")

        # Return empty list on failure
        return []

    async def fetch_github_trending(
        self,
        languages: List[str] = None,
        since: str = "daily"
    ) -> List[Dict[str, Any]]:
        """
        Fetch GitHub trending repositories

        Note: In production, use GitHub API or scraping
        """
        # Simulated - replace with actual API
        return []

    async def fetch_hacker_news(
        self,
        min_score: int = 100,
        max_stories: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Fetch top Hacker News stories

        Uses HN API to get recent high-scoring stories
        """
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                # Get top story IDs
                response = await client.get(
                    "https://hacker-news.firebaseio.com/v0/topstories.json",
                    timeout=15.0
                )

                if response.status_code != 200:
                    return []

                story_ids = response.json()[:max_stories]

                # Fetch story details
                stories = []
                for story_id in story_ids:
                    try:
                        story_response = await client.get(
                            f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json",
                            timeout=10.0
                        )
                        if story_response.status_code == 200:
                            story = story_response.json()
                            if story.get("score", 0) >= min_score:
                                stories.append({
                                    "title": story.get("title", ""),
                                    "url": story.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
                                    "score": story.get("score", 0),
                                    "source": "hacker_news"
                                })
                    except Exception as e:
                        logger.debug(f"Error fetching HN story {story_id}: {e}")

                return stories

        except ImportError:
            logger.warning("httpx not installed")
        except Exception as e:
            logger.error(f"Error fetching HN: {e}")

        return []

    async def assess_relevance(
        self,
        update: Dict[str, Any],
        syllabus_concepts: List[str],
        course_topic: str
    ) -> Tuple[float, List[str]]:
        """
        Assess how relevant an update is to a syllabus

        Args:
            update: Raw update data
            syllabus_concepts: Concepts in the syllabus
            course_topic: Overall course topic

        Returns:
            (relevance_score, related_concepts)
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You assess relevance of news/papers to educational curricula.

Score relevance 0.0-1.0:
- 0.0: Completely unrelated
- 0.3: Tangentially related
- 0.5: Moderately relevant, good context
- 0.7: Highly relevant, should be incorporated
- 1.0: Critical update, directly impacts course material

Output JSON:
{{
    "relevance_score": 0.0-1.0,
    "related_concepts": ["list of syllabus concepts this relates to"],
    "reason": "brief explanation",
    "priority": "critical/high/medium/low"
}}"""),
            ("human", """Course Topic: {topic}
Syllabus Concepts: {concepts}

Update to assess:
Title: {title}
Summary: {summary}

Assess relevance:""")
        ])

        try:
            messages = prompt.format_messages(
                topic=course_topic,
                concepts=", ".join(syllabus_concepts[:30]),
                title=update.get("title", ""),
                summary=update.get("summary", update.get("title", ""))[:500]
            )

            response = await self.llm.ainvoke(messages)

            # Parse JSON
            response_text = response.content
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end]

            import json
            assessment = json.loads(response_text.strip())

            return (
                assessment.get("relevance_score", 0.0),
                assessment.get("related_concepts", [])
            )

        except Exception as e:
            logger.error(f"Error assessing relevance: {e}")
            return (0.0, [])

    async def generate_update_module(
        self,
        update: DomainUpdate,
        syllabus: Dict[str, Any]
    ) -> Optional[SyllabusUpdateModule]:
        """
        Generate an update micro-module for the syllabus

        Args:
            update: The detected domain update
            syllabus: Current syllabus structure

        Returns:
            SyllabusUpdateModule to graft into syllabus
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You create brief educational update modules for existing curricula.

Generate a micro-module (5-15 minutes) that:
1. Explains the new development in context of what the learner already knows
2. Connects to existing syllabus concepts
3. Highlights practical implications
4. Suggests further exploration if interested

Output JSON:
{{
    "title": "Module title",
    "description": "2-3 sentence description",
    "concepts_covered": ["new concepts introduced"],
    "estimated_minutes": 5-15,
    "content_type": "briefing|deep_dive|practical",
    "learning_objectives": ["1-2 brief LOs"],
    "suggested_position": "After Module X" or "Standalone"
}}"""),
            ("human", """Domain Update:
Title: {update_title}
Summary: {update_summary}
Related Concepts: {related_concepts}

Current Syllabus Topic: {syllabus_topic}
Existing Modules: {modules}

Generate update module:""")
        ])

        try:
            module_list = "\n".join([
                f"- Week {m.get('week', '?')}: {m.get('title', 'Unknown')}"
                for m in syllabus.get("modules", [])
            ])

            messages = prompt.format_messages(
                update_title=update.title,
                update_summary=update.summary,
                related_concepts=", ".join(update.related_concepts),
                syllabus_topic=syllabus.get("overall_arc", ""),
                modules=module_list
            )

            response = await self.llm.ainvoke(messages)

            # Parse JSON
            response_text = response.content
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end]

            import json
            module_data = json.loads(response_text.strip())

            import uuid
            return SyllabusUpdateModule(
                id=str(uuid.uuid4())[:8],
                title=module_data.get("title", f"Update: {update.title}"),
                description=module_data.get("description", ""),
                concepts_covered=module_data.get("concepts_covered", []),
                estimated_minutes=module_data.get("estimated_minutes", 10),
                parent_module_id=None,  # Could be extracted from suggested_position
                prerequisite_concepts=update.related_concepts,
                content_type=module_data.get("content_type", "briefing"),
                priority=update.priority,
                source_update=update
            )

        except Exception as e:
            logger.error(f"Error generating update module: {e}")
            return None

    async def poll_sources(
        self,
        syllabus: Dict[str, Any],
        course_topic: str
    ) -> List[DomainUpdate]:
        """
        Poll configured sources for updates relevant to a syllabus

        Args:
            syllabus: Current syllabus structure
            course_topic: Main course topic

        Returns:
            List of relevant domain updates
        """
        all_updates: List[DomainUpdate] = []

        # Extract concepts from syllabus
        syllabus_concepts = []
        for module in syllabus.get("modules", []):
            syllabus_concepts.extend(module.get("concepts", []))

        if not syllabus_concepts:
            logger.warning("No concepts in syllabus to match against")
            return []

        # Determine domains to search
        domains = self.config.domains or [course_topic]

        # Poll each enabled source
        for source in self.config.enabled_sources:
            try:
                raw_updates = []

                if source == NewsSource.ARXIV:
                    raw_updates = await self.fetch_arxiv_updates(domains)
                elif source == NewsSource.HACKER_NEWS:
                    raw_updates = await self.fetch_hacker_news()
                # Add other sources as needed

                # Assess relevance of each update
                for raw_update in raw_updates:
                    update_id = self._generate_update_id(
                        raw_update.get("title", ""),
                        raw_update.get("url", "")
                    )

                    # Skip if already processed
                    if update_id in self._update_cache:
                        continue

                    relevance, related_concepts = await self.assess_relevance(
                        raw_update,
                        syllabus_concepts,
                        course_topic
                    )

                    if relevance >= self.config.min_relevance_score:
                        update = DomainUpdate(
                            id=update_id,
                            source=source,
                            title=raw_update.get("title", "Unknown"),
                            summary=raw_update.get("summary", ""),
                            url=raw_update.get("url", ""),
                            priority=self._determine_priority(relevance),
                            related_concepts=related_concepts,
                            relevance_score=relevance,
                            detected_at=datetime.utcnow(),
                            raw_data=raw_update
                        )

                        all_updates.append(update)
                        self._update_cache[update_id] = update

                self._last_poll[source] = datetime.utcnow()

            except Exception as e:
                logger.error(f"Error polling {source.value}: {e}")

        # Sort by relevance and limit
        all_updates.sort(key=lambda u: u.relevance_score, reverse=True)
        return all_updates[:self.config.max_updates_per_day]

    def _determine_priority(self, relevance: float) -> UpdatePriority:
        """Determine priority based on relevance score"""
        if relevance >= 0.9:
            return UpdatePriority.CRITICAL
        elif relevance >= 0.75:
            return UpdatePriority.HIGH
        elif relevance >= 0.6:
            return UpdatePriority.MEDIUM
        else:
            return UpdatePriority.LOW

    async def run_once(
        self,
        syllabus: Dict[str, Any],
        course_topic: str
    ) -> List[SyllabusUpdateModule]:
        """
        Run a single poll cycle and generate update modules

        Args:
            syllabus: Current syllabus
            course_topic: Course topic

        Returns:
            List of generated update modules
        """
        logger.info(f"Watchtower polling for updates on: {course_topic}")

        # Poll sources
        updates = await self.poll_sources(syllabus, course_topic)
        logger.info(f"Found {len(updates)} relevant updates")

        # Generate modules for top updates
        modules = []
        for update in updates[:3]:  # Limit to top 3
            module = await self.generate_update_module(update, syllabus)
            if module:
                modules.append(module)
                logger.info(f"Generated update module: {module.title}")

        return modules


# Global instance (lazy initialized)
_watchtower_agent: Optional[WatchtowerAgent] = None


def get_watchtower_agent() -> WatchtowerAgent:
    """Get or create the watchtower agent singleton"""
    global _watchtower_agent
    if _watchtower_agent is None:
        _watchtower_agent = WatchtowerAgent()
    return _watchtower_agent
