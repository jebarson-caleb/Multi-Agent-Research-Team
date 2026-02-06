"""Researcher Agent - Specializes in information gathering and fact extraction."""

from __future__ import annotations

import json
from typing import Any, Optional

from src.agents.base_agent import BaseAgent, AgentState


class ResearcherAgent(BaseAgent):
    """Agent specialized in information gathering, searching, and fact extraction.

    Capabilities:
    - Search for information on topics
    - Extract key facts and data points
    - Evaluate source reliability
    - Organize findings in structured format
    """

    RESEARCH_PROMPT_TEMPLATE = """You are conducting research on the following topic.

Topic: {topic}

Context from previous research:
{context}

Instructions:
{instructions}

Provide your findings in the following JSON format:
{{
    "findings": [
        {{
            "fact": "Key finding or fact",
            "confidence": 0.0-1.0,
            "source": "Source or reasoning",
            "category": "Category of finding"
        }}
    ],
    "summary": "Brief summary of all findings",
    "gaps": ["List of information gaps identified"],
    "suggested_follow_up": ["Suggested follow-up research topics"]
}}"""

    async def execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute a research task.

        Args:
            task: Task specification with 'topic', 'instructions', etc.

        Returns:
            Research results with findings, summary, and metadata.
        """
        self._state = AgentState.PROCESSING
        topic = task.get("topic", task.get("query", ""))
        instructions = task.get("instructions", "Gather comprehensive information on this topic.")
        depth = task.get("depth", "standard")

        try:
            # Get compressed context from shared memory
            context = await self.get_compressed_context("research")

            # Build research prompt
            prompt = self.RESEARCH_PROMPT_TEMPLATE.format(
                topic=topic,
                context=context if context else "No prior context available.",
                instructions=self._build_instructions(instructions, depth),
            )

            # Call LLM
            response = await self.call_llm(
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            result = self._parse_research_response(response["content"])

            # Store findings in shared memory
            await self.store_context(
                content=json.dumps(result, indent=2),
                namespace="research",
                priority="high",
            )

            self._state = AgentState.COMPLETED
            return {
                "agent": self._name,
                "task_type": "research",
                "topic": topic,
                "result": result,
                "tokens_used": response["tokens_input"] + response["tokens_output"],
                "latency": response["latency"],
            }

        except Exception as e:
            self._state = AgentState.ERROR
            return {
                "agent": self._name,
                "task_type": "research",
                "topic": topic,
                "error": str(e),
                "result": None,
            }

    def _build_instructions(self, base_instructions: str, depth: str) -> str:
        """Build detailed instructions based on research depth."""
        depth_instructions = {
            "quick": (
                f"{base_instructions}\n"
                "Focus on the most important 3-5 facts. Be concise."
            ),
            "standard": (
                f"{base_instructions}\n"
                "Provide comprehensive findings with 5-10 key facts. "
                "Include confidence levels and identify gaps."
            ),
            "deep": (
                f"{base_instructions}\n"
                "Provide exhaustive findings with 10-20 key facts. "
                "Analyze each finding deeply. Identify all gaps and "
                "suggest follow-up research. Consider multiple perspectives."
            ),
        }
        return depth_instructions.get(depth, depth_instructions["standard"])

    def _parse_research_response(self, content: str) -> dict[str, Any]:
        """Parse LLM response into structured research findings."""
        try:
            # Try to extract JSON from the response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])
        except json.JSONDecodeError:
            pass

        # Fallback: structure the raw response
        return {
            "findings": [
                {
                    "fact": content[:500],
                    "confidence": 0.5,
                    "source": "LLM analysis",
                    "category": "general",
                }
            ],
            "summary": content[:200],
            "gaps": [],
            "suggested_follow_up": [],
        }

    async def search(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Search for information on a topic.

        Args:
            query: Search query.
            max_results: Maximum number of results.

        Returns:
            List of search results.
        """
        response = await self.call_llm(
            messages=[{
                "role": "user",
                "content": (
                    f"Research the following topic and provide {max_results} key findings:\n\n"
                    f"{query}\n\n"
                    "Format each finding as a JSON object with 'fact', 'confidence', and 'source' keys."
                ),
            }],
        )
        return self._parse_research_response(response["content"]).get("findings", [])

    async def extract_facts(self, text: str) -> list[dict[str, Any]]:
        """Extract key facts from a text.

        Args:
            text: Text to extract facts from.

        Returns:
            List of extracted facts.
        """
        response = await self.call_llm(
            messages=[{
                "role": "user",
                "content": (
                    "Extract all key facts from the following text. "
                    "For each fact, provide a confidence level (0-1) and category.\n\n"
                    f"Text:\n{text}\n\n"
                    "Format as JSON array of objects with 'fact', 'confidence', 'category' keys."
                ),
            }],
        )
        result = self._parse_research_response(response["content"])
        return result.get("findings", [])
