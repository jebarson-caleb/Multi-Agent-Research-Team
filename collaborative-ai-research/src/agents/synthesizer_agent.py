"""Synthesizer Agent - Specializes in information synthesis and report generation."""

from __future__ import annotations

import json
from typing import Any

from src.agents.base_agent import BaseAgent, AgentState


class SynthesizerAgent(BaseAgent):
    """Agent specialized in information synthesis, report generation, and summarization.

    Capabilities:
    - Combine research findings and analysis into coherent narratives
    - Generate comprehensive reports with executive summaries
    - Identify gaps and contradictions
    - Create clear, structured output documents
    """

    SYNTHESIS_PROMPT_TEMPLATE = """You are synthesizing research findings and analysis into a comprehensive report.

Original Query: {query}

Research Findings:
{findings}

Verification Results:
{verification}

Additional Context:
{context}

Instructions:
Create a comprehensive synthesis that:
1. Addresses the original query directly
2. Integrates all research findings coherently
3. Highlights key insights and patterns
4. Notes any gaps, contradictions, or limitations
5. Provides actionable conclusions

Respond in the following JSON format:
{{
    "summary": "Executive summary (2-3 paragraphs)",
    "key_findings": [
        {{
            "finding": "Key finding statement",
            "confidence": 0.0-1.0,
            "supporting_evidence": ["Evidence items"]
        }}
    ],
    "analysis_highlights": ["Key analysis points"],
    "gaps_and_limitations": ["Identified gaps"],
    "contradictions": ["Any contradictions found"],
    "conclusions": ["Main conclusions"],
    "recommendations": ["Actionable recommendations"],
    "sources": ["Source citations"],
    "confidence": 0.0-1.0,
    "metadata": {{
        "topic_coverage": "comprehensive/partial/limited",
        "analysis_depth": "deep/moderate/surface"
    }}
}}"""

    async def execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute a synthesis task.

        Args:
            task: Task specification with 'query', 'findings', etc.

        Returns:
            Synthesis results with report, conclusions, and metadata.
        """
        self._state = AgentState.PROCESSING
        query = task.get("query", "")
        findings = task.get("findings", [])
        verification = task.get("verification", {})

        try:
            result = await self.synthesize(
                query=query,
                findings=findings,
                verification=verification,
            )

            self._state = AgentState.COMPLETED
            return {
                "agent": self._name,
                "task_type": "synthesis",
                "result": result,
            }

        except Exception as e:
            self._state = AgentState.ERROR
            return {
                "agent": self._name,
                "task_type": "synthesis",
                "error": str(e),
                "result": None,
            }

    async def synthesize(
        self,
        query: str,
        findings: Any,
        verification: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Synthesize findings into a comprehensive report.

        Args:
            query: Original research query.
            findings: Research findings (list of agent results).
            verification: Cross-verification results.

        Returns:
            Synthesis with summary, conclusions, and recommendations.
        """
        # Get compressed context
        context = await self.get_compressed_context("synthesis")

        # Format inputs
        if isinstance(findings, (list, dict)):
            findings_str = json.dumps(findings, indent=2)
        else:
            findings_str = str(findings)

        verification_str = json.dumps(verification, indent=2) if verification else "Not available."

        # Build synthesis prompt
        prompt = self.SYNTHESIS_PROMPT_TEMPLATE.format(
            query=query,
            findings=findings_str[:10000],
            verification=verification_str[:2000],
            context=context if context else "No prior synthesis context.",
        )

        # Call LLM with higher token limit for comprehensive synthesis
        response = await self.call_llm(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_tokens,
        )

        # Parse response
        result = self._parse_synthesis_response(response["content"])

        # Store synthesis in shared memory
        await self.store_context(
            content=json.dumps({
                "query": query,
                "summary": result.get("summary", ""),
                "conclusions": result.get("conclusions", []),
            }, indent=2),
            namespace="synthesis",
            priority="critical",
        )

        return result

    def _parse_synthesis_response(self, content: str) -> dict[str, Any]:
        """Parse LLM response into structured synthesis results."""
        try:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])
        except json.JSONDecodeError:
            pass

        # Fallback structure
        return {
            "summary": content,
            "key_findings": [],
            "analysis_highlights": [],
            "gaps_and_limitations": [],
            "contradictions": [],
            "conclusions": [content[:200]],
            "recommendations": [],
            "sources": [],
            "confidence": 0.5,
            "metadata": {
                "topic_coverage": "partial",
                "analysis_depth": "surface",
            },
        }

    async def generate_report(
        self,
        query: str,
        findings: Any,
        format: str = "detailed",
    ) -> str:
        """Generate a formatted research report.

        Args:
            query: Research query.
            findings: Research findings.
            format: Report format (brief, detailed, executive).

        Returns:
            Formatted report string.
        """
        response = await self.call_llm(
            messages=[{
                "role": "user",
                "content": (
                    f"Generate a {format} research report for the following:\n\n"
                    f"Query: {query}\n\n"
                    f"Findings:\n{json.dumps(findings, indent=2)[:8000]}\n\n"
                    "Format the report with clear sections, headers, and bullet points."
                ),
            }],
        )
        return response["content"]

    async def summarize(self, content: str, max_length: int = 500) -> str:
        """Generate a concise summary of content.

        Args:
            content: Content to summarize.
            max_length: Maximum summary length in characters.

        Returns:
            Summary string.
        """
        response = await self.call_llm(
            messages=[{
                "role": "user",
                "content": (
                    f"Summarize the following in {max_length} characters or less. "
                    "Preserve all key facts and conclusions:\n\n"
                    f"{content[:5000]}"
                ),
            }],
            max_tokens=max_length // 3,
        )
        return response["content"]
