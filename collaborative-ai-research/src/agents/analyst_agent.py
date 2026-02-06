"""Analyst Agent - Specializes in data analysis and pattern recognition."""

from __future__ import annotations

import json
from typing import Any

from src.agents.base_agent import BaseAgent, AgentState


class AnalystAgent(BaseAgent):
    """Agent specialized in data analysis, pattern recognition, and insight generation.

    Capabilities:
    - Analyze research findings for patterns
    - Identify trends and correlations
    - Generate actionable insights
    - Assess confidence levels and limitations
    """

    ANALYSIS_PROMPT_TEMPLATE = """You are analyzing research findings to identify patterns and generate insights.

Research Findings:
{findings}

Additional Context:
{context}

Analysis Instructions:
{instructions}

Provide your analysis in the following JSON format:
{{
    "patterns": [
        {{
            "pattern": "Description of pattern identified",
            "evidence": ["List of supporting evidence"],
            "confidence": 0.0-1.0,
            "significance": "high/medium/low"
        }}
    ],
    "insights": [
        {{
            "insight": "Actionable insight derived from analysis",
            "basis": "What this insight is based on",
            "confidence": 0.0-1.0,
            "implications": ["Potential implications"]
        }}
    ],
    "trends": [
        {{
            "trend": "Identified trend",
            "direction": "increasing/decreasing/stable/emerging",
            "confidence": 0.0-1.0
        }}
    ],
    "limitations": ["List of analysis limitations"],
    "summary": "Executive summary of analysis"
}}"""

    async def execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute an analysis task.

        Args:
            task: Task specification with 'findings', 'instructions', etc.

        Returns:
            Analysis results with patterns, insights, and trends.
        """
        self._state = AgentState.PROCESSING
        findings = task.get("findings", task.get("data", ""))
        instructions = task.get("instructions", "Perform comprehensive analysis.")

        try:
            # Get compressed context
            context = await self.get_compressed_context("analysis")

            # Format findings for analysis
            if isinstance(findings, (list, dict)):
                findings_str = json.dumps(findings, indent=2)
            else:
                findings_str = str(findings)

            # Build analysis prompt
            prompt = self.ANALYSIS_PROMPT_TEMPLATE.format(
                findings=findings_str[:8000],  # Limit to prevent context overflow
                context=context if context else "No prior analysis context.",
                instructions=instructions,
            )

            # Call LLM
            response = await self.call_llm(
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            result = self._parse_analysis_response(response["content"])

            # Store analysis in shared memory
            await self.store_context(
                content=json.dumps(result, indent=2),
                namespace="analysis",
                priority="high",
            )

            self._state = AgentState.COMPLETED
            return {
                "agent": self._name,
                "task_type": "analysis",
                "result": result,
                "tokens_used": response["tokens_input"] + response["tokens_output"],
                "latency": response["latency"],
            }

        except Exception as e:
            self._state = AgentState.ERROR
            return {
                "agent": self._name,
                "task_type": "analysis",
                "error": str(e),
                "result": None,
            }

    def _parse_analysis_response(self, content: str) -> dict[str, Any]:
        """Parse LLM response into structured analysis results."""
        try:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])
        except json.JSONDecodeError:
            pass

        # Fallback structure
        return {
            "patterns": [],
            "insights": [
                {
                    "insight": content[:500],
                    "basis": "LLM analysis",
                    "confidence": 0.5,
                    "implications": [],
                }
            ],
            "trends": [],
            "limitations": ["Unable to parse structured response"],
            "summary": content[:200],
        }

    async def analyze(self, data: Any, focus: str = "general") -> dict[str, Any]:
        """Analyze data with a specific focus area.

        Args:
            data: Data to analyze (can be any format).
            focus: Analysis focus area.

        Returns:
            Analysis results.
        """
        return await self.execute_task({
            "findings": data,
            "instructions": f"Focus your analysis on: {focus}. Provide deep, quantified insights.",
        })

    async def find_patterns(self, data: Any) -> list[dict[str, Any]]:
        """Identify patterns in the provided data.

        Args:
            data: Data to analyze for patterns.

        Returns:
            List of identified patterns.
        """
        result = await self.analyze(data, focus="pattern identification")
        return result.get("result", {}).get("patterns", [])

    async def generate_insights(self, data: Any) -> list[dict[str, Any]]:
        """Generate actionable insights from data.

        Args:
            data: Data to generate insights from.

        Returns:
            List of insights.
        """
        result = await self.analyze(data, focus="actionable insights generation")
        return result.get("result", {}).get("insights", [])
