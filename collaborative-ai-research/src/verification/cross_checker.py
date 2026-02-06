"""Cross-verification system for multi-agent consensus checking."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.base_agent import BaseAgent


class VerificationMode(str, Enum):
    """Verification modes."""
    PARALLEL = "parallel"      # Multiple agents verify independently
    SEQUENTIAL = "sequential"  # Chain of verification
    HIERARCHICAL = "hierarchical"  # Levels of scrutiny


@dataclass
class VerificationResult:
    """Result of a cross-verification."""
    score: float  # 0.0 - 1.0 consensus score
    verified: bool
    mode: VerificationMode
    verifier_results: list[dict[str, Any]] = field(default_factory=list)
    disputes: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class CrossChecker:
    """Cross-verification framework for multi-agent consensus checking.

    Supports three verification modes:
    1. Parallel: Multiple agents verify independently, then compare
    2. Sequential: Each agent builds on previous verification
    3. Hierarchical: Escalating levels of verification scrutiny

    Features:
    - Consensus scoring with configurable thresholds
    - Dispute detection and resolution
    - Fact-checking against sources
    - Consistency analysis
    """

    VERIFICATION_PROMPT = """You are verifying research findings for accuracy and consistency.

Original Findings:
{findings}

Your Task:
1. Check each finding for factual accuracy
2. Identify any inconsistencies or contradictions
3. Rate the overall reliability of the findings
4. Note any claims that need additional verification

Respond in JSON format:
{{
    "verified_findings": [
        {{
            "finding": "The finding being verified",
            "status": "verified|disputed|uncertain",
            "confidence": 0.0-1.0,
            "issues": ["Any issues found"]
        }}
    ],
    "overall_score": 0.0-1.0,
    "inconsistencies": ["List of inconsistencies found"],
    "recommendations": ["Recommendations for improvement"]
}}"""

    def __init__(
        self,
        consensus_threshold: float = 0.7,
        min_verifiers: int = 2,
        dispute_threshold: float = 0.3,
    ) -> None:
        """Initialize the cross-checker.

        Args:
            consensus_threshold: Minimum score for consensus (0-1).
            min_verifiers: Minimum number of verifying agents.
            dispute_threshold: Score difference that triggers dispute.
        """
        self._consensus_threshold = consensus_threshold
        self._min_verifiers = min_verifiers
        self._dispute_threshold = dispute_threshold
        self._verification_history: list[VerificationResult] = []

    async def verify(
        self,
        findings: list[dict[str, Any]],
        agents: list[BaseAgent] | None = None,
        mode: VerificationMode = VerificationMode.PARALLEL,
    ) -> dict[str, Any]:
        """Verify findings using specified agents and mode.

        Args:
            findings: List of findings to verify.
            agents: List of agents to use for verification.
            mode: Verification mode.

        Returns:
            Verification results dict.
        """
        if not agents:
            return self._no_verification_result(findings)

        if mode == VerificationMode.PARALLEL:
            result = await self._parallel_verify(findings, agents)
        elif mode == VerificationMode.SEQUENTIAL:
            result = await self._sequential_verify(findings, agents)
        elif mode == VerificationMode.HIERARCHICAL:
            result = await self._hierarchical_verify(findings, agents)
        else:
            result = await self._parallel_verify(findings, agents)

        self._verification_history.append(result)
        return self._result_to_dict(result)

    async def _parallel_verify(
        self,
        findings: list[dict[str, Any]],
        agents: list[BaseAgent],
    ) -> VerificationResult:
        """Parallel verification: each agent verifies independently."""
        findings_str = json.dumps(findings, indent=2)[:8000]

        # Each agent verifies independently
        verification_tasks = []
        for agent in agents:
            verification_tasks.append(
                self._verify_with_agent(agent, findings_str)
            )

        results = await asyncio.gather(*verification_tasks, return_exceptions=True)

        # Process results
        verifier_results = []
        scores = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                verifier_results.append({
                    "agent": agents[i].name,
                    "error": str(result),
                    "score": 0.0,
                })
            else:
                verifier_results.append(result)
                scores.append(result.get("overall_score", 0.0))

        # Calculate consensus
        avg_score = sum(scores) / len(scores) if scores else 0.0
        disputes = self._detect_disputes(verifier_results)

        return VerificationResult(
            score=avg_score,
            verified=avg_score >= self._consensus_threshold,
            mode=VerificationMode.PARALLEL,
            verifier_results=verifier_results,
            disputes=disputes,
            metadata={
                "num_verifiers": len(agents),
                "scores": scores,
                "consensus_threshold": self._consensus_threshold,
            },
        )

    async def _sequential_verify(
        self,
        findings: list[dict[str, Any]],
        agents: list[BaseAgent],
    ) -> VerificationResult:
        """Sequential verification: each agent builds on previous verification."""
        findings_str = json.dumps(findings, indent=2)[:8000]
        verifier_results = []
        accumulated_context = ""

        for agent in agents:
            enriched_findings = f"{findings_str}\n\nPrevious Verification Notes:\n{accumulated_context}"
            result = await self._verify_with_agent(agent, enriched_findings)
            verifier_results.append(result)

            # Build context for next verifier
            accumulated_context += (
                f"\n{agent.name}: Score={result.get('overall_score', 0)}, "
                f"Issues={result.get('inconsistencies', [])}"
            )

        # Final score is the last verifier's assessment (most informed)
        final_score = verifier_results[-1].get("overall_score", 0.0) if verifier_results else 0.0

        return VerificationResult(
            score=final_score,
            verified=final_score >= self._consensus_threshold,
            mode=VerificationMode.SEQUENTIAL,
            verifier_results=verifier_results,
            disputes=self._detect_disputes(verifier_results),
        )

    async def _hierarchical_verify(
        self,
        findings: list[dict[str, Any]],
        agents: list[BaseAgent],
    ) -> VerificationResult:
        """Hierarchical verification: increasing scrutiny levels."""
        findings_str = json.dumps(findings, indent=2)[:8000]
        verifier_results = []

        # Level 1: Quick check with first agent
        if agents:
            result = await self._verify_with_agent(agents[0], findings_str)
            verifier_results.append(result)
            score = result.get("overall_score", 0.0)

            # Level 2: If score is marginal, get second opinion
            if 0.5 <= score <= 0.9 and len(agents) > 1:
                result2 = await self._verify_with_agent(agents[1], findings_str)
                verifier_results.append(result2)
                score = (score + result2.get("overall_score", 0.0)) / 2

                # Level 3: If still uncertain, deep verification
                if 0.5 <= score <= 0.8 and len(agents) > 1:
                    deep_result = await self._deep_verify(
                        findings_str, agents[-1]
                    )
                    verifier_results.append(deep_result)
                    score = (score + deep_result.get("overall_score", 0.0)) / 2

        final_score = score if agents else 0.0

        return VerificationResult(
            score=final_score,
            verified=final_score >= self._consensus_threshold,
            mode=VerificationMode.HIERARCHICAL,
            verifier_results=verifier_results,
            disputes=self._detect_disputes(verifier_results),
            metadata={"verification_levels": len(verifier_results)},
        )

    async def _verify_with_agent(
        self,
        agent: BaseAgent,
        findings_str: str,
    ) -> dict[str, Any]:
        """Have a single agent verify findings."""
        prompt = self.VERIFICATION_PROMPT.format(findings=findings_str)

        try:
            response = await agent.call_llm(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )

            # Parse JSON response
            content = response["content"]
            try:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    result = json.loads(content[json_start:json_end])
                    result["agent"] = agent.name
                    result["tokens_used"] = (
                        response["tokens_input"] + response["tokens_output"]
                    )
                    return result
            except json.JSONDecodeError:
                pass

            return {
                "agent": agent.name,
                "overall_score": 0.5,
                "verified_findings": [],
                "inconsistencies": [],
                "recommendations": [content[:200]],
            }

        except Exception as e:
            return {
                "agent": agent.name,
                "error": str(e),
                "overall_score": 0.0,
            }

    async def _deep_verify(
        self,
        findings_str: str,
        agent: BaseAgent,
    ) -> dict[str, Any]:
        """Perform deep verification with additional scrutiny."""
        deep_prompt = (
            "Perform a deep, critical verification of these findings. "
            "Challenge every claim. Look for logical fallacies, unsupported "
            "claims, and potential biases.\n\n"
            f"{self.VERIFICATION_PROMPT.format(findings=findings_str)}"
        )

        try:
            response = await agent.call_llm(
                messages=[{"role": "user", "content": deep_prompt}],
                max_tokens=3000,
            )
            content = response["content"]
            try:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    result = json.loads(content[json_start:json_end])
                    result["agent"] = agent.name
                    result["verification_level"] = "deep"
                    return result
            except json.JSONDecodeError:
                pass
        except Exception:
            pass

        return {
            "agent": agent.name,
            "overall_score": 0.5,
            "verification_level": "deep",
        }

    def _detect_disputes(
        self,
        verifier_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Detect disputes between verifier results."""
        disputes = []
        scores = [r.get("overall_score", 0.0) for r in verifier_results if "error" not in r]

        if len(scores) < 2:
            return disputes

        # Check for significant score differences
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                diff = abs(scores[i] - scores[j])
                if diff > self._dispute_threshold:
                    disputes.append({
                        "verifier_a": verifier_results[i].get("agent", f"agent_{i}"),
                        "verifier_b": verifier_results[j].get("agent", f"agent_{j}"),
                        "score_a": scores[i],
                        "score_b": scores[j],
                        "difference": diff,
                        "resolution": "needs_review",
                    })

        return disputes

    def _no_verification_result(
        self,
        findings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return a default result when no verification agents are available."""
        return {
            "score": 0.0,
            "verified": False,
            "mode": "none",
            "message": "No verification agents available",
            "findings_count": len(findings),
        }

    def _result_to_dict(self, result: VerificationResult) -> dict[str, Any]:
        """Convert VerificationResult to dict."""
        return {
            "score": result.score,
            "verified": result.verified,
            "mode": result.mode.value,
            "verifier_results": result.verifier_results,
            "disputes": result.disputes,
            "metadata": result.metadata,
        }

    def get_history(self) -> list[dict[str, Any]]:
        """Get verification history."""
        return [self._result_to_dict(r) for r in self._verification_history]

    def get_metrics(self) -> dict[str, Any]:
        """Get verification metrics."""
        if not self._verification_history:
            return {"total_verifications": 0}

        scores = [r.score for r in self._verification_history]
        verified_count = sum(1 for r in self._verification_history if r.verified)
        dispute_count = sum(len(r.disputes) for r in self._verification_history)

        return {
            "total_verifications": len(self._verification_history),
            "avg_score": sum(scores) / len(scores),
            "verified_rate": verified_count / len(self._verification_history),
            "total_disputes": dispute_count,
            "mode_usage": {
                mode.value: sum(1 for r in self._verification_history if r.mode == mode)
                for mode in VerificationMode
            },
        }
