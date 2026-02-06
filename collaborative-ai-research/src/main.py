"""Main entry point for the Collaborative AI Research Team system."""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

from src.agents.coordinator_agent import CoordinatorAgent
from src.agents.researcher_agent import ResearcherAgent
from src.agents.analyst_agent import AnalystAgent
from src.agents.synthesizer_agent import SynthesizerAgent
from src.context.compressor import ContextCompressor
from src.context.tracker import ContextTracker
from src.context.shared_memory import SharedMemory
from src.communication.message_bus import MessageBus
from src.verification.cross_checker import CrossChecker
from src.verification.validator import OutputValidator
from src.security.input_sanitizer import InputSanitizer
from src.security.output_filter import OutputFilter
from src.security.audit_logger import AuditLogger
from src.utils.token_counter import TokenCounter
from src.utils.cost_calculator import CostCalculator
from src.utils.performance_monitor import PerformanceMonitor

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class ResearchResult:
    """Result from a research task."""

    query: str
    summary: str
    detailed_findings: list[dict[str, Any]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    verification_score: float = 0.0
    token_usage: int = 0
    cost: float = 0.0
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ResearchTeam:
    """Main interface for the collaborative AI research team.

    Orchestrates multiple specialized AI agents to perform research tasks
    with compressed context sharing, cross-verification, and security controls.
    """

    def __init__(self, config_dir: Optional[str] = None) -> None:
        """Initialize the research team with all components.

        Args:
            config_dir: Path to configuration directory. Defaults to project config/.
        """
        self._config_dir = Path(config_dir) if config_dir else PROJECT_ROOT / "config"
        self._configs = self._load_configs()

        # Initialize core components
        self._token_counter = TokenCounter()
        self._cost_calculator = CostCalculator()
        self._performance_monitor = PerformanceMonitor()

        # Initialize security components
        self._audit_logger = AuditLogger(self._configs.get("security", {}))
        self._input_sanitizer = InputSanitizer(self._configs.get("security", {}))
        self._output_filter = OutputFilter(self._configs.get("security", {}))

        # Initialize context management
        self._compressor = ContextCompressor(self._configs.get("compression", {}))
        self._shared_memory = SharedMemory(self._compressor)
        self._context_tracker = ContextTracker(PROJECT_ROOT / "CONTEXT_TRACE.md")

        # Initialize communication
        self._message_bus = MessageBus()

        # Initialize verification
        self._cross_checker = CrossChecker()
        self._validator = OutputValidator()

        # Initialize agents
        agent_configs = self._configs.get("agents", {})
        self._coordinator = CoordinatorAgent(
            config=agent_configs.get("coordinator", {}),
            shared_memory=self._shared_memory,
            message_bus=self._message_bus,
            token_counter=self._token_counter,
            audit_logger=self._audit_logger,
        )
        self._researcher = ResearcherAgent(
            config=agent_configs.get("researcher", {}),
            shared_memory=self._shared_memory,
            message_bus=self._message_bus,
            token_counter=self._token_counter,
            audit_logger=self._audit_logger,
        )
        self._analyst = AnalystAgent(
            config=agent_configs.get("analyst", {}),
            shared_memory=self._shared_memory,
            message_bus=self._message_bus,
            token_counter=self._token_counter,
            audit_logger=self._audit_logger,
        )
        self._synthesizer = SynthesizerAgent(
            config=agent_configs.get("synthesizer", {}),
            shared_memory=self._shared_memory,
            message_bus=self._message_bus,
            token_counter=self._token_counter,
            audit_logger=self._audit_logger,
        )

        # Register agents with coordinator
        self._coordinator.register_agents({
            "researcher": self._researcher,
            "analyst": self._analyst,
            "synthesizer": self._synthesizer,
        })

        # Subscribe agents to message bus
        self._setup_message_bus()

    def _load_configs(self) -> dict[str, Any]:
        """Load all configuration files."""
        def resolve_env(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: resolve_env(v) for k, v in value.items()}
            if isinstance(value, list):
                return [resolve_env(v) for v in value]
            if isinstance(value, str):
                pattern = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")

                def repl(match: re.Match[str]) -> str:
                    var = match.group(1)
                    default = match.group(2) or ""
                    return os.getenv(var, default)

                return pattern.sub(repl, value)
            return value

        configs = {}
        for config_file in self._config_dir.glob("*.yaml"):
            with open(config_file, "r") as f:
                raw = yaml.safe_load(f)
                configs[config_file.stem] = resolve_env(raw)
        return configs

    def _setup_message_bus(self) -> None:
        """Set up message bus subscriptions for all agents."""
        self._message_bus.subscribe("coordinator", self._coordinator.handle_message)
        self._message_bus.subscribe("researcher", self._researcher.handle_message)
        self._message_bus.subscribe("analyst", self._analyst.handle_message)
        self._message_bus.subscribe("synthesizer", self._synthesizer.handle_message)
        self._message_bus.subscribe("broadcast", self._handle_broadcast)

    async def _handle_broadcast(self, message: dict[str, Any]) -> None:
        """Handle broadcast messages to all agents."""
        for agent in [self._coordinator, self._researcher, self._analyst, self._synthesizer]:
            await agent.handle_message(message)

    def research(self, query: str, **kwargs: Any) -> ResearchResult:
        """Execute a research task synchronously.

        Args:
            query: The research query to investigate.
            **kwargs: Additional parameters (workflow, depth, verify, etc.)

        Returns:
            ResearchResult with findings, metrics, and verification.
        """
        return asyncio.run(self.aresearch(query, **kwargs))

    async def aresearch(self, query: str, **kwargs: Any) -> ResearchResult:
        """Execute a research task asynchronously.

        Args:
            query: The research query to investigate.
            **kwargs: Additional parameters.

        Returns:
            ResearchResult with findings, metrics, and verification.
        """
        start_time = time.time()

        # Security: Sanitize input
        sanitized_query = self._input_sanitizer.sanitize(query)
        self._audit_logger.log_event("research_request", {
            "query_length": len(query),
            "sanitized": query != sanitized_query,
        })

        try:
            # Phase 1: Coordinator decomposes the task
            task_plan = await self._coordinator.decompose_task(sanitized_query, **kwargs)

            # Phase 2: Execute subtasks via assigned agents
            agent_results = await self._coordinator.execute_plan(task_plan)

            # Phase 3: Cross-verify results
            verification = await self._cross_checker.verify(
                agent_results,
                agents=[self._researcher, self._analyst],
            )

            # Phase 4: Synthesize final output
            synthesis = await self._synthesizer.synthesize(
                query=sanitized_query,
                findings=agent_results,
                verification=verification,
            )

            # Security: Filter output
            filtered_summary = self._output_filter.filter(synthesis.get("summary", ""))

            # Calculate metrics
            duration = time.time() - start_time
            total_tokens = self._token_counter.get_total()
            total_cost = self._cost_calculator.calculate(self._token_counter.get_breakdown())

            # Track context
            self._context_tracker.update(
                phase="Research Execution",
                task=f"Query: {sanitized_query[:100]}...",
                details={
                    "tokens_used": total_tokens,
                    "cost": total_cost,
                    "duration": duration,
                    "verification_score": verification.get("score", 0),
                },
            )

            return ResearchResult(
                query=query,
                summary=filtered_summary,
                detailed_findings=agent_results,
                sources=synthesis.get("sources", []),
                confidence=synthesis.get("confidence", 0),
                verification_score=verification.get("score", 0),
                token_usage=total_tokens,
                cost=total_cost,
                duration=duration,
                metadata={
                    "task_plan": task_plan,
                    "compression_ratio": self._compressor.get_compression_ratio(),
                    "cache_hits": self._performance_monitor.get_cache_hits(),
                },
            )

        except Exception as e:
            self._audit_logger.log_event("research_error", {
                "error": str(e),
                "query_length": len(query),
            })
            raise

    def get_metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        return {
            "token_usage": self._token_counter.get_breakdown(),
            "total_cost": self._cost_calculator.get_total_cost(),
            "performance": self._performance_monitor.get_metrics(),
            "compression_ratio": self._compressor.get_compression_ratio(),
        }

    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        self._token_counter.reset()
        self._cost_calculator.reset()
        self._performance_monitor.reset()


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Collaborative AI Research Team")
    parser.add_argument("query", help="Research query to investigate")
    parser.add_argument("--workflow", default="standard", help="Workflow template")
    parser.add_argument("--depth", default="standard", choices=["quick", "standard", "deep"])
    parser.add_argument("--verify", action="store_true", default=True, help="Enable cross-verification")
    parser.add_argument("--config", help="Path to config directory")

    args = parser.parse_args()

    team = ResearchTeam(config_dir=args.config)
    result = team.research(
        args.query,
        workflow=args.workflow,
        depth=args.depth,
        verify=args.verify,
    )

    print(f"\n{'='*60}")
    print(f"Research Query: {result.query}")
    print(f"{'='*60}")
    print(f"\n{result.summary}")
    print(f"\n{'='*60}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Verification Score: {result.verification_score:.2%}")
    print(f"Tokens Used: {result.token_usage:,}")
    print(f"Cost: ${result.cost:.4f}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
