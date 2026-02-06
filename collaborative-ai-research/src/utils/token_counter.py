"""Token counter for tracking per-agent and per-model token usage."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    """Token usage record."""
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    agent: str = ""

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class TokenCounter:
    """Thread-safe token usage tracker with per-agent and per-model breakdowns.

    Tracks:
    - Total tokens (input + output) across all agents
    - Per-agent token usage
    - Per-model token usage
    - Usage over time for trend analysis
    """

    def __init__(self) -> None:
        """Initialize the token counter."""
        self._lock = threading.Lock()
        self._total_input = 0
        self._total_output = 0
        self._by_agent: dict[str, TokenUsage] = {}
        self._by_model: dict[str, TokenUsage] = {}
        self._history: list[dict[str, Any]] = []

    def add(
        self,
        agent: str,
        input_tokens: int,
        output_tokens: int,
        model: str = "",
    ) -> None:
        """Record token usage.

        Args:
            agent: Agent name.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model name.
        """
        with self._lock:
            self._total_input += input_tokens
            self._total_output += output_tokens

            # By agent
            if agent not in self._by_agent:
                self._by_agent[agent] = TokenUsage(agent=agent)
            self._by_agent[agent].input_tokens += input_tokens
            self._by_agent[agent].output_tokens += output_tokens
            self._by_agent[agent].model = model

            # By model
            if model:
                if model not in self._by_model:
                    self._by_model[model] = TokenUsage(model=model)
                self._by_model[model].input_tokens += input_tokens
                self._by_model[model].output_tokens += output_tokens

            # History
            self._history.append({
                "agent": agent,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            })

    def get_total(self) -> int:
        """Get total token count (input + output)."""
        with self._lock:
            return self._total_input + self._total_output

    def get_breakdown(self) -> dict[str, Any]:
        """Get detailed token usage breakdown."""
        with self._lock:
            return {
                "total": {
                    "input": self._total_input,
                    "output": self._total_output,
                    "total": self._total_input + self._total_output,
                },
                "by_agent": {
                    name: {
                        "input": usage.input_tokens,
                        "output": usage.output_tokens,
                        "total": usage.total,
                    }
                    for name, usage in self._by_agent.items()
                },
                "by_model": {
                    name: {
                        "input": usage.input_tokens,
                        "output": usage.output_tokens,
                        "total": usage.total,
                    }
                    for name, usage in self._by_model.items()
                },
                "request_count": len(self._history),
            }

    def get_agent_usage(self, agent: str) -> dict[str, int]:
        """Get token usage for a specific agent."""
        with self._lock:
            usage = self._by_agent.get(agent, TokenUsage())
            return {
                "input": usage.input_tokens,
                "output": usage.output_tokens,
                "total": usage.total,
            }

    def reset(self) -> None:
        """Reset all counters."""
        with self._lock:
            self._total_input = 0
            self._total_output = 0
            self._by_agent.clear()
            self._by_model.clear()
            self._history.clear()
