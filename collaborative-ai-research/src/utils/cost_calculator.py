"""Cost calculator for API usage with per-model pricing."""

from __future__ import annotations

from typing import Any


class CostCalculator:
    """Calculate API costs based on token usage and model pricing.

    Supports:
    - Per-model pricing (input vs output tokens)
    - Real-time cost tracking
    - Cost prediction based on usage patterns
    - Budget alerts
    """

    # Pricing per 1M tokens (as of early 2026)
    MODEL_PRICING: dict[str, dict[str, float]] = {
        # Gemini models
        "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
        "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
        "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        # OpenAI models
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }

    # Default pricing if model not found
    DEFAULT_PRICING: dict[str, float] = {"input": 0.15, "output": 0.60}

    def __init__(self, budget_limit: float = 0.0) -> None:
        """Initialize the cost calculator.

        Args:
            budget_limit: Maximum budget in USD (0 = unlimited).
        """
        self._budget_limit = budget_limit
        self._total_cost = 0.0
        self._cost_by_model: dict[str, float] = {}
        self._cost_by_agent: dict[str, float] = {}
        self._cost_history: list[dict[str, Any]] = []

    def calculate(self, token_breakdown: dict[str, Any]) -> float:
        """Calculate total cost from a token breakdown.

        Args:
            token_breakdown: Token breakdown from TokenCounter.get_breakdown().

        Returns:
            Total cost in USD.
        """
        total_cost = 0.0

        # Calculate by model
        by_model = token_breakdown.get("by_model", {})
        for model_name, usage in by_model.items():
            pricing = self.MODEL_PRICING.get(model_name, self.DEFAULT_PRICING)
            input_cost = (usage["input"] / 1_000_000) * pricing["input"]
            output_cost = (usage["output"] / 1_000_000) * pricing["output"]
            model_cost = input_cost + output_cost
            total_cost += model_cost
            self._cost_by_model[model_name] = model_cost

        # If no model breakdown, calculate from totals
        if not by_model:
            total = token_breakdown.get("total", {})
            input_tokens = total.get("input", 0)
            output_tokens = total.get("output", 0)
            pricing = self.DEFAULT_PRICING
            total_cost = (
                (input_tokens / 1_000_000) * pricing["input"]
                + (output_tokens / 1_000_000) * pricing["output"]
            )

        self._total_cost = total_cost
        return total_cost

    def calculate_request_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "",
        agent: str = "",
    ) -> float:
        """Calculate cost for a single request.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model name.
            agent: Agent name.

        Returns:
            Cost in USD.
        """
        pricing = self.MODEL_PRICING.get(model, self.DEFAULT_PRICING)
        cost = (
            (input_tokens / 1_000_000) * pricing["input"]
            + (output_tokens / 1_000_000) * pricing["output"]
        )

        self._total_cost += cost

        if model:
            self._cost_by_model[model] = self._cost_by_model.get(model, 0) + cost
        if agent:
            self._cost_by_agent[agent] = self._cost_by_agent.get(agent, 0) + cost

        self._cost_history.append({
            "model": model,
            "agent": agent,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
        })

        return cost

    def predict_cost(
        self,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        model: str = "",
    ) -> float:
        """Predict cost for a future request.

        Args:
            estimated_input_tokens: Estimated input tokens.
            estimated_output_tokens: Estimated output tokens.
            model: Model name.

        Returns:
            Predicted cost in USD.
        """
        pricing = self.MODEL_PRICING.get(model, self.DEFAULT_PRICING)
        return (
            (estimated_input_tokens / 1_000_000) * pricing["input"]
            + (estimated_output_tokens / 1_000_000) * pricing["output"]
        )

    def check_budget(self) -> dict[str, Any]:
        """Check budget status.

        Returns:
            Budget status dict.
        """
        return {
            "total_cost": self._total_cost,
            "budget_limit": self._budget_limit,
            "remaining": (
                self._budget_limit - self._total_cost
                if self._budget_limit > 0 else float("inf")
            ),
            "budget_exceeded": (
                self._total_cost > self._budget_limit
                if self._budget_limit > 0 else False
            ),
            "utilization": (
                self._total_cost / self._budget_limit
                if self._budget_limit > 0 else 0.0
            ),
        }

    def get_total_cost(self) -> float:
        """Get total accumulated cost."""
        return self._total_cost

    def get_cost_breakdown(self) -> dict[str, Any]:
        """Get detailed cost breakdown."""
        return {
            "total_cost": self._total_cost,
            "by_model": dict(self._cost_by_model),
            "by_agent": dict(self._cost_by_agent),
            "request_count": len(self._cost_history),
            "avg_cost_per_request": (
                self._total_cost / len(self._cost_history)
                if self._cost_history else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset all cost tracking."""
        self._total_cost = 0.0
        self._cost_by_model.clear()
        self._cost_by_agent.clear()
        self._cost_history.clear()
