"""Context tracker - Automatically updates CONTEXT_TRACE.md with project context."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class ContextTracker:
    """Manages the CONTEXT_TRACE.md file with automatic updates.

    Provides structured logging of:
    - Phase/task tracking
    - Decision rationale
    - Token usage metrics
    - Cross-references
    - Verification results
    """

    ENTRY_TEMPLATE = """
## {timestamp} - {phase}

### Task: {task}

**Decision Rationale:**
{rationale}

**Implementation Details:**
{details}

**Verification Results:**
{verification}

**Token Metrics:**
{metrics}

**Issues & Resolutions:**
{issues}

**Next Steps:**
{next_steps}

---
"""

    def __init__(self, trace_file: Path | str) -> None:
        """Initialize the context tracker.

        Args:
            trace_file: Path to the CONTEXT_TRACE.md file.
        """
        self._trace_file = Path(trace_file)
        self._entries: list[dict[str, Any]] = []
        self._cumulative_tokens = 0
        self._cumulative_cost = 0.0

    def update(
        self,
        phase: str,
        task: str,
        details: dict[str, Any] | None = None,
        rationale: str = "",
        verification: dict[str, Any] | None = None,
        issues: list[str] | None = None,
        next_steps: list[str] | None = None,
    ) -> None:
        """Add an entry to the context trace.

        Args:
            phase: Current phase name.
            task: Task description.
            details: Implementation details dict.
            rationale: Decision rationale text.
            verification: Verification results dict.
            issues: List of issues encountered.
            next_steps: List of next steps.
        """
        details = details or {}
        verification = verification or {}
        issues = issues or []
        next_steps = next_steps or []

        # Update cumulative metrics
        tokens_used = details.get("tokens_used", 0)
        cost = details.get("cost", 0.0)
        self._cumulative_tokens += tokens_used
        self._cumulative_cost += cost

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "task": task,
            "rationale": rationale,
            "details": details,
            "verification": verification,
            "issues": issues,
            "next_steps": next_steps,
            "cumulative_tokens": self._cumulative_tokens,
            "cumulative_cost": self._cumulative_cost,
        }

        self._entries.append(entry)
        self._write_entry(entry)

    def _write_entry(self, entry: dict[str, Any]) -> None:
        """Write a single entry to the CONTEXT_TRACE.md file."""
        # Format details
        details = entry.get("details", {})
        details_str = "\n".join(
            f"- {k}: {v}" for k, v in details.items()
        ) if details else "- No specific details"

        # Format verification
        verification = entry.get("verification", {})
        verification_str = "\n".join(
            f"- {k}: {v}" for k, v in verification.items()
        ) if verification else "- Pending verification"

        # Format metrics
        metrics_str = (
            f"- Tokens used this task: {details.get('tokens_used', 0):,}\n"
            f"- Cumulative tokens: {entry['cumulative_tokens']:,}\n"
            f"- Compression ratio: {details.get('compression_ratio', 'N/A')}\n"
            f"- Cost this task: ${details.get('cost', 0):.4f}\n"
            f"- Cumulative cost: ${entry['cumulative_cost']:.4f}"
        )

        # Format issues
        issues = entry.get("issues", [])
        issues_str = "\n".join(f"- {i}" for i in issues) if issues else "- No issues"

        # Format next steps
        next_steps = entry.get("next_steps", [])
        next_str = "\n".join(f"- {s}" for s in next_steps) if next_steps else "- Continue to next task"

        # Format rationale
        rationale = entry.get("rationale", "")
        rationale_str = f"- {rationale}" if rationale else "- Standard implementation approach"

        # Build formatted entry
        formatted = self.ENTRY_TEMPLATE.format(
            timestamp=entry["timestamp"],
            phase=entry["phase"],
            task=entry["task"],
            rationale=rationale_str,
            details=details_str,
            verification=verification_str,
            metrics=metrics_str,
            issues=issues_str,
            next_steps=next_str,
        )

        # Append to file
        try:
            with open(self._trace_file, "a") as f:
                f.write(formatted)
        except IOError:
            # If file doesn't exist or can't be written, store in memory only
            pass

    def get_recent_entries(self, count: int = 5) -> list[dict[str, Any]]:
        """Get the most recent context entries.

        Args:
            count: Number of recent entries to return.

        Returns:
            List of recent entry dicts.
        """
        return self._entries[-count:]

    def get_phase_entries(self, phase: str) -> list[dict[str, Any]]:
        """Get all entries for a specific phase.

        Args:
            phase: Phase name to filter by.

        Returns:
            List of entries for the phase.
        """
        return [e for e in self._entries if e["phase"] == phase]

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the context trace."""
        return {
            "total_entries": len(self._entries),
            "cumulative_tokens": self._cumulative_tokens,
            "cumulative_cost": self._cumulative_cost,
            "phases_covered": list(set(e["phase"] for e in self._entries)),
            "total_issues": sum(len(e.get("issues", [])) for e in self._entries),
        }

    def get_compressed_context(self, max_entries: int = 10) -> str:
        """Get a compressed version of the context trace for agent consumption.

        Args:
            max_entries: Maximum number of recent entries to include.

        Returns:
            Compressed context string.
        """
        recent = self._entries[-max_entries:]
        if not recent:
            return "No prior context available."

        lines = ["## Recent Context Trace\n"]
        for entry in recent:
            lines.append(f"### {entry['phase']} - {entry['task']}")
            if entry.get("rationale"):
                lines.append(f"Rationale: {entry['rationale']}")
            if entry.get("details"):
                key_details = {k: v for k, v in entry["details"].items()
                             if k in ("tokens_used", "duration", "verification_score")}
                if key_details:
                    lines.append(f"Metrics: {json.dumps(key_details)}")
            lines.append("")

        return "\n".join(lines)
