"""Audit logger with tamper-evident hash chain logging."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class AuditLogger:
    """Security audit logger with tamper-evident hash chain.

    Features:
    - Structured JSON logging
    - Hash chain integrity (each entry includes hash of previous)
    - Configurable log levels and events
    - Log rotation support
    - Secrets masking
    - Tamper detection
    """

    # Fields that should be masked in logs
    SENSITIVE_FIELDS = {
        "api_key", "apikey", "api-key",
        "password", "secret", "token",
        "access_token", "private_key",
        "authorization", "credential",
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the audit logger.

        Args:
            config: Security configuration dict.
        """
        config = config or {}
        audit_config = config.get("audit_logging", {})

        self._enabled = audit_config.get("enabled", True)
        self._log_level = audit_config.get("log_level", "INFO")

        # Hash chain
        hash_config = audit_config.get("hash_chain", {})
        self._hash_chain_enabled = hash_config.get("enabled", True)
        self._hash_algorithm = hash_config.get("algorithm", "sha256")
        self._last_hash = "genesis"

        # Events to log
        self._tracked_events = set(audit_config.get("events", [
            "agent_request", "agent_response", "security_violation",
            "prompt_injection_attempt", "pii_redaction",
            "rate_limit_exceeded", "authentication_failure",
            "context_access", "encryption_operation",
        ]))

        # Log entries (in-memory for testing/access)
        self._entries: list[dict[str, Any]] = []

        # File logging
        log_path = audit_config.get("log_path", "logs/audit.log")
        self._log_path = Path(log_path)
        self._setup_file_logger()

        # Metrics
        self._event_counts: dict[str, int] = {}

    def _setup_file_logger(self) -> None:
        """Set up file-based logging."""
        self._logger = logging.getLogger("audit")
        self._logger.setLevel(getattr(logging, self._log_level, logging.INFO))

        # Avoid duplicate handlers
        if not self._logger.handlers:
            try:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                handler = logging.FileHandler(str(self._log_path))
                handler.setFormatter(logging.Formatter("%(message)s"))
                self._logger.addHandler(handler)
            except (OSError, PermissionError):
                # Fallback to NullHandler if file logging fails
                self._logger.addHandler(logging.NullHandler())

    def log_event(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
        level: str = "INFO",
    ) -> dict[str, Any]:
        """Log a security audit event.

        Args:
            event_type: Type of event (e.g., 'agent_request', 'security_violation').
            data: Event data dict.
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

        Returns:
            The logged entry dict.
        """
        if not self._enabled:
            return {}

        data = data or {}

        # Mask sensitive fields
        masked_data = self._mask_sensitive(data)

        # Build log entry
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "level": level,
            "data": masked_data,
        }

        # Add hash chain
        if self._hash_chain_enabled:
            entry_str = json.dumps(entry, sort_keys=True, default=str)
            chain_input = f"{self._last_hash}:{entry_str}"
            entry_hash = hashlib.new(
                self._hash_algorithm,
                chain_input.encode(),
            ).hexdigest()
            entry["previous_hash"] = self._last_hash
            entry["entry_hash"] = entry_hash
            self._last_hash = entry_hash

        # Store in memory
        self._entries.append(entry)

        # Track event count
        self._event_counts[event_type] = self._event_counts.get(event_type, 0) + 1

        # Write to file
        try:
            log_level = getattr(logging, level, logging.INFO)
            self._logger.log(log_level, json.dumps(entry, default=str))
        except Exception:
            pass  # Don't fail on logging errors

        return entry

    def log_security_violation(
        self,
        violation_type: str,
        details: dict[str, Any],
    ) -> dict[str, Any]:
        """Log a security violation event.

        Args:
            violation_type: Type of violation.
            details: Violation details.

        Returns:
            The logged entry.
        """
        return self.log_event(
            event_type="security_violation",
            data={
                "violation_type": violation_type,
                **details,
            },
            level="WARNING",
        )

    def log_prompt_injection(
        self,
        input_text: str,
        risk_score: float,
        patterns_matched: list[str],
    ) -> dict[str, Any]:
        """Log a prompt injection attempt.

        Args:
            input_text: The suspicious input (truncated).
            risk_score: Calculated risk score.
            patterns_matched: Patterns that matched.

        Returns:
            The logged entry.
        """
        return self.log_event(
            event_type="prompt_injection_attempt",
            data={
                "input_preview": input_text[:200],  # Don't log full malicious input
                "risk_score": risk_score,
                "patterns_matched": patterns_matched,
                "action_taken": "blocked" if risk_score > 0.7 else "warned",
            },
            level="WARNING",
        )

    def verify_integrity(self) -> tuple[bool, list[int]]:
        """Verify the integrity of the audit log hash chain.

        Returns:
            Tuple of (is_valid, list of tampered entry indices).
        """
        if not self._hash_chain_enabled or len(self._entries) < 2:
            return True, []

        tampered = []
        prev_hash = "genesis"

        for i, entry in enumerate(self._entries):
            if "entry_hash" not in entry:
                continue

            # Reconstruct the expected hash
            check_entry = {k: v for k, v in entry.items()
                         if k not in ("previous_hash", "entry_hash")}
            entry_str = json.dumps(check_entry, sort_keys=True, default=str)
            chain_input = f"{prev_hash}:{entry_str}"
            expected_hash = hashlib.new(
                self._hash_algorithm,
                chain_input.encode(),
            ).hexdigest()

            if entry.get("entry_hash") != expected_hash:
                tampered.append(i)

            prev_hash = entry.get("entry_hash", prev_hash)

        return len(tampered) == 0, tampered

    def _mask_sensitive(self, data: dict[str, Any]) -> dict[str, Any]:
        """Mask sensitive fields in data dict."""
        masked = {}
        for key, value in data.items():
            if key.lower() in self.SENSITIVE_FIELDS:
                if isinstance(value, str) and len(value) > 4:
                    masked[key] = f"***{value[-4:]}"
                else:
                    masked[key] = "***"
            elif isinstance(value, dict):
                masked[key] = self._mask_sensitive(value)
            else:
                masked[key] = value
        return masked

    def get_entries(
        self,
        event_type: str | None = None,
        level: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get audit log entries with optional filtering.

        Args:
            event_type: Filter by event type.
            level: Filter by log level.
            limit: Maximum entries to return.

        Returns:
            List of log entry dicts.
        """
        entries = self._entries

        if event_type:
            entries = [e for e in entries if e.get("event_type") == event_type]
        if level:
            entries = [e for e in entries if e.get("level") == level]

        return entries[-limit:]

    def get_security_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get security-related events."""
        security_types = {
            "security_violation", "prompt_injection_attempt",
            "rate_limit_exceeded", "authentication_failure",
        }
        return [
            e for e in self._entries[-limit:]
            if e.get("event_type") in security_types
        ]

    def get_metrics(self) -> dict[str, Any]:
        """Get audit logging metrics."""
        return {
            "total_entries": len(self._entries),
            "event_counts": dict(self._event_counts),
            "security_events": sum(
                self._event_counts.get(t, 0)
                for t in ["security_violation", "prompt_injection_attempt",
                          "rate_limit_exceeded", "authentication_failure"]
            ),
            "hash_chain_enabled": self._hash_chain_enabled,
            "integrity_valid": self.verify_integrity()[0],
        }
