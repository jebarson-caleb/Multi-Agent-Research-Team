"""Output filter for PII redaction and sensitive information filtering."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FilterResult:
    """Result of output filtering."""
    original: str
    filtered: str
    redactions: list[dict[str, str]] = field(default_factory=list)
    sensitive_keywords_found: list[str] = field(default_factory=list)
    was_modified: bool = False


class OutputFilter:
    """Output filtering with PII redaction and sensitive info detection.

    Features:
    - PII detection and redaction (email, phone, SSN, credit card, IP)
    - Sensitive keyword detection
    - Output size limits
    - Error message sanitization
    - Custom pattern support
    """

    # Default PII patterns
    DEFAULT_PII_PATTERNS: dict[str, tuple[str, str]] = {
        "email": (
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
            "[EMAIL_REDACTED]",
        ),
        "phone": (
            r"(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}",
            "[PHONE_REDACTED]",
        ),
        "ssn": (
            r"\b\d{3}-\d{2}-\d{4}\b",
            "[SSN_REDACTED]",
        ),
        "credit_card": (
            r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
            "[CC_REDACTED]",
        ),
        "ip_address": (
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
            "[IP_REDACTED]",
        ),
        "api_key": (
            r"(?:sk|pk|api)[_-][a-zA-Z0-9]{20,}",
            "[API_KEY_REDACTED]",
        ),
    }

    # Default sensitive keywords
    DEFAULT_SENSITIVE_KEYWORDS: list[str] = [
        "password", "secret", "api_key", "api-key", "apikey",
        "access_token", "access-token", "private_key", "private-key",
        "auth_token", "bearer", "credential",
    ]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the output filter.

        Args:
            config: Security configuration dict.
        """
        config = config or {}
        filter_config = config.get("output_filtering", {})

        self._enabled = filter_config.get("enabled", True)
        self._max_output_length = filter_config.get("max_output_length", 50000)

        # PII patterns
        self._pii_patterns: dict[str, tuple[str, str]] = dict(self.DEFAULT_PII_PATTERNS)

        # Load custom PII patterns from config
        pii_config = filter_config.get("pii_redaction", {})
        if pii_config.get("enabled", True):
            for pii_type, settings in pii_config.get("types", {}).items():
                pattern = settings.get("pattern")
                replacement = settings.get("replacement", f"[{pii_type.upper()}_REDACTED]")
                if pattern:
                    self._pii_patterns[pii_type] = (pattern, replacement)

        # Sensitive keywords
        self._sensitive_keywords = list(self.DEFAULT_SENSITIVE_KEYWORDS)
        extra_keywords = filter_config.get("sensitive_keywords", [])
        self._sensitive_keywords.extend(extra_keywords)

        # Metrics
        self._total_filters = 0
        self._total_redactions = 0
        self._pii_by_type: dict[str, int] = {}

    def filter(self, text: str) -> str:
        """Filter output text, redacting PII and sensitive information.

        Args:
            text: Output text to filter.

        Returns:
            Filtered text with PII redacted.
        """
        if not self._enabled:
            return text

        result = self.check(text)
        return result.filtered

    def check(self, text: str) -> FilterResult:
        """Check output for PII and sensitive information.

        Args:
            text: Text to check.

        Returns:
            FilterResult with redaction details.
        """
        self._total_filters += 1
        redactions = []
        filtered = text

        # 1. Apply length limit
        if len(filtered) > self._max_output_length:
            filtered = filtered[:self._max_output_length]
            redactions.append({
                "type": "length_truncation",
                "detail": f"Truncated from {len(text)} to {self._max_output_length}",
            })

        # 2. PII redaction
        for pii_type, (pattern, replacement) in self._pii_patterns.items():
            matches = re.findall(pattern, filtered)
            if matches:
                filtered = re.sub(pattern, replacement, filtered)
                for match in matches:
                    redactions.append({
                        "type": f"pii_{pii_type}",
                        "replacement": replacement,
                    })
                self._pii_by_type[pii_type] = (
                    self._pii_by_type.get(pii_type, 0) + len(matches)
                )
                self._total_redactions += len(matches)

        # 3. Sensitive keyword detection
        sensitive_found = []
        for keyword in self._sensitive_keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", filtered, re.IGNORECASE):
                sensitive_found.append(keyword)

        # 4. Sanitize error messages (remove stack traces)
        filtered = self._sanitize_errors(filtered)

        was_modified = filtered != text

        return FilterResult(
            original=text,
            filtered=filtered,
            redactions=redactions,
            sensitive_keywords_found=sensitive_found,
            was_modified=was_modified,
        )

    def _sanitize_errors(self, text: str) -> str:
        """Remove stack traces and internal error details."""
        # Remove Python stack traces
        text = re.sub(
            r"Traceback \(most recent call last\):.*?(?=\n\n|\Z)",
            "[ERROR_DETAILS_REDACTED]",
            text,
            flags=re.DOTALL,
        )

        # Remove file paths that look internal
        text = re.sub(
            r"(?:File\s+\")?/(?:home|usr|var|opt|etc)/[^\s\"]+",
            "[PATH_REDACTED]",
            text,
        )

        return text

    def redact_pii(self, text: str) -> str:
        """Redact only PII from text (no other filtering).

        Args:
            text: Text to redact PII from.

        Returns:
            Text with PII redacted.
        """
        result = text
        for pii_type, (pattern, replacement) in self._pii_patterns.items():
            result = re.sub(pattern, replacement, result)
        return result

    def get_metrics(self) -> dict[str, Any]:
        """Get filter metrics."""
        return {
            "total_filters": self._total_filters,
            "total_redactions": self._total_redactions,
            "pii_by_type": dict(self._pii_by_type),
            "redaction_rate": (
                self._total_redactions / self._total_filters
                if self._total_filters > 0 else 0.0
            ),
        }
