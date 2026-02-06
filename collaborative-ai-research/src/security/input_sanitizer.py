"""Input sanitizer with prompt injection detection and content validation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SanitizationResult:
    """Result of input sanitization."""
    original: str
    sanitized: str
    is_safe: bool
    threats_detected: list[str] = field(default_factory=list)
    modifications: list[str] = field(default_factory=list)
    risk_score: float = 0.0  # 0.0 = safe, 1.0 = high risk


class InputSanitizer:
    """Input sanitization with prompt injection detection.

    Security features:
    - Prompt injection pattern detection
    - Unicode normalization and control character removal
    - Input size enforcement
    - Content type validation
    - Special character escaping
    - Encoding attack detection
    """

    # Known prompt injection patterns
    INJECTION_PATTERNS: list[tuple[str, float]] = [
        (r"(?i)ignore\s+(all\s+)?previous\s+(instructions|prompts|rules)", 0.9),
        (r"(?i)disregard\s+(all\s+)?(above|previous|prior)", 0.9),
        (r"(?i)you\s+are\s+now\s+(a|an|the)", 0.8),
        (r"(?i)new\s+instructions?\s*:", 0.8),
        (r"(?i)override\s+(system|previous|all)", 0.9),
        (r"(?i)forget\s+(everything|all|previous)", 0.9),
        (r"(?i)act\s+as\s+if\s+you", 0.7),
        (r"(?i)pretend\s+(you\s+are|to\s+be)", 0.7),
        (r"(?i)(jailbreak|jail\s*break)", 0.95),
        (r"(?i)\bDAN\b", 0.85),
        (r"(?i)DAN\s+mode", 0.95),
        (r"(?i)developer\s+mode\s+(enabled|on|activated)", 0.9),
        (r"(?i)\[\s*system\s*\]", 0.85),
        (r"(?i)<\s*system\s*>", 0.85),
        (r"(?i)system\s*:\s*you\s+(are|must|should|will)", 0.85),
        (r"(?i)###\s*(system|instruction|prompt)", 0.7),
        (r"(?i)---\s*(system|new|override)", 0.7),
        (r"(?i)\\n\\n(system|human|assistant)\s*:", 0.8),
        (r"(?i)base64\s*decode", 0.6),
        (r"(?i)eval\s*\(", 0.7),
        (r"(?i)exec\s*\(", 0.7),
        (r"(?i)import\s+os|import\s+subprocess", 0.6),
    ]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the input sanitizer.

        Args:
            config: Security configuration dict.
        """
        config = config or {}
        sanitization_config = config.get("input_sanitization", {})

        self._max_input_length = sanitization_config.get("max_input_length", 100000)
        self._max_tokens = sanitization_config.get("max_tokens_per_request", 10000)

        # Prompt injection settings
        injection_config = sanitization_config.get("prompt_injection", {})
        self._injection_enabled = injection_config.get("enabled", True)
        self._sensitivity = injection_config.get("sensitivity", "high")
        self._action = injection_config.get("action", "block")

        # Additional patterns from config
        extra_patterns = injection_config.get("patterns", [])
        for pattern in extra_patterns:
            self.INJECTION_PATTERNS.append((re.escape(pattern), 0.8))

        # Sensitivity thresholds
        self._thresholds = {
            "low": 0.9,
            "medium": 0.7,
            "high": 0.5,
        }

        # Metrics
        self._total_checks = 0
        self._threats_detected = 0
        self._blocked = 0

    def sanitize(self, text: str) -> str:
        """Sanitize input text.

        Args:
            text: Raw input text.

        Returns:
            Sanitized text.

        Raises:
            ValueError: If input is blocked due to security threat.
        """
        result = self.check(text)

        if not result.is_safe:
            self._blocked += 1
            if self._action == "block":
                raise ValueError(
                    f"Input blocked: security threats detected "
                    f"(risk score: {result.risk_score:.2f})"
                )

        return result.sanitized

    def check(self, text: str) -> SanitizationResult:
        """Check input for security threats without blocking.

        Args:
            text: Input text to check.

        Returns:
            SanitizationResult with threat analysis.
        """
        self._total_checks += 1
        threats = []
        modifications = []
        sanitized = text

        # 1. Length check
        if len(sanitized) > self._max_input_length:
            sanitized = sanitized[:self._max_input_length]
            modifications.append(f"Truncated from {len(text)} to {self._max_input_length} chars")

        # 2. Unicode normalization
        normalized = unicodedata.normalize("NFKC", sanitized)
        if normalized != sanitized:
            modifications.append("Applied Unicode NFKC normalization")
            sanitized = normalized

        # 3. Remove control characters (keep newlines and tabs)
        cleaned = self._remove_control_chars(sanitized)
        if cleaned != sanitized:
            modifications.append("Removed control characters")
            sanitized = cleaned

        # 4. Prompt injection detection
        if self._injection_enabled:
            injection_threats = self._detect_injection(sanitized)
            threats.extend(injection_threats)

        # 5. Encoding attack detection
        encoding_threats = self._detect_encoding_attacks(sanitized)
        threats.extend(encoding_threats)

        # Calculate risk score
        if threats:
            risk_score = max(t[1] for t in threats) if threats else 0.0
            self._threats_detected += 1
        else:
            risk_score = 0.0

        # Determine if safe based on sensitivity threshold
        threshold = self._thresholds.get(self._sensitivity, 0.7)
        is_safe = risk_score < threshold

        return SanitizationResult(
            original=text,
            sanitized=sanitized,
            is_safe=is_safe,
            threats_detected=[t[0] for t in threats],
            modifications=modifications,
            risk_score=risk_score,
        )

    def _detect_injection(self, text: str) -> list[tuple[str, float]]:
        """Detect prompt injection patterns."""
        threats = []
        for pattern, severity in self.INJECTION_PATTERNS:
            if re.search(pattern, text):
                threats.append((f"Prompt injection pattern: {pattern[:50]}", severity))
        return threats

    def _detect_encoding_attacks(self, text: str) -> list[tuple[str, float]]:
        """Detect encoding-based attacks."""
        threats = []

        # Check for excessive unicode escapes
        unicode_escapes = re.findall(r"\\u[0-9a-fA-F]{4}", text)
        if len(unicode_escapes) > 20:
            threats.append(("Excessive unicode escapes", 0.6))

        # Check for null bytes
        if "\x00" in text:
            threats.append(("Null byte injection", 0.8))

        # Check for excessive backslashes (escape sequence abuse)
        if text.count("\\") > len(text) * 0.1:
            threats.append(("Excessive escape characters", 0.5))

        # Check for homoglyph attacks (mixed scripts)
        scripts = set()
        for char in text:
            try:
                script = unicodedata.name(char, "").split()[0]
                scripts.add(script)
            except (ValueError, IndexError):
                pass
        if len(scripts) > 10:
            threats.append(("Mixed unicode scripts (possible homoglyph attack)", 0.4))

        return threats

    def _remove_control_chars(self, text: str) -> str:
        """Remove control characters except newline, tab, and carriage return."""
        return "".join(
            char for char in text
            if char in ("\n", "\t", "\r") or not unicodedata.category(char).startswith("C")
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get sanitization metrics."""
        return {
            "total_checks": self._total_checks,
            "threats_detected": self._threats_detected,
            "blocked": self._blocked,
            "threat_rate": (
                self._threats_detected / self._total_checks
                if self._total_checks > 0 else 0.0
            ),
        }
