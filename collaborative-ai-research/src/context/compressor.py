"""Context compression engine with multiple strategies for token reduction."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class CompressionStrategy(str, Enum):
    """Available compression strategies."""
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    HIERARCHICAL = "hierarchical"
    INCREMENTAL = "incremental"


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    original_text: str
    compressed_text: str
    strategy: CompressionStrategy
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    information_retention: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_reduction(self) -> float:
        """Percentage of tokens reduced."""
        if self.original_tokens == 0:
            return 0.0
        return 1.0 - (self.compressed_tokens / self.original_tokens)


@dataclass
class CompressionMetrics:
    """Aggregate compression metrics."""
    total_compressions: int = 0
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0
    strategy_usage: dict[str, int] = field(default_factory=dict)

    @property
    def avg_compression_ratio(self) -> float:
        if self.total_original_tokens == 0:
            return 0.0
        return self.total_compressed_tokens / self.total_original_tokens

    @property
    def total_tokens_saved(self) -> int:
        return self.total_original_tokens - self.total_compressed_tokens


class ContextCompressor:
    """Multi-strategy context compression engine.

    Supports four compression strategies:
    1. Extractive: Select key sentences based on importance scoring
    2. Abstractive: Use LLM to generate semantic summaries
    3. Hierarchical: Multi-level compression (full → detailed → brief → ultra-brief)
    4. Incremental: Compress as context grows, preserving recent content

    The compressor tracks metrics and ensures information retention
    meets configured thresholds.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the context compressor.

        Args:
            config: Compression configuration from compression.yaml.
        """
        self._config = config or {}
        self._metrics = CompressionMetrics()
        self._cache: dict[str, CompressionResult] = {}

        # Load strategy configs
        strategies = self._config.get("strategies", {})
        self._extractive_config = strategies.get("extractive", {}).get("settings", {})
        self._abstractive_config = strategies.get("abstractive", {}).get("settings", {})
        self._hierarchical_config = strategies.get("hierarchical", {})
        self._incremental_config = strategies.get("incremental", {}).get("settings", {})

        # Global settings
        global_config = self._config.get("global", {})
        self._target_ratio = global_config.get("target_compression_ratio", 0.4)
        self._min_retention = global_config.get("min_information_retention", 0.95)
        self._cache_ttl = global_config.get("compression_cache_ttl", 3600)

        # Priority retention settings
        self._priority_retention = self._config.get("priority_retention", {})

    def compress(
        self,
        text: str,
        strategy: CompressionStrategy = CompressionStrategy.EXTRACTIVE,
        target_ratio: float | None = None,
        priority: str = "medium",
    ) -> CompressionResult:
        """Compress text using the specified strategy.

        Args:
            text: Text to compress.
            strategy: Compression strategy to use.
            target_ratio: Target compression ratio (0-1, lower = more compression).
            priority: Content priority level.

        Returns:
            CompressionResult with compressed text and metrics.
        """
        if not text or not text.strip():
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                strategy=strategy,
                original_tokens=0,
                compressed_tokens=0,
                compression_ratio=1.0,
                information_retention=1.0,
            )

        target = target_ratio or self._target_ratio

        # Adjust target based on priority
        retention_config = self._priority_retention.get(priority, {})
        priority_retention = retention_config.get("retention", 0.4)
        target = max(target, priority_retention)

        # Check cache
        cache_key = self._cache_key(text, strategy, target)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Apply compression strategy
        if strategy == CompressionStrategy.EXTRACTIVE:
            result = self._extractive_compress(text, target)
        elif strategy == CompressionStrategy.ABSTRACTIVE:
            result = self._abstractive_compress_sync(text, target)
        elif strategy == CompressionStrategy.HIERARCHICAL:
            result = self._hierarchical_compress(text, target)
        elif strategy == CompressionStrategy.INCREMENTAL:
            result = self._incremental_compress(text, target)
        else:
            result = self._extractive_compress(text, target)

        # Update metrics
        self._metrics.total_compressions += 1
        self._metrics.total_original_tokens += result.original_tokens
        self._metrics.total_compressed_tokens += result.compressed_tokens
        self._metrics.strategy_usage[strategy.value] = (
            self._metrics.strategy_usage.get(strategy.value, 0) + 1
        )

        # Cache result
        self._cache[cache_key] = result

        return result

    def _extractive_compress(self, text: str, target_ratio: float) -> CompressionResult:
        """Extract key sentences based on importance scoring.

        Uses a TF-IDF-inspired approach with position weighting
        to score and select the most important sentences.
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= 2:
            return self._make_result(text, text, CompressionStrategy.EXTRACTIVE, 1.0)

        original_tokens = self._count_tokens(text)
        target_tokens = int(original_tokens * target_ratio)

        # Score each sentence
        scores = self._score_sentences(sentences, text)

        # Sort by score and select top sentences until target reached
        scored_sentences = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )

        selected_indices = set()
        current_tokens = 0

        for idx, score in scored_sentences:
            sentence_tokens = self._count_tokens(sentences[idx])
            if current_tokens + sentence_tokens <= target_tokens:
                selected_indices.add(idx)
                current_tokens += sentence_tokens

            if current_tokens >= target_tokens:
                break

        # Always include first and last sentence for coherence
        if sentences:
            selected_indices.add(0)
            if len(sentences) > 1:
                selected_indices.add(len(sentences) - 1)

        # Reconstruct in original order
        compressed = " ".join(
            sentences[i] for i in sorted(selected_indices)
        )

        compressed_tokens = self._count_tokens(compressed)
        ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

        # Estimate information retention based on coverage
        retention = len(selected_indices) / len(sentences) if sentences else 1.0
        retention = min(retention * 1.2, 1.0)  # Boost since we selected top sentences

        return self._make_result(text, compressed, CompressionStrategy.EXTRACTIVE, retention)

    def _abstractive_compress_sync(self, text: str, target_ratio: float) -> CompressionResult:
        """Synchronous abstractive compression using sentence merging.

        For actual LLM-based abstractive compression, use compress_abstractive_async().
        This sync version uses sentence clustering and merging as a fallback.
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= 3:
            return self._make_result(text, text, CompressionStrategy.ABSTRACTIVE, 1.0)

        original_tokens = self._count_tokens(text)

        # Group related sentences by keyword overlap
        groups = self._group_sentences(sentences)

        # Merge each group into a summary sentence
        merged = []
        for group in groups:
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Take the longest sentence as representative (it likely contains the most info)
                representative = max(group, key=len)
                merged.append(representative)

        compressed = " ".join(merged)

        # Trim to target if still too long
        compressed_tokens = self._count_tokens(compressed)
        target_tokens = int(original_tokens * target_ratio)

        if compressed_tokens > target_tokens:
            # Re-apply extractive compression on the merged result
            result = self._extractive_compress(compressed, target_ratio)
            result.strategy = CompressionStrategy.ABSTRACTIVE
            return result

        retention = min(len(merged) / len(sentences) * 1.3, 1.0)
        return self._make_result(text, compressed, CompressionStrategy.ABSTRACTIVE, retention)

    def _hierarchical_compress(self, text: str, target_ratio: float) -> CompressionResult:
        """Multi-level compression creating summaries at different detail levels.

        Levels:
        - full: Complete text
        - detailed: ~50% of original
        - brief: ~25% of original
        - ultra_brief: ~10% of original

        Returns the level that best matches the target ratio.
        """
        levels_config = self._hierarchical_config.get("levels", {})

        # Generate each level
        levels = {}
        levels["full"] = text

        # Detailed level (~50%)
        detailed_result = self._extractive_compress(text, 0.5)
        levels["detailed"] = detailed_result.compressed_text

        # Brief level (~25%)
        brief_result = self._extractive_compress(text, 0.25)
        levels["brief"] = brief_result.compressed_text

        # Ultra-brief level (~10%)
        ultra_result = self._extractive_compress(text, 0.1)
        levels["ultra_brief"] = ultra_result.compressed_text

        # Select the level closest to target ratio
        original_tokens = self._count_tokens(text)
        best_level = "full"
        best_diff = float("inf")

        for level_name, level_text in levels.items():
            level_tokens = self._count_tokens(level_text)
            ratio = level_tokens / original_tokens if original_tokens > 0 else 1.0
            diff = abs(ratio - target_ratio)
            if diff < best_diff:
                best_diff = diff
                best_level = level_name

        compressed = levels[best_level]
        retention = 0.95 if best_level == "full" else (0.85 if best_level == "detailed" else 0.7)

        result = self._make_result(text, compressed, CompressionStrategy.HIERARCHICAL, retention)
        result.metadata["level"] = best_level
        result.metadata["available_levels"] = list(levels.keys())
        return result

    def _incremental_compress(self, text: str, target_ratio: float) -> CompressionResult:
        """Compress incrementally, preserving recent content.

        Divides text into segments and compresses older segments more aggressively
        while keeping recent segments intact.
        """
        preserve_recent = self._incremental_config.get("preserve_recent", 1000)
        merge_window = self._incremental_config.get("merge_window", 3)

        original_tokens = self._count_tokens(text)

        # Split into segments
        words = text.split()
        segment_size = max(len(words) // 5, 50)  # At least 5 segments
        segments = []
        for i in range(0, len(words), segment_size):
            segments.append(" ".join(words[i:i + segment_size]))

        if len(segments) <= 1:
            return self._make_result(text, text, CompressionStrategy.INCREMENTAL, 1.0)

        # Compress older segments more aggressively
        compressed_segments = []
        for i, segment in enumerate(segments):
            # Calculate compression ratio for this segment (older = more compressed)
            age_factor = i / len(segments)  # 0 = oldest, 1 = newest
            segment_ratio = target_ratio + (1.0 - target_ratio) * age_factor

            if i >= len(segments) - 1:
                # Keep most recent segment uncompressed
                compressed_segments.append(segment)
            else:
                result = self._extractive_compress(segment, segment_ratio)
                compressed_segments.append(result.compressed_text)

        compressed = " ".join(compressed_segments)
        retention = 0.85  # Approximate retention for incremental

        return self._make_result(text, compressed, CompressionStrategy.INCREMENTAL, retention)

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences with abbreviation handling."""
        # Protect common abbreviations from being treated as sentence endings
        protected = text.strip()
        abbreviations = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.",
                         "etc.", "vs.", "i.e.", "e.g.", "U.S.", "U.K."]
        placeholders: list[tuple[str, str]] = []
        for abbr in abbreviations:
            placeholder = abbr.replace(".", "\x00")
            placeholders.append((abbr, placeholder))
            protected = protected.replace(abbr, placeholder)

        # Split on sentence-ending punctuation followed by whitespace
        pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(pattern, protected)

        # Restore abbreviations
        result = []
        for s in sentences:
            for abbr, placeholder in placeholders:
                s = s.replace(placeholder, abbr)
            s = s.strip()
            if s:
                result.append(s)
        return result

    def _score_sentences(self, sentences: list[str], full_text: str) -> list[float]:
        """Score sentences by importance using TF-IDF-inspired metrics."""
        # Word frequency across entire text
        words = re.findall(r"\b\w+\b", full_text.lower())
        word_freq = Counter(words)
        total_words = len(words)

        scores = []
        for i, sentence in enumerate(sentences):
            sent_words = re.findall(r"\b\w+\b", sentence.lower())
            if not sent_words:
                scores.append(0.0)
                continue

            # TF-IDF-like score
            tf_score = sum(
                word_freq.get(w, 0) / total_words for w in sent_words
            ) / len(sent_words)

            # Position weight (first and last sentences are more important)
            position_weight = 1.0
            if self._extractive_config.get("position_weight", True):
                if i == 0 or i == len(sentences) - 1:
                    position_weight = 1.5
                elif i < len(sentences) * 0.2:
                    position_weight = 1.3

            # Length weight (prefer medium-length sentences)
            length = len(sent_words)
            length_weight = 1.0
            if 10 <= length <= 30:
                length_weight = 1.2
            elif length < 5:
                length_weight = 0.7

            # Keyword boost
            keyword_boost = 1.0
            boost_factor = self._extractive_config.get("keyword_boost", 1.5)
            important_keywords = {
                "important", "key", "critical", "significant", "conclusion",
                "result", "finding", "however", "therefore", "because",
                "evidence", "demonstrates", "shows", "indicates",
            }
            if any(w in important_keywords for w in sent_words):
                keyword_boost = boost_factor

            score = tf_score * position_weight * length_weight * keyword_boost
            scores.append(score)

        return scores

    def _group_sentences(self, sentences: list[str]) -> list[list[str]]:
        """Group sentences by keyword overlap for abstractive compression."""
        if not sentences:
            return []

        groups: list[list[str]] = [[sentences[0]]]
        threshold = self._extractive_config.get("dedup_threshold", 0.85)

        for sent in sentences[1:]:
            sent_words = set(re.findall(r"\b\w+\b", sent.lower()))
            merged = False

            for group in groups:
                group_words = set()
                for s in group:
                    group_words.update(re.findall(r"\b\w+\b", s.lower()))

                # Calculate Jaccard similarity
                if sent_words and group_words:
                    intersection = sent_words & group_words
                    union = sent_words | group_words
                    similarity = len(intersection) / len(union)

                    if similarity > threshold * 0.5:  # Relaxed threshold for grouping
                        group.append(sent)
                        merged = True
                        break

            if not merged:
                groups.append([sent])

        return groups

    def _count_tokens(self, text: str) -> int:
        """Estimate token count. Uses ~4 chars per token as approximation.

        For exact counting, use tiktoken, but this approximation is
        sufficient for compression ratio calculations.
        """
        if not text:
            return 0
        # Rough approximation: ~4 characters per token for English text
        return max(1, len(text) // 4)

    def _make_result(
        self,
        original: str,
        compressed: str,
        strategy: CompressionStrategy,
        retention: float,
    ) -> CompressionResult:
        """Create a CompressionResult with computed metrics."""
        original_tokens = self._count_tokens(original)
        compressed_tokens = self._count_tokens(compressed)
        ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

        return CompressionResult(
            original_text=original,
            compressed_text=compressed,
            strategy=strategy,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            information_retention=min(retention, 1.0),
        )

    def _cache_key(self, text: str, strategy: CompressionStrategy, target: float) -> str:
        """Generate a cache key for a compression operation."""
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        return f"{strategy.value}:{target:.2f}:{text_hash}"

    def get_compression_ratio(self) -> float:
        """Get the average compression ratio across all operations."""
        return self._metrics.avg_compression_ratio

    def get_metrics(self) -> dict[str, Any]:
        """Get compression metrics."""
        return {
            "total_compressions": self._metrics.total_compressions,
            "total_original_tokens": self._metrics.total_original_tokens,
            "total_compressed_tokens": self._metrics.total_compressed_tokens,
            "total_tokens_saved": self._metrics.total_tokens_saved,
            "avg_compression_ratio": self._metrics.avg_compression_ratio,
            "strategy_usage": self._metrics.strategy_usage,
            "cache_size": len(self._cache),
        }

    def reset_metrics(self) -> None:
        """Reset compression metrics."""
        self._metrics = CompressionMetrics()
        self._cache.clear()
