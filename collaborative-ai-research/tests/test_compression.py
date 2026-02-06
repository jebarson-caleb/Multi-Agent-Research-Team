"""Tests for the compression system."""

from __future__ import annotations

import pytest
from src.context.compressor import (
    ContextCompressor,
    CompressionStrategy,
    CompressionResult,
    CompressionMetrics,
)
from src.context.shared_memory import SharedMemory, ContextEntry
from src.context.tracker import ContextTracker


class TestCompressionStrategy:
    """Tests for CompressionStrategy enum."""

    def test_strategy_values(self):
        assert CompressionStrategy.EXTRACTIVE.value == "extractive"
        assert CompressionStrategy.ABSTRACTIVE.value == "abstractive"
        assert CompressionStrategy.HIERARCHICAL.value == "hierarchical"
        assert CompressionStrategy.INCREMENTAL.value == "incremental"


class TestCompressionResult:
    """Tests for CompressionResult dataclass."""

    def test_compression_result_creation(self):
        result = CompressionResult(
            original_text="Hello world this is a test",
            compressed_text="Hello test",
            strategy=CompressionStrategy.EXTRACTIVE,
            original_tokens=6,
            compressed_tokens=2,
            compression_ratio=0.333,
            information_retention=0.8,
        )
        assert result.compression_ratio == 0.333
        assert result.information_retention == 0.8

    def test_token_reduction_property(self):
        result = CompressionResult(
            original_text="test",
            compressed_text="t",
            strategy=CompressionStrategy.EXTRACTIVE,
            original_tokens=100,
            compressed_tokens=40,
            compression_ratio=0.4,
            information_retention=0.9,
        )
        assert result.token_reduction == pytest.approx(0.6)


class TestCompressionMetrics:
    """Tests for CompressionMetrics dataclass."""

    def test_default_metrics(self):
        metrics = CompressionMetrics()
        assert metrics.total_compressions == 0
        assert metrics.avg_compression_ratio == 0.0
        assert metrics.total_tokens_saved == 0

    def test_metrics_properties(self):
        metrics = CompressionMetrics(
            total_compressions=5,
            total_original_tokens=1000,
            total_compressed_tokens=400,
        )
        assert metrics.avg_compression_ratio == pytest.approx(0.4)
        assert metrics.total_tokens_saved == 600


class TestContextCompressor:
    """Tests for the ContextCompressor."""

    @pytest.fixture
    def compressor(self):
        return ContextCompressor()

    @pytest.fixture
    def long_text(self):
        """Generate a reasonably long text for compression testing."""
        paragraphs = [
            "Artificial intelligence has revolutionized the way we process and analyze data. "
            "Machine learning algorithms can identify patterns that humans might miss. "
            "Deep learning models have achieved remarkable accuracy in image recognition tasks.",
            "Natural language processing enables computers to understand human language. "
            "Transformer architectures have significantly improved text generation capabilities. "
            "Large language models can perform a wide variety of tasks with minimal fine-tuning.",
            "Computer vision systems are now capable of real-time object detection and tracking. "
            "Autonomous vehicles rely heavily on computer vision for navigation and safety. "
            "Medical imaging has been transformed by AI-powered diagnostic tools.",
            "Reinforcement learning allows agents to learn optimal strategies through trial and error. "
            "Game-playing AI systems have surpassed human-level performance in many domains. "
            "Robotics applications benefit greatly from reinforcement learning techniques.",
            "AI ethics and safety have become critical areas of research and policy discussion. "
            "Bias in AI systems can lead to unfair outcomes for certain groups of people. "
            "Transparency and explainability are essential for building trust in AI systems.",
        ]
        return "\n\n".join(paragraphs)

    @pytest.fixture
    def short_text(self):
        return "This is a short text."

    def test_extractive_compression(self, compressor, long_text):
        result = compressor.compress(
            long_text,
            strategy=CompressionStrategy.EXTRACTIVE,
            target_ratio=0.5,
        )
        assert isinstance(result, CompressionResult)
        assert len(result.compressed_text) < len(result.original_text)
        assert result.strategy == CompressionStrategy.EXTRACTIVE
        assert 0 < result.compression_ratio <= 1

    def test_abstractive_compression(self, compressor, long_text):
        result = compressor.compress(
            long_text,
            strategy=CompressionStrategy.ABSTRACTIVE,
            target_ratio=0.5,
        )
        assert isinstance(result, CompressionResult)
        assert len(result.compressed_text) < len(result.original_text)
        assert result.strategy == CompressionStrategy.ABSTRACTIVE

    def test_hierarchical_compression(self, compressor, long_text):
        result = compressor.compress(
            long_text,
            strategy=CompressionStrategy.HIERARCHICAL,
            target_ratio=0.5,
        )
        assert isinstance(result, CompressionResult)
        assert result.strategy == CompressionStrategy.HIERARCHICAL
        assert len(result.compressed_text) > 0

    def test_incremental_compression(self, compressor, long_text):
        result = compressor.compress(
            long_text,
            strategy=CompressionStrategy.INCREMENTAL,
            target_ratio=0.5,
        )
        assert isinstance(result, CompressionResult)
        assert result.strategy == CompressionStrategy.INCREMENTAL

    def test_compression_empty_text(self, compressor):
        result = compressor.compress(
            "",
            strategy=CompressionStrategy.EXTRACTIVE,
            target_ratio=0.5,
        )
        assert result.compressed_text == ""
        assert result.compression_ratio == 1.0

    def test_compression_short_text(self, compressor, short_text):
        result = compressor.compress(
            short_text,
            strategy=CompressionStrategy.EXTRACTIVE,
            target_ratio=0.5,
        )
        assert isinstance(result, CompressionResult)

    def test_compression_caching(self, compressor, long_text):
        """Test that identical inputs return cached results."""
        result1 = compressor.compress(
            long_text,
            strategy=CompressionStrategy.EXTRACTIVE,
            target_ratio=0.5,
        )
        result2 = compressor.compress(
            long_text,
            strategy=CompressionStrategy.EXTRACTIVE,
            target_ratio=0.5,
        )
        assert result1.compressed_text == result2.compressed_text

    def test_compression_metrics(self, compressor, long_text):
        compressor.compress(long_text, strategy=CompressionStrategy.EXTRACTIVE)
        compressor.compress(long_text, strategy=CompressionStrategy.ABSTRACTIVE)
        metrics = compressor.get_metrics()
        assert isinstance(metrics, (dict, CompressionMetrics))
        if isinstance(metrics, dict):
            assert metrics["total_compressions"] >= 2
        else:
            assert metrics.total_compressions >= 2

    def test_compression_different_ratios(self, compressor, long_text):
        result_high = compressor.compress(
            long_text,
            strategy=CompressionStrategy.EXTRACTIVE,
            target_ratio=0.3,
        )
        compressor._cache.clear()
        result_low = compressor.compress(
            long_text,
            strategy=CompressionStrategy.EXTRACTIVE,
            target_ratio=0.7,
        )
        assert isinstance(result_high, CompressionResult)
        assert isinstance(result_low, CompressionResult)
        assert len(result_high.compressed_text) > 0

    def test_sentence_splitting(self, compressor):
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sentences = compressor._split_sentences(text)
        assert len(sentences) >= 3

    def test_sentence_scoring(self, compressor):
        text = (
            "Artificial intelligence is important. "
            "The weather is nice. "
            "Machine learning enables AI."
        )
        sentences = compressor._split_sentences(text)
        scores = compressor._score_sentences(sentences, text)
        assert len(scores) == len(sentences)

    def test_token_counting(self, compressor):
        text = "Hello world this is a test"
        tokens = compressor._count_tokens(text)
        assert tokens > 0


class TestSharedMemory:
    """Tests for the SharedMemory system."""

    @pytest.fixture
    def memory(self):
        compressor = ContextCompressor()
        return SharedMemory(compressor)

    @pytest.mark.asyncio
    async def test_store_and_get(self, memory):
        await memory.store(
            agent="researcher",
            namespace="research",
            content="Test content",
        )
        entries = await memory.get("research")
        assert len(entries) == 1
        assert entries[0].content == "Test content"

    @pytest.mark.asyncio
    async def test_store_with_namespace(self, memory):
        await memory.store(
            agent="researcher",
            namespace="research",
            content="Content A",
        )
        await memory.store(
            agent="analyst",
            namespace="analysis",
            content="Content B",
        )
        research_entries = await memory.get("research")
        assert len(research_entries) == 1
        assert research_entries[0].content == "Content A"

    @pytest.mark.asyncio
    async def test_store_with_priority(self, memory):
        await memory.store(
            agent="test",
            namespace="ns",
            content="Low priority",
            priority="low",
        )
        await memory.store(
            agent="test",
            namespace="ns",
            content="High priority",
            priority="high",
        )
        entries = await memory.get("ns", priority_min="high")
        assert len(entries) == 1
        assert entries[0].content == "High priority"

    @pytest.mark.asyncio
    async def test_versioning(self, memory):
        await memory.store(
            agent="test",
            namespace="ns",
            content="Version 1",
        )
        await memory.store(
            agent="test",
            namespace="ns",
            content="Version 2",
        )
        entries = await memory.get("ns")
        assert len(entries) == 2
        latest = [e for e in entries if e.content == "Version 2"]
        assert len(latest) >= 1

    @pytest.mark.asyncio
    async def test_get_compressed(self, memory):
        await memory.store(
            agent="test",
            namespace="ns",
            content="This is some test content that should be compressed for efficiency.",
        )
        await memory.store(
            agent="test",
            namespace="ns",
            content="Another piece of content for testing compression.",
        )
        compressed = await memory.get_compressed("ns")
        assert isinstance(compressed, str)
        assert len(compressed) > 0


class TestContextTracker:
    """Tests for the ContextTracker."""

    @pytest.fixture
    def tracker(self, tmp_path):
        return ContextTracker(str(tmp_path / "CONTEXT_TRACE.md"))

    def test_update(self, tracker):
        tracker.update(
            phase="Phase 1",
            task="Test task",
            details={"tokens_used": 100, "compression_ratio": 0.5},
            rationale="Testing the tracker",
        )
        entries = tracker.get_recent_entries(1)
        assert len(entries) == 1
        assert entries[0]["phase"] == "Phase 1"
        assert entries[0]["task"] == "Test task"

    def test_multiple_entries(self, tracker):
        for i in range(5):
            tracker.update(
                phase=f"Phase {i}",
                task=f"Task {i}",
                details={"tokens_used": 100 * (i + 1)},
            )
        entries = tracker.get_recent_entries(3)
        assert len(entries) == 3

    def test_get_phase_entries(self, tracker):
        tracker.update(phase="Phase 1", task="T1")
        tracker.update(phase="Phase 2", task="T2")
        tracker.update(phase="Phase 1", task="T3")
        phase1 = tracker.get_phase_entries("Phase 1")
        assert len(phase1) == 2

    def test_get_summary(self, tracker):
        tracker.update(
            phase="Phase 1",
            task="T1",
            details={"tokens_used": 100},
        )
        summary = tracker.get_summary()
        assert "total_entries" in summary
        assert summary["total_entries"] == 1

    def test_get_compressed_context(self, tracker):
        tracker.update(phase="Phase 1", task="T1")
        tracker.update(phase="Phase 2", task="T2")
        context = tracker.get_compressed_context(max_entries=2)
        assert isinstance(context, str)
        assert "Phase 1" in context or "Phase 2" in context
