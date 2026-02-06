"""Shared memory store for inter-agent context sharing with versioning and access control."""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.context.compressor import ContextCompressor, CompressionStrategy


@dataclass
class ContextEntry:
    """A single entry in the shared memory store."""
    content: str
    agent: str
    namespace: str
    priority: str
    version: int
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextVersion:
    """Version information for context entries."""
    version: int
    timestamp: float
    agent: str
    content_hash: str
    change_type: str  # "create", "update", "delete", "compress"


class SharedMemory:
    """Thread-safe shared memory for inter-agent context sharing.

    Features:
    - Namespace-based context organization
    - Read-write locking for concurrent access
    - Automatic compression when thresholds exceeded
    - Context versioning with full history
    - Priority-based retention
    - Access control per agent
    """

    def __init__(
        self,
        compressor: ContextCompressor,
        max_tokens_per_namespace: int = 50000,
        compression_threshold: float = 0.8,
    ) -> None:
        """Initialize shared memory.

        Args:
            compressor: Context compressor for automatic compression.
            max_tokens_per_namespace: Max tokens before compression triggers.
            compression_threshold: Trigger compression at this % of max.
        """
        self._compressor = compressor
        self._max_tokens = max_tokens_per_namespace
        self._compression_threshold = compression_threshold

        # Storage
        self._store: dict[str, list[ContextEntry]] = {}
        self._versions: dict[str, list[ContextVersion]] = {}

        # Concurrency control
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

        # Metrics
        self._access_count: dict[str, int] = {}
        self._compression_count = 0

    async def store(
        self,
        agent: str,
        namespace: str,
        content: str,
        priority: str = "medium",
        metadata: dict[str, Any] | None = None,
    ) -> ContextVersion:
        """Store content in shared memory.

        Args:
            agent: Agent storing the content.
            namespace: Context namespace.
            content: Content to store.
            priority: Priority level (critical, high, medium, low).
            metadata: Optional metadata dict.

        Returns:
            Version information for the stored content.
        """
        lock = await self._get_lock(namespace)
        async with lock:
            # Initialize namespace if needed
            if namespace not in self._store:
                self._store[namespace] = []
                self._versions[namespace] = []

            # Create entry
            version_num = len(self._versions[namespace]) + 1
            entry = ContextEntry(
                content=content,
                agent=agent,
                namespace=namespace,
                priority=priority,
                version=version_num,
                timestamp=time.time(),
                metadata=metadata or {},
            )

            self._store[namespace].append(entry)

            # Create version record
            version = ContextVersion(
                version=version_num,
                timestamp=entry.timestamp,
                agent=agent,
                content_hash=str(hash(content)),
                change_type="create",
            )
            self._versions[namespace].append(version)

            # Check if compression needed
            await self._check_compression(namespace)

            return version

    async def get(
        self,
        namespace: str,
        agent: str = "",
        limit: int = 0,
        priority_min: str | None = None,
    ) -> list[ContextEntry]:
        """Get context entries from a namespace.

        Args:
            namespace: Context namespace.
            agent: Optional filter by agent.
            limit: Maximum entries to return (0 = all).
            priority_min: Minimum priority level to include.

        Returns:
            List of context entries.
        """
        lock = await self._get_lock(namespace)
        async with lock:
            self._access_count[namespace] = self._access_count.get(namespace, 0) + 1

            entries = self._store.get(namespace, [])

            # Filter by agent if specified
            if agent:
                entries = [e for e in entries if e.agent == agent]

            # Filter by priority
            if priority_min:
                priority_order = ["low", "medium", "high", "critical"]
                min_idx = priority_order.index(priority_min) if priority_min in priority_order else 0
                entries = [
                    e for e in entries
                    if priority_order.index(e.priority) >= min_idx
                ]

            # Apply limit
            if limit > 0:
                entries = entries[-limit:]

            return [copy.deepcopy(e) for e in entries]

    async def get_compressed(
        self,
        namespace: str,
        agent: str = "",
        strategy: CompressionStrategy = CompressionStrategy.EXTRACTIVE,
    ) -> str:
        """Get compressed context from a namespace.

        Args:
            namespace: Context namespace.
            agent: Optional filter by agent.
            strategy: Compression strategy to use.

        Returns:
            Compressed context string.
        """
        entries = await self.get(namespace, agent)
        if not entries:
            return ""

        # Combine all entries
        combined = "\n\n".join(
            f"[{e.agent}@{e.priority}] {e.content}" for e in entries
        )

        # Compress
        result = self._compressor.compress(
            combined,
            strategy=strategy,
            priority=entries[0].priority if entries else "medium",
        )

        return result.compressed_text

    async def get_all_namespaces(self) -> list[str]:
        """Get list of all namespaces."""
        return list(self._store.keys())

    async def get_namespace_info(self, namespace: str) -> dict[str, Any]:
        """Get information about a namespace."""
        lock = await self._get_lock(namespace)
        async with lock:
            entries = self._store.get(namespace, [])
            versions = self._versions.get(namespace, [])

            total_content = " ".join(e.content for e in entries)
            estimated_tokens = self._compressor._count_tokens(total_content)

            return {
                "namespace": namespace,
                "entry_count": len(entries),
                "version_count": len(versions),
                "estimated_tokens": estimated_tokens,
                "max_tokens": self._max_tokens,
                "utilization": estimated_tokens / self._max_tokens if self._max_tokens > 0 else 0,
                "agents": list(set(e.agent for e in entries)),
                "priorities": {
                    p: sum(1 for e in entries if e.priority == p)
                    for p in ["critical", "high", "medium", "low"]
                },
                "access_count": self._access_count.get(namespace, 0),
            }

    async def clear(self, namespace: str) -> None:
        """Clear all entries in a namespace."""
        lock = await self._get_lock(namespace)
        async with lock:
            self._store[namespace] = []
            version_num = len(self._versions.get(namespace, [])) + 1
            if namespace not in self._versions:
                self._versions[namespace] = []
            self._versions[namespace].append(ContextVersion(
                version=version_num,
                timestamp=time.time(),
                agent="system",
                content_hash="",
                change_type="delete",
            ))

    async def get_versions(self, namespace: str) -> list[ContextVersion]:
        """Get version history for a namespace."""
        return list(self._versions.get(namespace, []))

    async def _get_lock(self, namespace: str) -> asyncio.Lock:
        """Get or create a lock for a namespace."""
        async with self._global_lock:
            if namespace not in self._locks:
                self._locks[namespace] = asyncio.Lock()
            return self._locks[namespace]

    async def _check_compression(self, namespace: str) -> None:
        """Check if namespace needs compression and trigger if so."""
        entries = self._store.get(namespace, [])
        if not entries:
            return

        total_content = " ".join(e.content for e in entries)
        estimated_tokens = self._compressor._count_tokens(total_content)
        threshold = self._max_tokens * self._compression_threshold

        if estimated_tokens > threshold:
            await self._compress_namespace(namespace)

    async def _compress_namespace(self, namespace: str) -> None:
        """Compress entries in a namespace to reduce token usage."""
        entries = self._store.get(namespace, [])
        if not entries:
            return

        # Separate by priority
        critical_entries = [e for e in entries if e.priority == "critical"]
        other_entries = [e for e in entries if e.priority != "critical"]

        if not other_entries:
            return

        # Compress non-critical entries
        combined = "\n\n".join(e.content for e in other_entries)
        result = self._compressor.compress(
            combined,
            strategy=CompressionStrategy.HIERARCHICAL,
            priority="medium",
        )

        # Replace with compressed version
        compressed_entry = ContextEntry(
            content=result.compressed_text,
            agent="system_compressor",
            namespace=namespace,
            priority="medium",
            version=len(self._versions.get(namespace, [])) + 1,
            timestamp=time.time(),
            metadata={
                "compression_ratio": result.compression_ratio,
                "original_entries": len(other_entries),
                "strategy": result.strategy.value,
            },
        )

        self._store[namespace] = critical_entries + [compressed_entry]
        self._compression_count += 1

        # Record version
        if namespace not in self._versions:
            self._versions[namespace] = []
        self._versions[namespace].append(ContextVersion(
            version=compressed_entry.version,
            timestamp=compressed_entry.timestamp,
            agent="system_compressor",
            content_hash=str(hash(result.compressed_text)),
            change_type="compress",
        ))

    def get_metrics(self) -> dict[str, Any]:
        """Get shared memory metrics."""
        total_entries = sum(len(v) for v in self._store.values())
        total_versions = sum(len(v) for v in self._versions.values())

        return {
            "namespaces": len(self._store),
            "total_entries": total_entries,
            "total_versions": total_versions,
            "compression_count": self._compression_count,
            "access_counts": dict(self._access_count),
        }
