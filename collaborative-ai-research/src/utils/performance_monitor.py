"""Performance monitor for tracking latency, error rates, and throughput."""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests


class LRUCache:
    """Simple LRU cache with TTL for caching agent responses."""

    def __init__(self, max_size: int = 1000, ttl: int = 3600) -> None:
        """Initialize the cache.

        Args:
            max_size: Maximum number of cached items.
            ttl: Time to live in seconds.
        """
        self._max_size = max_size
        self._ttl = ttl
        self._cache: dict[str, tuple[Any, float]] = {}
        self._access_order: list[str] = []
        self._metrics = CacheMetrics()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """Get a value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/expired.
        """
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    self._metrics.hits += 1
                    # Move to end (most recently used)
                    if key in self._access_order:
                        self._access_order.remove(key)
                    self._access_order.append(key)
                    return value
                else:
                    # Expired
                    del self._cache[key]
                    if key in self._access_order:
                        self._access_order.remove(key)

            self._metrics.misses += 1
            return None

    def set(self, key: str, value: Any) -> None:
        """Set a value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self._max_size and self._access_order:
                oldest_key = self._access_order.pop(0)
                self._cache.pop(oldest_key, None)
                self._metrics.evictions += 1

            self._cache[key] = (value, time.time())
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    @property
    def metrics(self) -> CacheMetrics:
        return self._metrics


class PerformanceMonitor:
    """Monitor system performance metrics.

    Tracks:
    - Request latency (per agent, percentiles)
    - Error rates
    - Throughput (requests per second)
    - Cache performance
    - Resource utilization
    """

    def __init__(
        self,
        window_size: int = 1000,
        cache_max_size: int = 1000,
        cache_ttl: int = 3600,
    ) -> None:
        """Initialize the performance monitor.

        Args:
            window_size: Number of recent measurements to keep.
            cache_max_size: Maximum cache size.
            cache_ttl: Cache TTL in seconds.
        """
        self._lock = threading.Lock()
        self._window_size = window_size

        # Latency tracking
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._latencies_by_agent: dict[str, deque[float]] = {}

        # Error tracking
        self._errors: deque[dict[str, Any]] = deque(maxlen=window_size)
        self._total_requests = 0
        self._total_errors = 0

        # Throughput tracking
        self._request_timestamps: deque[float] = deque(maxlen=window_size)

        # Cache
        self._cache = LRUCache(max_size=cache_max_size, ttl=cache_ttl)

    def record_request(
        self,
        agent: str,
        latency: float,
        success: bool = True,
        error: str = "",
    ) -> None:
        """Record a request metric.

        Args:
            agent: Agent name.
            latency: Request latency in seconds.
            success: Whether the request succeeded.
            error: Error message if failed.
        """
        with self._lock:
            now = time.time()
            self._total_requests += 1
            self._latencies.append(latency)
            self._request_timestamps.append(now)

            # Per-agent latency
            if agent not in self._latencies_by_agent:
                self._latencies_by_agent[agent] = deque(maxlen=self._window_size)
            self._latencies_by_agent[agent].append(latency)

            # Error tracking
            if not success:
                self._total_errors += 1
                self._errors.append({
                    "agent": agent,
                    "error": error,
                    "timestamp": now,
                    "latency": latency,
                })

    def get_cache_hits(self) -> int:
        """Get total cache hits."""
        return self._cache.metrics.hits

    def get_cache(self) -> LRUCache:
        """Get the cache instance."""
        return self._cache

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        with self._lock:
            latencies = list(self._latencies)
            return {
                "latency": self._calculate_latency_stats(latencies),
                "latency_by_agent": {
                    agent: self._calculate_latency_stats(list(lats))
                    for agent, lats in self._latencies_by_agent.items()
                },
                "errors": {
                    "total": self._total_errors,
                    "rate": (
                        self._total_errors / self._total_requests
                        if self._total_requests > 0 else 0.0
                    ),
                    "recent": [
                        {"agent": e["agent"], "error": e["error"]}
                        for e in list(self._errors)[-5:]
                    ],
                },
                "throughput": self._calculate_throughput(),
                "cache": {
                    "hits": self._cache.metrics.hits,
                    "misses": self._cache.metrics.misses,
                    "hit_rate": self._cache.metrics.hit_rate,
                    "evictions": self._cache.metrics.evictions,
                },
                "total_requests": self._total_requests,
            }

    def _calculate_latency_stats(self, latencies: list[float]) -> dict[str, float]:
        """Calculate latency statistics."""
        if not latencies:
            return {"avg": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}

        sorted_lats = sorted(latencies)
        n = len(sorted_lats)

        return {
            "avg": sum(sorted_lats) / n,
            "min": sorted_lats[0],
            "max": sorted_lats[-1],
            "p50": sorted_lats[int(n * 0.5)],
            "p95": sorted_lats[min(int(n * 0.95), n - 1)],
            "p99": sorted_lats[min(int(n * 0.99), n - 1)],
        }

    def _calculate_throughput(self) -> dict[str, float]:
        """Calculate throughput metrics."""
        timestamps = list(self._request_timestamps)
        if len(timestamps) < 2:
            return {"rps": 0.0, "rpm": 0.0}

        duration = timestamps[-1] - timestamps[0]
        if duration == 0:
            return {"rps": 0.0, "rpm": 0.0}

        rps = len(timestamps) / duration
        return {
            "rps": rps,
            "rpm": rps * 60,
        }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._latencies.clear()
            self._latencies_by_agent.clear()
            self._errors.clear()
            self._total_requests = 0
            self._total_errors = 0
            self._request_timestamps.clear()
            self._cache.clear()
