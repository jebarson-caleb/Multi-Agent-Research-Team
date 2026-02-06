"""Base agent class with OpenAI-compatible API integration, retry logic, and token tracking."""

from __future__ import annotations

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import openai
import aiohttp
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, InternalServerError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.context.shared_memory import SharedMemory
from src.communication.message_bus import MessageBus
from src.utils.token_counter import TokenCounter
from src.security.audit_logger import AuditLogger


class AgentRole(str, Enum):
    """Agent role enumeration."""
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    SYNTHESIZER = "synthesizer"


class AgentState(str, Enum):
    """Agent lifecycle states."""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class AgentMetrics:
    """Metrics tracked per agent."""
    total_requests: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_errors: int = 0
    total_retries: int = 0
    total_latency: float = 0.0
    requests_per_minute: list[float] = field(default_factory=list)

    @property
    def avg_latency(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency / self.total_requests

    @property
    def total_tokens(self) -> int:
        return self.total_tokens_input + self.total_tokens_output

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_tokens_input": self.total_tokens_input,
            "total_tokens_output": self.total_tokens_output,
            "total_tokens": self.total_tokens,
            "total_errors": self.total_errors,
            "total_retries": self.total_retries,
            "avg_latency": self.avg_latency,
            "error_rate": self.error_rate,
        }


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 100000,
    ) -> None:
        self._rpm_limit = requests_per_minute
        self._tpm_limit = tokens_per_minute
        self._request_timestamps: list[float] = []
        self._token_counts: list[tuple[float, int]] = []
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 0) -> None:
        """Wait until rate limit allows the request."""
        async with self._lock:
            now = time.time()
            minute_ago = now - 60

            # Clean old entries
            self._request_timestamps = [
                ts for ts in self._request_timestamps if ts > minute_ago
            ]
            self._token_counts = [
                (ts, count) for ts, count in self._token_counts if ts > minute_ago
            ]

            # Check request rate
            while len(self._request_timestamps) >= self._rpm_limit:
                wait_time = self._request_timestamps[0] - minute_ago
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                now = time.time()
                minute_ago = now - 60
                self._request_timestamps = [
                    ts for ts in self._request_timestamps if ts > minute_ago
                ]

            # Check token rate
            current_tokens = sum(count for _, count in self._token_counts)
            while current_tokens + estimated_tokens > self._tpm_limit:
                wait_time = self._token_counts[0][0] - minute_ago
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                now = time.time()
                minute_ago = now - 60
                self._token_counts = [
                    (ts, count) for ts, count in self._token_counts if ts > minute_ago
                ]
                current_tokens = sum(count for _, count in self._token_counts)

            # Record this request
            self._request_timestamps.append(now)
            if estimated_tokens > 0:
                self._token_counts.append((now, estimated_tokens))


class BaseAgent(ABC):
    """Abstract base class for all AI agents.

    Provides:
    - OpenAI-compatible API integration (supports Gemini, GPT, etc. via gateway)
    - Automatic retry with exponential backoff
    - Rate limiting
    - Token usage tracking
    - State management
    - Audit logging
    - Context management via shared memory
    """

    def __init__(
        self,
        config: dict[str, Any],
        shared_memory: SharedMemory,
        message_bus: MessageBus,
        token_counter: TokenCounter,
        audit_logger: AuditLogger,
    ) -> None:
        """Initialize the base agent.

        Args:
            config: Agent-specific configuration from agents.yaml.
            shared_memory: Shared memory store for context sharing.
            message_bus: Message bus for inter-agent communication.
            token_counter: Token counter for usage tracking.
            audit_logger: Audit logger for security logging.
        """
        self._config = config
        self._shared_memory = shared_memory
        self._message_bus = message_bus
        self._token_counter = token_counter
        self._audit_logger = audit_logger

        # Agent identity
        self._name = config.get("name", self.__class__.__name__)
        self._role = self._get_role()
        self._model = config.get("model", os.getenv("DEFAULT_MODEL", "gemini-2.5-flash"))
        env_max_tokens = os.getenv("MAX_TOKENS", "")
        if env_max_tokens.isdigit():
            self._max_tokens = int(env_max_tokens)
        else:
            self._max_tokens = config.get("max_tokens", 4096)
        self._temperature = config.get("temperature", 0.3)
        self._system_prompt = config.get("system_prompt", "")

        # State management
        self._state = AgentState.IDLE
        self._metrics = AgentMetrics()

        # Rate limiter
        rate_config = config.get("rate_limit", {})
        self._rate_limiter = RateLimiter(
            requests_per_minute=rate_config.get("requests_per_minute", 60),
            tokens_per_minute=rate_config.get("tokens_per_minute", 100000),
        )

        # OpenAI-compatible client (lazy initialization)
        self._client: Optional[AsyncOpenAI] = None

    def _get_role(self) -> AgentRole:
        """Get the agent's role based on class name."""
        class_name = self.__class__.__name__.lower()
        for role in AgentRole:
            if role.value in class_name:
                return role
        return AgentRole.RESEARCHER

    def _get_client(self) -> AsyncOpenAI:
        """Get or create the OpenAI-compatible client."""
        if self._client is None:
            api_key = os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", ""))
            base_url = os.getenv("API_BASE_URL", "https://router.requesty.ai/v1")
            if not api_key:
                raise ValueError(
                    "API_KEY environment variable not set. "
                    "Set it in your .env file or environment."
                )
            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        return self._client

    @property
    def name(self) -> str:
        return self._name

    @property
    def role(self) -> AgentRole:
        return self._role

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def metrics(self) -> AgentMetrics:
        return self._metrics

    async def call_llm(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an API call to the LLM with retry and rate limiting.

        Uses the OpenAI-compatible chat completions API, supporting
        Gemini, GPT, and other models via gateway.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            system: System prompt override.
            max_tokens: Max tokens override.
            temperature: Temperature override.
            **kwargs: Additional API parameters.

        Returns:
            Dict with 'content', 'tokens_input', 'tokens_output', 'model'.
        """
        self._state = AgentState.PROCESSING
        start_time = time.time()

        # Apply rate limiting
        estimated_tokens = sum(len(m.get("content", "")) // 4 for m in messages)
        await self._rate_limiter.acquire(estimated_tokens)

        try:
            # Prepend system message if provided
            system_prompt = system or self._system_prompt
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)

            response = await self._make_api_call(
                messages=full_messages,
                max_tokens=max_tokens or self._max_tokens,
                temperature=temperature if temperature is not None else self._temperature,
                **kwargs,
            )

            # Extract response data
            if isinstance(response, dict):
                content = response.get("content", "")
                tokens_input = response.get("tokens_input", 0)
                tokens_output = response.get("tokens_output", 0)
            else:
                content = response.choices[0].message.content if response.choices else ""
                tokens_input = response.usage.prompt_tokens if response.usage else 0
                tokens_output = response.usage.completion_tokens if response.usage else 0

            # Update metrics
            latency = time.time() - start_time
            self._metrics.total_requests += 1
            self._metrics.total_tokens_input += tokens_input
            self._metrics.total_tokens_output += tokens_output
            self._metrics.total_latency += latency

            # Track tokens globally
            self._token_counter.add(
                agent=self._name,
                input_tokens=tokens_input,
                output_tokens=tokens_output,
                model=self._model,
            )

            # Audit log
            self._audit_logger.log_event("agent_request", {
                "agent": self._name,
                "model": self._model,
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                "latency": latency,
            })

            self._state = AgentState.IDLE
            return {
                "content": content,
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                "model": self._model,
                "latency": latency,
            }

        except Exception as e:
            self._metrics.total_errors += 1
            self._state = AgentState.ERROR
            self._audit_logger.log_event("agent_error", {
                "agent": self._name,
                "error": str(e),
            })
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((
            openai.RateLimitError,
            openai.InternalServerError,
            openai.APIConnectionError,
            aiohttp.ClientError,
        )),
    )
    async def _make_api_call(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> Any:
        """Make the actual API call with retry logic.

        Uses tenacity for automatic retry with exponential backoff
        on rate limit, server, and connection errors.
        """
        provider = os.getenv("API_PROVIDER", "openai").lower()
        if provider in {"google", "google_ai", "google_ai_studio", "gemini"}:
            return await self._make_google_api_call(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        if provider in {"ollama", "local", "llama_cpp"}:
            return await self._make_ollama_api_call(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

        client = self._get_client()
        return await client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
            **kwargs,
        )

    def _normalize_google_model(self, model: str) -> str:
        """Normalize model names for Google AI Studio."""
        model_name = model.replace("models/", "")
        if model_name.startswith("gpt-"):
            return "gemini-2.0-flash"
        if not model_name.startswith("gemini-"):
            return "gemini-2.0-flash"
        return model_name

    async def _make_google_api_call(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call Google AI Studio (Gemini) API using generateContent."""
        api_key = os.getenv("API_KEY", "")
        base_url = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
        if not api_key:
            raise ValueError(
                "API_KEY environment variable not set. "
                "Set it in your .env file or environment."
            )

        system_instruction = None
        contents: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                system_instruction = content
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": content}]})

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        model_name = self._normalize_google_model(self._model)
        url = f"{base_url}/models/{model_name}:generateContent"
        params = {"key": api_key}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params, json=payload) as response:
                data = await response.json()
                if response.status >= 400:
                    message = data.get("error", {}).get("message", str(data))
                    raise RuntimeError(f"Google AI Studio error: {message}")

        content_text = ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            content_text = "".join(p.get("text", "") for p in parts)

        usage = data.get("usageMetadata", {})
        tokens_input = usage.get("promptTokenCount", 0)
        tokens_output = usage.get("candidatesTokenCount", 0)

        return {
            "content": content_text,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
        }

    async def _make_ollama_api_call(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call a local Ollama server using the chat API."""
        base_url = os.getenv("API_BASE_URL", "http://localhost:11434")
        url = f"{base_url}/api/chat"

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                if response.status >= 400:
                    message = data.get("error", str(data))
                    raise RuntimeError(f"Ollama error: {message}")

        message = data.get("message", {})
        content_text = message.get("content", "")
        tokens_input = data.get("prompt_eval_count", 0)
        tokens_output = data.get("eval_count", 0)

        return {
            "content": content_text,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
        }

    async def get_compressed_context(self, namespace: str = "global") -> str:
        """Get compressed shared context for this agent.

        Args:
            namespace: Context namespace to retrieve.

        Returns:
            Compressed context string.
        """
        return await self._shared_memory.get_compressed(
            namespace=namespace,
            agent=self._name,
        )

    async def store_context(
        self,
        content: str,
        namespace: str = "global",
        priority: str = "medium",
    ) -> None:
        """Store context in shared memory.

        Args:
            content: Context content to store.
            namespace: Context namespace.
            priority: Priority level (critical, high, medium, low).
        """
        await self._shared_memory.store(
            agent=self._name,
            namespace=namespace,
            content=content,
            priority=priority,
        )

    async def handle_message(self, message: dict[str, Any]) -> None:
        """Handle an incoming message from the message bus.

        Args:
            message: Message dict with 'type', 'from', 'content', etc.
        """
        msg_type = message.get("type", "")
        if msg_type == "task":
            await self._handle_task(message)
        elif msg_type == "query":
            await self._handle_query(message)
        elif msg_type == "status":
            await self._handle_status_request(message)
        else:
            self._audit_logger.log_event("unknown_message_type", {
                "agent": self._name,
                "type": msg_type,
            })

    async def _handle_task(self, message: dict[str, Any]) -> None:
        """Handle a task assignment message."""
        task = message.get("content", {})
        result = await self.execute_task(task)
        await self._message_bus.publish(
            channel=message.get("reply_to", "coordinator"),
            message={
                "type": "result",
                "from": self._name,
                "task_id": message.get("task_id"),
                "content": result,
            },
        )

    async def _handle_query(self, message: dict[str, Any]) -> None:
        """Handle a query message."""
        pass  # Subclasses can override

    async def _handle_status_request(self, message: dict[str, Any]) -> None:
        """Handle a status request message."""
        await self._message_bus.publish(
            channel=message.get("reply_to", "coordinator"),
            message={
                "type": "status",
                "from": self._name,
                "content": {
                    "state": self._state.value,
                    "metrics": self._metrics.to_dict(),
                },
            },
        )

    @abstractmethod
    async def execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute a task assigned to this agent.

        Args:
            task: Task specification dict.

        Returns:
            Task result dict.
        """
        ...

    def get_status(self) -> dict[str, Any]:
        """Get current agent status."""
        return {
            "name": self._name,
            "role": self._role.value,
            "state": self._state.value,
            "model": self._model,
            "metrics": self._metrics.to_dict(),
        }
