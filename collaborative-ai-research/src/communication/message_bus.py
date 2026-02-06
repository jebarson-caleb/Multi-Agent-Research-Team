"""Message bus for inter-agent communication with pub/sub pattern."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional


@dataclass
class Message:
    """Structured message for inter-agent communication."""
    id: str
    channel: str
    type: str
    sender: str
    content: Any
    timestamp: float
    reply_to: str = ""
    correlation_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


MessageHandler = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class MessageBus:
    """Async pub/sub message bus for inter-agent communication.

    Features:
    - Channel-based publish/subscribe
    - Request/response pattern support
    - Broadcast messages to all subscribers
    - Message history for debugging
    - Dead letter queue for undeliverable messages
    """

    def __init__(self, max_history: int = 1000) -> None:
        """Initialize the message bus.

        Args:
            max_history: Maximum messages to keep in history.
        """
        self._subscribers: dict[str, list[MessageHandler]] = defaultdict(list)
        self._history: list[Message] = []
        self._max_history = max_history
        self._dead_letters: list[Message] = []
        self._pending_responses: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
        self._message_count = 0

    def subscribe(self, channel: str, handler: MessageHandler) -> None:
        """Subscribe a handler to a channel.

        Args:
            channel: Channel name to subscribe to.
            handler: Async callable to handle messages.
        """
        self._subscribers[channel].append(handler)

    def unsubscribe(self, channel: str, handler: MessageHandler) -> None:
        """Unsubscribe a handler from a channel.

        Args:
            channel: Channel to unsubscribe from.
            handler: Handler to remove.
        """
        if channel in self._subscribers:
            self._subscribers[channel] = [
                h for h in self._subscribers[channel] if h != handler
            ]

    async def publish(
        self,
        channel: str,
        message: dict[str, Any],
        sender: str = "system",
    ) -> str:
        """Publish a message to a channel.

        Args:
            channel: Target channel.
            message: Message content dict.
            sender: Sender identifier.

        Returns:
            Message ID.
        """
        msg_id = str(uuid.uuid4())[:8]
        self._message_count += 1

        msg = Message(
            id=msg_id,
            channel=channel,
            type=message.get("type", "generic"),
            sender=sender,
            content=message,
            timestamp=time.time(),
            reply_to=message.get("reply_to", ""),
            correlation_id=message.get("correlation_id", ""),
            metadata=message.get("metadata", {}),
        )

        # Store in history
        async with self._lock:
            self._history.append(msg)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        # Deliver to subscribers
        handlers = self._subscribers.get(channel, [])
        if not handlers:
            self._dead_letters.append(msg)
            return msg_id

        delivery_tasks = []
        for handler in handlers:
            delivery_tasks.append(self._deliver(handler, message, msg))

        if delivery_tasks:
            await asyncio.gather(*delivery_tasks, return_exceptions=True)

        # Check for pending response
        correlation_id = message.get("correlation_id", "")
        if correlation_id and correlation_id in self._pending_responses:
            self._pending_responses[correlation_id].set_result(message)

        return msg_id

    async def request(
        self,
        channel: str,
        message: dict[str, Any],
        sender: str = "system",
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Send a request and wait for a response.

        Args:
            channel: Target channel.
            message: Request message.
            sender: Sender identifier.
            timeout: Response timeout in seconds.

        Returns:
            Response message dict.

        Raises:
            asyncio.TimeoutError: If no response within timeout.
        """
        correlation_id = str(uuid.uuid4())[:8]
        message["correlation_id"] = correlation_id
        message["reply_to"] = sender

        # Create future for response
        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_responses[correlation_id] = future

        try:
            await self.publish(channel, message, sender)
            return await asyncio.wait_for(future, timeout)
        finally:
            self._pending_responses.pop(correlation_id, None)

    async def broadcast(
        self,
        message: dict[str, Any],
        sender: str = "system",
        exclude: list[str] | None = None,
    ) -> list[str]:
        """Broadcast a message to all channels.

        Args:
            message: Message to broadcast.
            sender: Sender identifier.
            exclude: Channels to exclude.

        Returns:
            List of message IDs.
        """
        exclude = exclude or []
        msg_ids = []
        for channel in self._subscribers:
            if channel not in exclude:
                msg_id = await self.publish(channel, message, sender)
                msg_ids.append(msg_id)
        return msg_ids

    async def _deliver(
        self,
        handler: MessageHandler,
        message: dict[str, Any],
        msg: Message,
    ) -> None:
        """Deliver a message to a handler with error handling."""
        try:
            await handler(message)
        except Exception as e:
            self._dead_letters.append(msg)

    def get_history(
        self,
        channel: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get message history.

        Args:
            channel: Optional filter by channel.
            limit: Maximum messages to return.

        Returns:
            List of message dicts.
        """
        messages = self._history
        if channel:
            messages = [m for m in messages if m.channel == channel]

        recent = messages[-limit:]
        return [
            {
                "id": m.id,
                "channel": m.channel,
                "type": m.type,
                "sender": m.sender,
                "timestamp": m.timestamp,
                "content_preview": str(m.content)[:200],
            }
            for m in recent
        ]

    def get_dead_letters(self) -> list[dict[str, Any]]:
        """Get undeliverable messages."""
        return [
            {
                "id": m.id,
                "channel": m.channel,
                "sender": m.sender,
                "timestamp": m.timestamp,
            }
            for m in self._dead_letters
        ]

    def get_metrics(self) -> dict[str, Any]:
        """Get message bus metrics."""
        return {
            "total_messages": self._message_count,
            "channels": len(self._subscribers),
            "subscribers": {
                ch: len(handlers) for ch, handlers in self._subscribers.items()
            },
            "history_size": len(self._history),
            "dead_letters": len(self._dead_letters),
            "pending_responses": len(self._pending_responses),
        }
