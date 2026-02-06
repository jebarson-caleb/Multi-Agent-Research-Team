"""Communication protocol definitions for inter-agent messaging."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageType(str, Enum):
    """Types of messages in the inter-agent protocol."""
    TASK = "task"
    RESULT = "result"
    QUERY = "query"
    RESPONSE = "response"
    STATUS = "status"
    ERROR = "error"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"
    VERIFICATION_REQUEST = "verification_request"
    VERIFICATION_RESULT = "verification_result"


class Priority(str, Enum):
    """Message priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MessageProtocol:
    """Standardized message format for inter-agent communication.

    All messages between agents follow this protocol to ensure
    consistent handling, routing, and logging.
    """
    # Header
    message_id: str = ""
    message_type: MessageType = MessageType.TASK
    sender: str = ""
    recipient: str = ""
    priority: Priority = Priority.MEDIUM
    timestamp: float = 0.0
    correlation_id: str = ""
    reply_to: str = ""

    # Body
    content: dict[str, Any] = field(default_factory=dict)

    # Metadata
    ttl: int = 300  # Time to live in seconds
    retry_count: int = 0
    max_retries: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message_id:
            self.message_id = str(uuid.uuid4())[:12]
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for transmission."""
        return {
            "message_id": self.message_id,
            "type": self.message_type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "content": self.content,
            "ttl": self.ttl,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessageProtocol:
        """Create from dictionary."""
        return cls(
            message_id=data.get("message_id", ""),
            message_type=MessageType(data.get("type", "task")),
            sender=data.get("sender", ""),
            recipient=data.get("recipient", ""),
            priority=Priority(data.get("priority", "medium")),
            timestamp=data.get("timestamp", 0.0),
            correlation_id=data.get("correlation_id", ""),
            reply_to=data.get("reply_to", ""),
            content=data.get("content", {}),
            ttl=data.get("ttl", 300),
            retry_count=data.get("retry_count", 0),
            metadata=data.get("metadata", {}),
        )

    def is_expired(self) -> bool:
        """Check if the message has expired based on TTL."""
        return (time.time() - self.timestamp) > self.ttl

    def can_retry(self) -> bool:
        """Check if the message can be retried."""
        return self.retry_count < self.max_retries

    def create_reply(
        self,
        content: dict[str, Any],
        sender: str = "",
    ) -> MessageProtocol:
        """Create a reply message."""
        return MessageProtocol(
            message_type=MessageType.RESPONSE,
            sender=sender or self.recipient,
            recipient=self.sender,
            priority=self.priority,
            correlation_id=self.message_id,
            reply_to=self.sender,
            content=content,
        )

    @staticmethod
    def create_task(
        sender: str,
        recipient: str,
        task: dict[str, Any],
        priority: Priority = Priority.MEDIUM,
    ) -> MessageProtocol:
        """Create a task assignment message."""
        return MessageProtocol(
            message_type=MessageType.TASK,
            sender=sender,
            recipient=recipient,
            priority=priority,
            content=task,
        )

    @staticmethod
    def create_broadcast(
        sender: str,
        content: dict[str, Any],
        priority: Priority = Priority.LOW,
    ) -> MessageProtocol:
        """Create a broadcast message."""
        return MessageProtocol(
            message_type=MessageType.BROADCAST,
            sender=sender,
            recipient="broadcast",
            priority=priority,
            content=content,
        )

    @staticmethod
    def create_verification_request(
        sender: str,
        recipient: str,
        data_to_verify: dict[str, Any],
    ) -> MessageProtocol:
        """Create a verification request message."""
        return MessageProtocol(
            message_type=MessageType.VERIFICATION_REQUEST,
            sender=sender,
            recipient=recipient,
            priority=Priority.HIGH,
            content={"data": data_to_verify},
        )
