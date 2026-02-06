"""Communication module for inter-agent messaging."""

from src.communication.message_bus import MessageBus
from src.communication.protocol import MessageProtocol, MessageType

__all__ = ["MessageBus", "MessageProtocol", "MessageType"]
