"""Tests for the agent framework."""

from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.base_agent import BaseAgent, AgentRole, AgentState, AgentMetrics, RateLimiter
from src.agents.researcher_agent import ResearcherAgent
from src.agents.analyst_agent import AnalystAgent
from src.agents.synthesizer_agent import SynthesizerAgent
from src.agents.coordinator_agent import CoordinatorAgent
from src.context.shared_memory import SharedMemory
from src.context.compressor import ContextCompressor
from src.communication.message_bus import MessageBus
from src.utils.token_counter import TokenCounter
from src.security.audit_logger import AuditLogger


@pytest.fixture
def mock_components():
    """Create mock components for agent initialization."""
    compressor = ContextCompressor()
    shared_memory = SharedMemory(compressor)
    message_bus = MessageBus()
    token_counter = TokenCounter()
    audit_logger = AuditLogger()
    return {
        "shared_memory": shared_memory,
        "message_bus": message_bus,
        "token_counter": token_counter,
        "audit_logger": audit_logger,
    }


@pytest.fixture
def agent_config():
    """Create a standard agent config."""
    return {
        "name": "TestAgent",
        "model": "gemini-2.5-flash",
        "max_tokens": 4096,
        "temperature": 0.3,
        "system_prompt": "You are a test agent.",
        "rate_limit": {
            "requests_per_minute": 60,
            "tokens_per_minute": 100000,
        },
    }


@pytest.fixture
def researcher(mock_components, agent_config):
    """Create a ResearcherAgent instance."""
    agent_config["name"] = "Researcher"
    return ResearcherAgent(config=agent_config, **mock_components)


@pytest.fixture
def analyst(mock_components, agent_config):
    """Create an AnalystAgent instance."""
    agent_config["name"] = "Analyst"
    return AnalystAgent(config=agent_config, **mock_components)


@pytest.fixture
def synthesizer(mock_components, agent_config):
    """Create a SynthesizerAgent instance."""
    agent_config["name"] = "Synthesizer"
    agent_config["max_tokens"] = 8192
    return SynthesizerAgent(config=agent_config, **mock_components)


@pytest.fixture
def coordinator(mock_components, agent_config):
    """Create a CoordinatorAgent instance."""
    agent_config["name"] = "Coordinator"
    return CoordinatorAgent(config=agent_config, **mock_components)


class TestAgentMetrics:
    """Tests for AgentMetrics dataclass."""

    def test_initial_metrics(self):
        metrics = AgentMetrics()
        assert metrics.total_requests == 0
        assert metrics.total_tokens == 0
        assert metrics.avg_latency == 0.0
        assert metrics.error_rate == 0.0

    def test_metrics_calculation(self):
        metrics = AgentMetrics(
            total_requests=10,
            total_tokens_input=5000,
            total_tokens_output=3000,
            total_errors=1,
            total_latency=20.0,
        )
        assert metrics.total_tokens == 8000
        assert metrics.avg_latency == 2.0
        assert metrics.error_rate == 0.1

    def test_metrics_to_dict(self):
        metrics = AgentMetrics(total_requests=5)
        d = metrics.to_dict()
        assert "total_requests" in d
        assert d["total_requests"] == 5


class TestRateLimiter:
    """Tests for the RateLimiter."""

    @pytest.mark.asyncio
    async def test_acquire_within_limits(self):
        limiter = RateLimiter(requests_per_minute=100, tokens_per_minute=1000000)
        # Should not block
        await limiter.acquire(100)

    @pytest.mark.asyncio
    async def test_rate_limiter_creation(self):
        limiter = RateLimiter(requests_per_minute=10, tokens_per_minute=10000)
        assert limiter._rpm_limit == 10
        assert limiter._tpm_limit == 10000


class TestAgentInitialization:
    """Tests for agent initialization."""

    def test_researcher_init(self, researcher):
        assert researcher.name == "Researcher"
        assert researcher.state == AgentState.IDLE

    def test_analyst_init(self, analyst):
        assert analyst.name == "Analyst"
        assert analyst.state == AgentState.IDLE

    def test_synthesizer_init(self, synthesizer):
        assert synthesizer.name == "Synthesizer"
        assert synthesizer.state == AgentState.IDLE

    def test_coordinator_init(self, coordinator):
        assert coordinator.name == "Coordinator"
        assert coordinator.state == AgentState.IDLE

    def test_agent_status(self, researcher):
        status = researcher.get_status()
        assert status["name"] == "Researcher"
        assert status["state"] == "idle"
        assert "metrics" in status


class TestResearcherAgent:
    """Tests for the ResearcherAgent."""

    def test_build_instructions_quick(self, researcher):
        instructions = researcher._build_instructions("Test", "quick")
        assert "concise" in instructions.lower() or "3-5" in instructions

    def test_build_instructions_standard(self, researcher):
        instructions = researcher._build_instructions("Test", "standard")
        assert "comprehensive" in instructions.lower()

    def test_build_instructions_deep(self, researcher):
        instructions = researcher._build_instructions("Test", "deep")
        assert "exhaustive" in instructions.lower()

    def test_parse_research_response_valid_json(self, researcher):
        json_response = json.dumps({
            "findings": [{"fact": "Test fact", "confidence": 0.9}],
            "summary": "Test summary",
            "gaps": [],
            "suggested_follow_up": [],
        })
        result = researcher._parse_research_response(json_response)
        assert "findings" in result
        assert len(result["findings"]) == 1

    def test_parse_research_response_invalid_json(self, researcher):
        result = researcher._parse_research_response("This is not JSON")
        assert "findings" in result
        assert result["findings"][0]["confidence"] == 0.5


class TestAnalystAgent:
    """Tests for the AnalystAgent."""

    def test_parse_analysis_response_valid(self, analyst):
        json_response = json.dumps({
            "patterns": [{"pattern": "Test pattern", "confidence": 0.8}],
            "insights": [],
            "trends": [],
            "limitations": [],
            "summary": "Test analysis",
        })
        result = analyst._parse_analysis_response(json_response)
        assert "patterns" in result
        assert len(result["patterns"]) == 1

    def test_parse_analysis_response_invalid(self, analyst):
        result = analyst._parse_analysis_response("Not JSON content")
        assert "insights" in result
        assert "summary" in result


class TestSynthesizerAgent:
    """Tests for the SynthesizerAgent."""

    def test_parse_synthesis_response_valid(self, synthesizer):
        json_response = json.dumps({
            "summary": "Test synthesis",
            "key_findings": [],
            "conclusions": ["Conclusion 1"],
            "recommendations": [],
            "sources": [],
            "confidence": 0.85,
        })
        result = synthesizer._parse_synthesis_response(json_response)
        assert result["summary"] == "Test synthesis"
        assert result["confidence"] == 0.85

    def test_parse_synthesis_response_invalid(self, synthesizer):
        result = synthesizer._parse_synthesis_response("Plain text response")
        assert "summary" in result
        assert result["confidence"] == 0.5


class TestCoordinatorAgent:
    """Tests for the CoordinatorAgent."""

    def test_register_agents(self, coordinator, researcher, analyst):
        coordinator.register_agents({
            "researcher": researcher,
            "analyst": analyst,
        })
        assert "researcher" in coordinator._agents
        assert "analyst" in coordinator._agents

    def test_parse_task_plan_valid(self, coordinator):
        json_plan = json.dumps({
            "task_plan": {
                "objective": "Test research",
                "subtasks": [
                    {
                        "id": "task_1",
                        "agent": "researcher",
                        "type": "research",
                        "description": "Gather information",
                        "topic": "Test topic",
                        "instructions": "Test instructions",
                        "dependencies": [],
                        "priority": "high",
                        "estimated_tokens": 1000,
                    }
                ],
                "execution_order": ["task_1"],
                "parallel_groups": [["task_1"]],
            }
        })
        result = coordinator._parse_task_plan(json_plan)
        assert "task_plan" in result
        assert len(result["task_plan"]["subtasks"]) == 1

    def test_parse_task_plan_fallback(self, coordinator):
        result = coordinator._parse_task_plan("Not valid JSON")
        assert "task_plan" in result
        assert len(result["task_plan"]["subtasks"]) == 2  # Fallback creates 2 tasks

    def test_get_progress_empty(self, coordinator):
        progress = coordinator.get_progress()
        assert progress["total_tasks"] == 0
        assert progress["progress_pct"] == 0


class TestMessageHandling:
    """Tests for agent message handling."""

    @pytest.mark.asyncio
    async def test_handle_status_request(self, researcher):
        """Test that agents respond to status requests."""
        received_messages = []

        async def capture_message(msg):
            received_messages.append(msg)

        researcher._message_bus.subscribe("test_reply", capture_message)

        await researcher._handle_status_request({
            "reply_to": "test_reply",
        })

        assert len(received_messages) == 1
        assert received_messages[0]["type"] == "status"
