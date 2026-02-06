"""Agent module for the collaborative AI research team."""

from src.agents.base_agent import BaseAgent
from src.agents.researcher_agent import ResearcherAgent
from src.agents.analyst_agent import AnalystAgent
from src.agents.synthesizer_agent import SynthesizerAgent
from src.agents.coordinator_agent import CoordinatorAgent

__all__ = [
    "BaseAgent",
    "ResearcherAgent",
    "AnalystAgent",
    "SynthesizerAgent",
    "CoordinatorAgent",
]
