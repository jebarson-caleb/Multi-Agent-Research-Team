"""Coordinator Agent - Orchestrates research tasks and manages workflow."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

from src.agents.base_agent import BaseAgent, AgentState


class CoordinatorAgent(BaseAgent):
    """Agent that orchestrates research tasks across the team.

    Capabilities:
    - Decompose complex queries into subtasks
    - Assign subtasks to appropriate specialist agents
    - Manage dependencies and execution order
    - Track progress and handle failures
    - Synthesize results from multiple agents
    """

    DECOMPOSITION_PROMPT = """You are a research coordinator. Decompose the following research query into subtasks.

Query: {query}

Available Agents:
- researcher: Information gathering, fact extraction, source evaluation
- analyst: Data analysis, pattern recognition, insight generation
- synthesizer: Information synthesis, report generation, summarization

Depth: {depth}

Decompose into subtasks in the following JSON format:
{{
    "task_plan": {{
        "objective": "Main research objective",
        "subtasks": [
            {{
                "id": "task_1",
                "agent": "researcher|analyst|synthesizer",
                "type": "research|analysis|synthesis",
                "description": "What this subtask should accomplish",
                "topic": "Specific topic or focus area",
                "instructions": "Detailed instructions for the agent",
                "dependencies": [],
                "priority": "critical|high|medium|low",
                "estimated_tokens": 1000
            }}
        ],
        "execution_order": ["task_1", "task_2"],
        "parallel_groups": [["task_1", "task_2"], ["task_3"]]
    }}
}}

Rules:
1. Research tasks should come before analysis tasks
2. Analysis tasks should come before synthesis tasks
3. Identify tasks that can run in parallel
4. Estimate token usage for each subtask
5. For "{depth}" depth, adjust the number and detail of subtasks accordingly"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._agents: dict[str, BaseAgent] = {}
        self._task_results: dict[str, dict[str, Any]] = {}
        self._progress: dict[str, str] = {}

    def register_agents(self, agents: dict[str, BaseAgent]) -> None:
        """Register specialist agents with the coordinator.

        Args:
            agents: Dict mapping agent names to agent instances.
        """
        self._agents = agents

    async def execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute a coordination task (decompose and orchestrate).

        Args:
            task: Task with 'query' and optional parameters.

        Returns:
            Coordinated results from all agents.
        """
        query = task.get("query", "")
        depth = task.get("depth", "standard")

        task_plan = await self.decompose_task(query, depth=depth)
        results = await self.execute_plan(task_plan)

        return {
            "agent": self._name,
            "task_type": "coordination",
            "task_plan": task_plan,
            "results": results,
        }

    async def decompose_task(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Decompose a research query into subtasks.

        Args:
            query: Research query to decompose.
            **kwargs: Additional parameters (depth, workflow, etc.)

        Returns:
            Task plan with subtasks and execution order.
        """
        self._state = AgentState.PROCESSING
        depth = kwargs.get("depth", "standard")

        prompt = self.DECOMPOSITION_PROMPT.format(
            query=query,
            depth=depth,
        )

        response = await self.call_llm(
            messages=[{"role": "user", "content": prompt}],
        )

        task_plan = self._parse_task_plan(response["content"])

        # Store plan in shared memory
        await self.store_context(
            content=json.dumps(task_plan, indent=2),
            namespace="coordination",
            priority="critical",
        )

        self._state = AgentState.IDLE
        return task_plan

    async def execute_plan(self, task_plan: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute a task plan by delegating to agents.

        Args:
            task_plan: Task plan from decompose_task.

        Returns:
            List of results from all subtasks.
        """
        self._state = AgentState.PROCESSING
        results = []
        plan = task_plan.get("task_plan", task_plan)
        subtasks = plan.get("subtasks", [])
        parallel_groups = plan.get("parallel_groups", [[t["id"] for t in subtasks]])

        # Execute subtasks respecting dependencies and parallel groups
        for group in parallel_groups:
            group_tasks = [t for t in subtasks if t.get("id") in group]

            if len(group_tasks) == 1:
                # Single task, execute directly
                result = await self._execute_subtask(group_tasks[0])
                results.append(result)
            elif len(group_tasks) > 1:
                # Multiple tasks, execute in parallel
                group_results = await asyncio.gather(
                    *[self._execute_subtask(t) for t in group_tasks],
                    return_exceptions=True,
                )
                for r in group_results:
                    if isinstance(r, Exception):
                        results.append({
                            "error": str(r),
                            "task_type": "failed",
                        })
                    else:
                        results.append(r)

        self._state = AgentState.IDLE
        return results

    async def _execute_subtask(self, subtask: dict[str, Any]) -> dict[str, Any]:
        """Execute a single subtask by delegating to the appropriate agent.

        Args:
            subtask: Subtask specification.

        Returns:
            Subtask result.
        """
        task_id = subtask.get("id", "unknown")
        agent_name = subtask.get("agent", "researcher")
        self._progress[task_id] = "in_progress"

        agent = self._agents.get(agent_name)
        if agent is None:
            self._progress[task_id] = "failed"
            return {
                "task_id": task_id,
                "error": f"Agent '{agent_name}' not found",
            }

        try:
            # Build task for the agent
            task = {
                "topic": subtask.get("topic", subtask.get("description", "")),
                "query": subtask.get("topic", subtask.get("description", "")),
                "instructions": subtask.get("instructions", ""),
                "depth": "standard",
                "findings": self._get_previous_results(),
            }

            result = await agent.execute_task(task)
            self._task_results[task_id] = result
            self._progress[task_id] = "completed"
            return result

        except Exception as e:
            self._progress[task_id] = "failed"
            return {
                "task_id": task_id,
                "agent": agent_name,
                "error": str(e),
            }

    def _get_previous_results(self) -> list[dict[str, Any]]:
        """Get results from previously completed tasks."""
        return [
            r for r in self._task_results.values()
            if r.get("result") is not None
        ]

    def _parse_task_plan(self, content: str) -> dict[str, Any]:
        """Parse LLM response into a structured task plan."""
        try:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])
        except json.JSONDecodeError:
            pass

        # Fallback: create a simple sequential plan
        return {
            "task_plan": {
                "objective": "Research task",
                "subtasks": [
                    {
                        "id": "task_1",
                        "agent": "researcher",
                        "type": "research",
                        "description": content[:200],
                        "topic": content[:100],
                        "instructions": "Gather information on this topic.",
                        "dependencies": [],
                        "priority": "high",
                        "estimated_tokens": 2000,
                    },
                    {
                        "id": "task_2",
                        "agent": "analyst",
                        "type": "analysis",
                        "description": "Analyze research findings",
                        "topic": "Analysis of gathered research",
                        "instructions": "Analyze the research findings for patterns and insights.",
                        "dependencies": ["task_1"],
                        "priority": "high",
                        "estimated_tokens": 2000,
                    },
                ],
                "execution_order": ["task_1", "task_2"],
                "parallel_groups": [["task_1"], ["task_2"]],
            },
        }

    def get_progress(self) -> dict[str, Any]:
        """Get current execution progress."""
        total = len(self._progress)
        completed = sum(1 for s in self._progress.values() if s == "completed")
        failed = sum(1 for s in self._progress.values() if s == "failed")

        return {
            "total_tasks": total,
            "completed": completed,
            "failed": failed,
            "in_progress": total - completed - failed,
            "progress_pct": (completed / total * 100) if total > 0 else 0,
            "tasks": dict(self._progress),
        }
