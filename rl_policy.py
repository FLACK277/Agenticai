"""Lightweight RL policy for response strategy selection.

This module keeps a tiny Q-table on disk and updates it from user feedback.
"""

import json
import os
import random
import threading
from typing import Dict, List


DEFAULT_STATES = [
    "realtime",
    "comparison",
    "explainer",
    "summary",
    "coding",
    "general",
]

DEFAULT_ACTIONS = [
    "concise",
    "detailed",
    "with_citations",
    "step_by_step",
]

ACTION_PROMPTS = {
    "concise": "Keep your response short and direct, with only essential details.",
    "detailed": "Provide a comprehensive response with context and clear structure.",
    "with_citations": "Prefer evidence-based claims and cite sources used by the tools.",
    "step_by_step": "Explain your reasoning as a sequence of actionable steps.",
}


class ReinforcementPolicy:
    """Small epsilon-greedy Q-learning policy persisted as JSON."""

    def __init__(
        self,
        file_path: str = "rl_policy.json",
        alpha: float = 0.2,
        gamma: float = 0.0,
        epsilon: float = 0.15,
        states: List[str] | None = None,
        actions: List[str] | None = None,
    ):
        self.file_path = file_path
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.states = states or DEFAULT_STATES
        self.actions = actions or DEFAULT_ACTIONS
        self._lock = threading.Lock()
        self.q_table: Dict[str, Dict[str, float]] = {}
        self._load()

    def _fresh_table(self) -> Dict[str, Dict[str, float]]:
        return {state: {action: 0.0 for action in self.actions} for state in self.states}

    def _load(self) -> None:
        if not os.path.exists(self.file_path):
            self.q_table = self._fresh_table()
            self._save()
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = self._fresh_table()

        # Ensure schema safety when states/actions evolve.
        for state in self.states:
            data.setdefault(state, {})
            for action in self.actions:
                data[state].setdefault(action, 0.0)

        self.q_table = data
        self._save()

    def _save(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.q_table, f, indent=2, sort_keys=True)

    def classify_state(self, text: str) -> str:
        content = text.lower()

        if any(word in content for word in ["price", "weather", "news", "today", "live"]):
            return "realtime"
        if any(word in content for word in ["compare", "vs", "difference", "better"]):
            return "comparison"
        if any(word in content for word in ["explain", "why", "how does", "what is"]):
            return "explainer"
        if any(word in content for word in ["summarize", "summary", "tl;dr"]):
            return "summary"
        if any(word in content for word in ["code", "python", "bug", "function", "api"]):
            return "coding"
        return "general"

    def choose_action(self, state: str) -> str:
        state = state if state in self.q_table else "general"

        with self._lock:
            if random.random() < self.epsilon:
                return random.choice(self.actions)

            state_values = self.q_table[state]
            max_q = max(state_values.values())
            candidates = [action for action, value in state_values.items() if value == max_q]
            return sorted(candidates)[0]

    def update(self, state: str, action: str, reward: float, next_state: str | None = None) -> float:
        if state not in self.q_table or action not in self.q_table[state]:
            return 0.0

        with self._lock:
            current_q = self.q_table[state][action]
            max_next_q = 0.0

            if next_state and next_state in self.q_table:
                max_next_q = max(self.q_table[next_state].values())

            td_target = reward + (self.gamma * max_next_q)
            updated_q = current_q + self.alpha * (td_target - current_q)
            self.q_table[state][action] = round(updated_q, 4)
            self._save()
            return updated_q

    def get_style_instruction(self, action: str) -> str:
        return ACTION_PROMPTS.get(action, ACTION_PROMPTS["detailed"])

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return json.loads(json.dumps(self.q_table))
