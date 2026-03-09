"""Lightweight RL policy for response strategy selection.

This module keeps a tiny Q-table on disk and updates it from user feedback.
"""

import json
import os
import random
import threading
import uuid
from typing import Any, Dict, List


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

POSITIVE_REACTION_MARKERS = [
    "thanks",
    "thank you",
    "helpful",
    "that works",
    "works for me",
    "great",
    "perfect",
    "makes sense",
    "clear",
    "solved",
]

NEGATIVE_REACTION_MARKERS = [
    "confused",
    "not helpful",
    "wrong",
    "incorrect",
    "doesn't make sense",
    "does not make sense",
    "bad",
    "issue",
    "problem",
    "you missed",
    "that is not right",
]


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

        if any(
            word in content
            for word in [
                "price",
                "weather",
                "news",
                "today",
                "live",
                "latest",
                "recent",
                "yesterday",
                "this week",
                "breaking",
                "score",
                "match",
                "fixture",
                "predict",
                "prediction",
                "winner",
                "standings",
                "injury",
            ]
        ):
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


class InteractionMemory:
    """Persist structured learning records for each assistant turn."""

    def __init__(self, file_path: str = "interaction_memory.json"):
        self.file_path = file_path
        self._lock = threading.Lock()
        self.data: Dict[str, Any] = {}
        self._load()

    def _fresh_store(self) -> Dict[str, Any]:
        return {
            "entries": [],
            "pending_by_thread": {},
            "patterns": {
                "user_preference_patterns": {},
                "successful_reasoning_strategies": {},
                "mistakes_to_avoid": {},
            },
        }

    def _load(self) -> None:
        if not os.path.exists(self.file_path):
            self.data = self._fresh_store()
            self._save()
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = self._fresh_store()

        data.setdefault("entries", [])
        data.setdefault("pending_by_thread", {})
        data.setdefault("patterns", {})
        patterns = data["patterns"]
        patterns.setdefault("user_preference_patterns", {})
        patterns.setdefault("successful_reasoning_strategies", {})
        patterns.setdefault("mistakes_to_avoid", {})

        self.data = data
        self._save()

    def _save(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, sort_keys=True)

    def _truncate(self, text: str, limit: int = 180) -> str:
        compact = " ".join((text or "").split())
        if len(compact) <= limit:
            return compact
        return f"{compact[: limit - 3]}..."

    def _quality_from_reward(self, reward: float) -> str:
        if reward > 0:
            return "good"
        if reward < 0:
            return "bad"
        return "neutral"

    def infer_reward_from_reaction(self, reaction_text: str) -> int:
        content = (reaction_text or "").strip().lower()
        if not content:
            return 0

        positive = any(marker in content for marker in POSITIVE_REACTION_MARKERS)
        negative = any(marker in content for marker in NEGATIVE_REACTION_MARKERS)

        if positive and not negative:
            return 1
        if negative and not positive:
            return -1
        return 0

    def _find_entry_index(self, entry_id: str) -> int | None:
        for index, entry in enumerate(self.data["entries"]):
            if entry["id"] == entry_id:
                return index
        return None

    def _bump_pattern(self, bucket: str, pattern: str) -> None:
        if not pattern:
            return
        patterns = self.data["patterns"][bucket]
        patterns[pattern] = patterns.get(pattern, 0) + 1

    def _finalize_existing_pending(self, thread_id: str) -> None:
        pending_id = self.data["pending_by_thread"].get(thread_id)
        if not pending_id:
            return

        index = self._find_entry_index(pending_id)
        if index is None:
            self.data["pending_by_thread"].pop(thread_id, None)
            return

        entry = self.data["entries"][index]
        if entry["agent_response_quality"] == "neutral":
            entry["lessons_learned"] = (
                f"Outcome stayed unclear for the '{entry['rl_action']}' strategy on this turn."
            )
            entry["future_adjustment"] = (
                "Keep the response adaptable and wait for stronger user signals before reinforcing it."
            )

        self.data["pending_by_thread"].pop(thread_id, None)

    def _build_lessons(self, entry: Dict[str, Any], reward: float) -> tuple[str, str, str]:
        action = entry["rl_action"]
        state = entry["rl_state"]
        reaction = self._truncate(entry.get("user_reaction", ""), limit=90)
        reaction_suffix = f" User feedback: {reaction}" if reaction else ""
        if reward > 0:
            preference = f"Users respond well to '{action}' answers for {state} requests."
            lesson = (
                f"The '{action}' strategy was useful and matched the user's needs in this interaction."
                f"{reaction_suffix}"
            )
            adjustment = f"Keep favoring '{action}' when similar {state} prompts appear."
            return preference, lesson, adjustment
        if reward < 0:
            preference = f"Avoid relying on '{action}' answers for {state} requests without clearer grounding."
            lesson = (
                f"The '{action}' strategy created friction or failed to resolve the user's need."
                f"{reaction_suffix}"
            )
            adjustment = (
                f"Use a different response style the next time a {state} prompt appears."
                + (f" Address this issue: {reaction}" if reaction else "")
            )
            return preference, lesson, adjustment

        preference = f"Preference is still unclear for '{action}' answers on {state} requests."
        lesson = f"The user reaction was unclear, so the value of '{action}' remains unconfirmed.{reaction_suffix}"
        adjustment = "Keep monitoring this pattern before reinforcing or avoiding it."
        return preference, lesson, adjustment

    def start_interaction(
        self,
        thread_id: str,
        user_message: str,
        assistant_response: str,
        rl_state: str,
        rl_action: str,
    ) -> Dict[str, Any]:
        with self._lock:
            self._finalize_existing_pending(thread_id)

            entry: Dict[str, Any] = {
                "id": str(uuid.uuid4()),
                "thread_id": thread_id,
                "conversation_summary": self._truncate(
                    f"User asked: {user_message} Assistant answered: {assistant_response}",
                    limit=240,
                ),
                "user_intent": self._truncate(user_message, limit=140),
                "agent_response_quality": "neutral",
                "reward": 0,
                "lessons_learned": (
                    f"Initial record created for the '{rl_action}' strategy on a {rl_state} turn."
                ),
                "future_adjustment": "Wait for explicit or inferred user reaction before reinforcing this turn.",
                "rl_state": rl_state,
                "rl_action": rl_action,
                "assistant_response": self._truncate(assistant_response, limit=180),
                "user_reaction": "",
            }
            self.data["entries"].append(entry)
            self.data["pending_by_thread"][thread_id] = entry["id"]
            self._save()
            return json.loads(json.dumps(entry))

    def apply_feedback(
        self,
        feedback_text: str | None = None,
        explicit_reward: float | None = None,
        entry_id: str | None = None,
        thread_id: str | None = None,
    ) -> Dict[str, Any] | None:
        with self._lock:
            resolved_entry_id = entry_id
            if not resolved_entry_id and thread_id:
                resolved_entry_id = self.data["pending_by_thread"].get(thread_id)
            if not resolved_entry_id:
                return None

            index = self._find_entry_index(resolved_entry_id)
            if index is None:
                return None

            reward = explicit_reward
            if reward is None:
                reward = self.infer_reward_from_reaction(feedback_text or "")

            entry = self.data["entries"][index]
            entry["reward"] = int(reward)
            entry["agent_response_quality"] = self._quality_from_reward(reward)
            entry["user_reaction"] = self._truncate(feedback_text or "", limit=120)

            preference, lesson, adjustment = self._build_lessons(entry, reward)
            entry["lessons_learned"] = lesson
            entry["future_adjustment"] = adjustment

            self._bump_pattern("user_preference_patterns", preference)
            if reward > 0:
                self._bump_pattern("successful_reasoning_strategies", lesson)
            elif reward < 0:
                self._bump_pattern("mistakes_to_avoid", lesson)

            if thread_id:
                self.data["pending_by_thread"].pop(thread_id, None)
            elif entry.get("thread_id"):
                self.data["pending_by_thread"].pop(entry["thread_id"], None)

            self._save()
            return json.loads(json.dumps(entry))

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self.data))
