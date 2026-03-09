import os
import unittest
from unittest.mock import patch
from langchain_core.messages import HumanMessage, ToolMessage
from rl_policy import InteractionMemory, ReinforcementPolicy


class TestAgentImport(unittest.TestCase):
    """Test that the agent module can be imported and the graph is built correctly."""

    @patch.dict(os.environ, {
        "TAVILY_API_KEY": "test_tavily_key",
        "GOOGLE_API_KEY": "test_google_key",
    })
    def test_agent_graph_construction(self):
        """Test that the agent graph compiles without errors."""
        # Import inside the test so env vars are set first
        import importlib
        import agent as agent_module
        importlib.reload(agent_module)

        # The compiled app should exist
        self.assertIsNotNone(agent_module.app)

    @patch.dict(os.environ, {
        "TAVILY_API_KEY": "test_tavily_key",
        "GOOGLE_API_KEY": "test_google_key",
    })
    def test_agent_graph_has_expected_nodes(self):
        """Test that the agent graph contains the expected nodes."""
        import importlib
        import agent as agent_module
        importlib.reload(agent_module)

        graph = agent_module.app.get_graph()
        node_names = [node.name for node in graph.nodes.values()]
        self.assertIn("agent", node_names)
        self.assertIn("tools", node_names)

    @patch.dict(os.environ, {
        "TAVILY_API_KEY": "test_tavily_key",
        "GOOGLE_API_KEY": "test_google_key",
    })
    def test_search_tool_configured(self):
        """Test that the search tool is properly configured."""
        import importlib
        import agent as agent_module
        importlib.reload(agent_module)

        self.assertIsNotNone(agent_module.search_tool)
        self.assertEqual(len(agent_module.tools), 1)
        assert agent_module.search_tool is not None
        self.assertEqual(agent_module.search_tool.name, "tavily_search")

    @patch.dict(os.environ, {
        "TAVILY_API_KEY": "test_tavily_key",
        "GOOGLE_API_KEY": "test_google_key",
    })
    def test_system_prompt_includes_date_rules(self):
        import importlib
        import agent as agent_module
        importlib.reload(agent_module)

        prompt = agent_module._build_system_instruction().content
        self.assertIn("The current UTC date is", prompt)
        self.assertIn("treat it as the past, not the future", prompt)
        self.assertIn("recent news", prompt)
        self.assertIn("predict a future match", prompt)

    @patch.dict(os.environ, {
        "TAVILY_API_KEY": "test_tavily_key",
        "GOOGLE_API_KEY": "test_google_key",
    })
    def test_response_mode_detects_live_search_prediction(self):
        import importlib
        import agent as agent_module
        importlib.reload(agent_module)

        used_search, response_mode = agent_module._response_mode_for_turn(
            [
                HumanMessage(content="Predict Arsenal vs Chelsea tomorrow"),
                ToolMessage(content="search results", tool_call_id="tool-1"),
            ]
        )

        self.assertTrue(used_search)
        self.assertEqual(response_mode, "live_search_prediction")


class TestReinforcementPolicy(unittest.TestCase):
    """Test lightweight RL policy behavior."""

    def setUp(self):
        self.policy_file = "test_rl_policy.json"
        if os.path.exists(self.policy_file):
            os.remove(self.policy_file)
        self.policy = ReinforcementPolicy(file_path=self.policy_file, epsilon=0.0)

    def tearDown(self):
        if os.path.exists(self.policy_file):
            os.remove(self.policy_file)

    def test_policy_classification(self):
        self.assertEqual(self.policy.classify_state("What is the weather today?"), "realtime")
        self.assertEqual(self.policy.classify_state("Compare Rust vs Go"), "comparison")
        self.assertEqual(self.policy.classify_state("Explain how transformers work"), "explainer")
        self.assertEqual(self.policy.classify_state("Predict the score for Arsenal vs Chelsea tomorrow"), "realtime")
        self.assertEqual(self.policy.classify_state("Give me the latest football news"), "realtime")

    def test_policy_update_increases_q_value(self):
        before = self.policy.snapshot()["general"]["detailed"]
        after = self.policy.update("general", "detailed", reward=1.0, next_state="general")
        self.assertGreater(after, before)


class TestInteractionMemory(unittest.TestCase):
    """Test structured post-conversation learning records."""

    def setUp(self):
        self.memory_file = "test_interaction_memory.json"
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)
        self.memory = InteractionMemory(file_path=self.memory_file)

    def tearDown(self):
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)

    def test_start_interaction_creates_neutral_record(self):
        entry = self.memory.start_interaction(
            thread_id="thread-1",
            user_message="Explain the DP transition",
            assistant_response="Here is the transition in three steps.",
            rl_state="explainer",
            rl_action="step_by_step",
        )

        self.assertEqual(entry["agent_response_quality"], "neutral")
        self.assertEqual(entry["reward"], 0)

        snapshot = self.memory.snapshot()
        self.assertEqual(snapshot["pending_by_thread"]["thread-1"], entry["id"])
        self.assertEqual(snapshot["entries"][0]["user_intent"], "Explain the DP transition")

    def test_apply_feedback_updates_reward_and_patterns(self):
        entry = self.memory.start_interaction(
            thread_id="thread-2",
            user_message="Summarize the design doc",
            assistant_response="Here is the summary.",
            rl_state="summary",
            rl_action="concise",
        )

        updated = self.memory.apply_feedback(
            feedback_text="Thanks, that was helpful.",
            entry_id=entry["id"],
            thread_id="thread-2",
        )

        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated["reward"], 1)
        self.assertEqual(updated["agent_response_quality"], "good")

        snapshot = self.memory.snapshot()
        self.assertFalse(snapshot["pending_by_thread"])
        self.assertTrue(snapshot["patterns"]["successful_reasoning_strategies"])

    def test_negative_feedback_note_is_persisted(self):
        entry = self.memory.start_interaction(
            thread_id="thread-3",
            user_message="Who will win tomorrow?",
            assistant_response="Team A should win.",
            rl_state="realtime",
            rl_action="detailed",
        )

        updated = self.memory.apply_feedback(
            feedback_text="Negative feedback: used outdated team news.",
            explicit_reward=-1,
            entry_id=entry["id"],
            thread_id="thread-3",
        )

        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertIn("outdated team news", updated["user_reaction"])
        self.assertIn("outdated team news", updated["future_adjustment"])

    def test_infer_reward_detects_negative_reaction(self):
        self.assertEqual(self.memory.infer_reward_from_reaction("This is wrong and confusing"), -1)


if __name__ == "__main__":
    unittest.main()
