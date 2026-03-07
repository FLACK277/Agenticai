import os
import unittest
from unittest.mock import patch
from rl_policy import ReinforcementPolicy


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
        self.assertEqual(agent_module.search_tool.name, "tavily_search")


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

    def test_policy_update_increases_q_value(self):
        before = self.policy.snapshot()["general"]["detailed"]
        after = self.policy.update("general", "detailed", reward=1.0, next_state="general")
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
