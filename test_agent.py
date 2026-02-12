import os
import unittest
from unittest.mock import patch


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


if __name__ == "__main__":
    unittest.main()
