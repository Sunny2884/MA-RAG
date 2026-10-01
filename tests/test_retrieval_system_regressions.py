import ast
import unittest
from pathlib import Path


SOURCE_PATH = Path(__file__).parents[1] / "microservice" / "RetrievalSystem.py"


class RetrievalSystemRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_bm25_uses_the_project_index_path(self):
        self.assertNotIn("/cache/corpus/", self.source)
        self.assertIn(
            "LuceneSearcher(str(self.index_dir.absolute()))",
            self.source,
        )

    def test_medcpt_reranker_is_moved_to_the_selected_device(self):
        expected = (
            "AutoModelForSequenceClassification.from_pretrained("
            "reranker_paths[self.reranker_name]).to(device).eval()"
        )
        self.assertIn(expected, self.source)

    def test_uvicorn_runs_the_initialized_app_object(self):
        uvicorn_calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "uvicorn"
            and node.func.attr == "run"
        ]
        self.assertEqual(len(uvicorn_calls), 1)
        first_argument = uvicorn_calls[0].args[0]
        self.assertIsInstance(first_argument, ast.Name)
        self.assertEqual(first_argument.id, "app")


if __name__ == "__main__":
    unittest.main()
