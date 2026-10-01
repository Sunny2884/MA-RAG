import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).parents[1] / "utils.py"


class InferenceTests(unittest.TestCase):
    def test_reasoning_mode_accepts_models_without_think_tags(self):
        dotenv = types.ModuleType("dotenv")
        dotenv.find_dotenv = lambda: ""
        dotenv.load_dotenv = lambda *_args, **_kwargs: None

        numpy = types.ModuleType("numpy")
        numpy.exp = lambda values: values

        requests = types.ModuleType("requests")

        scipy = types.ModuleType("scipy")
        scipy_stats = types.ModuleType("scipy.stats")
        scipy_stats.entropy = lambda _values: 0.0

        openai = types.ModuleType("openai")
        openai_types = types.ModuleType("openai.types")
        openai_chat = types.ModuleType("openai.types.chat")
        openai_chat_completion = types.ModuleType("openai.types.chat.chat_completion")
        openai_chat_completion.ChatCompletion = object

        microservice = types.ModuleType("microservice")
        microservice.CustomLanguageModel = object

        fake_modules = {
            "dotenv": dotenv,
            "numpy": numpy,
            "requests": requests,
            "scipy": scipy,
            "scipy.stats": scipy_stats,
            "openai": openai,
            "openai.types": openai_types,
            "openai.types.chat": openai_chat,
            "openai.types.chat.chat_completion": openai_chat_completion,
            "microservice": microservice,
        }

        with patch.dict(sys.modules, fake_modules):
            spec = importlib.util.spec_from_file_location("utils_inference_test", MODULE_PATH)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        top_logprob = types.SimpleNamespace(logprob=-0.1)
        token = types.SimpleNamespace(
            token="answer",
            logprob=-0.1,
            top_logprobs=[top_logprob],
        )
        choice = types.SimpleNamespace(
            message=types.SimpleNamespace(content="[Query 1] facial nerve injury"),
            logprobs=types.SimpleNamespace(content=[token]),
        )
        response = types.SimpleNamespace(choices=[choice])

        class FakeModel:
            calls = 0

            def generate(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("inference retried a valid response")
                return response

        result = module.inference(
            "system",
            "prompt",
            FakeModel(),
            enable_thinking=True,
        )

        self.assertEqual(result[0][0], "")
        self.assertEqual(result[0][1], "[Query 1] facial nerve injury")


if __name__ == "__main__":
    unittest.main()
