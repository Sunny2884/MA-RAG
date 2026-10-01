import importlib.util
import logging
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).parents[1] / "microservice" / "CustomLanguageModel.py"


class CustomLanguageModelEnvTests(unittest.TestCase):
    def test_loads_dotenv_before_creating_openai_client(self):
        captured = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        dotenv = types.ModuleType("dotenv")

        def load_dotenv():
            os.environ["API_KEY"] = "from-dotenv"
            os.environ["BASE_URL"] = "http://localhost:8000/v1"

        dotenv.load_dotenv = load_dotenv

        numpy = types.ModuleType("numpy")
        numpy.random = types.SimpleNamespace(uniform=lambda *_: 0)

        openai = types.ModuleType("openai")
        openai.OpenAI = FakeOpenAI
        openai_types = types.ModuleType("openai.types")
        openai_chat = types.ModuleType("openai.types.chat")
        openai_chat_completion = types.ModuleType("openai.types.chat.chat_completion")
        openai_chat_completion.ChatCompletion = object

        fake_modules = {
            "dotenv": dotenv,
            "numpy": numpy,
            "openai": openai,
            "openai.types": openai_types,
            "openai.types.chat": openai_chat,
            "openai.types.chat.chat_completion": openai_chat_completion,
        }

        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, fake_modules):
            spec = importlib.util.spec_from_file_location("custom_language_model_test", MODULE_PATH)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.CustomLanguageModel("llama-3.1-8b-instruct", logging.getLogger("test"))

        self.assertEqual(captured["api_key"], "from-dotenv")
        self.assertEqual(captured["base_url"], "http://localhost:8000/v1")

    def test_llama_ignores_qwen_thinking_option(self):
        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return "response"

        class FakeOpenAI:
            def __init__(self, **_kwargs):
                self.chat = types.SimpleNamespace(completions=FakeCompletions())

        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda: None

        numpy = types.ModuleType("numpy")
        numpy.random = types.SimpleNamespace(uniform=lambda *_: 0)

        openai = types.ModuleType("openai")
        openai.OpenAI = FakeOpenAI
        openai_types = types.ModuleType("openai.types")
        openai_chat = types.ModuleType("openai.types.chat")
        openai_chat_completion = types.ModuleType("openai.types.chat.chat_completion")
        openai_chat_completion.ChatCompletion = object

        fake_modules = {
            "dotenv": dotenv,
            "numpy": numpy,
            "openai": openai,
            "openai.types": openai_types,
            "openai.types.chat": openai_chat,
            "openai.types.chat.chat_completion": openai_chat_completion,
        }

        with patch.dict(sys.modules, fake_modules):
            spec = importlib.util.spec_from_file_location("custom_language_model_llama_test", MODULE_PATH)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = module.CustomLanguageModel(
                "llama-3.1-8b-instruct",
                logging.getLogger("test"),
                base_url="http://localhost:8000/v1",
            )
            response = model.generate(
                [{"role": "user", "content": "hello"}],
                enable_thinking=True,
            )

        self.assertEqual(response, "response")
        self.assertEqual(captured["extra_body"], {})


if __name__ == "__main__":
    unittest.main()
