#!/usr/bin/env python3
# Lightweight test: PFC decomposition + global theta memory generation.

import argparse
import json
import os
import sys
from pathlib import Path
import importlib.util


def _load_module(alias: str, path: Path):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {alias} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


class MockLLMClient:
    def __init__(self, question: str):
        self.question = question
        self.call_log = []

    def generate(self, prompt: str, meta=None, temperature: float = 0.2) -> str:
        if "QUESTION SCHEMA ANALYZER" in prompt:
            payload = {
                "question_type": "other",
                "answer_form": "other",
                "focus": "generic",
                "entities": [],
                "constraints": [],
                "notes": "",
            }
            return json.dumps(payload)
        if "\"subquestions\"" in prompt:
            payload = {
                "subquestions": [
                    {
                        "subquestion": self.question,
                        "core_entity": "",
                        "expected_answer_type": "unknown",
                    }
                ]
            }
            return json.dumps(payload)
        return "{}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test PFC decomposition and global theta memory generation."
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Are director of film Move (1970 Film) and director of film Mediterranee (1963 Film) from the same country?",
        help="Main question to decompose.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="2wiki",
        help="Dataset name used in memory file naming.",
    )
    parser.add_argument(
        "--example-index",
        type=int,
        default=0,
        help="Example index used in memory file naming.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a mock LLM instead of calling the real API.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    gamma_path = repo_root / "Theta-Gamma" / "Gamma" / "gamma.py"
    theta_path = repo_root / "Theta-Gamma" / "Theta" / "theta.py"

    if not gamma_path.exists():
        print(f"Missing gamma module at {gamma_path}")
        return 1
    if not theta_path.exists():
        print(f"Missing theta module at {theta_path}")
        return 1

    _load_module("gamma_gpt35", gamma_path)
    theta_mod = _load_module("theta_gpt35", theta_path)

    if args.mock:
        llm_client = MockLLMClient(args.question)
    else:
        if not (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")):
            print("Missing API key. Set OPENROUTER_API_KEY or OPENAI_API_KEY, or run with --mock.")
            return 2
        llm_client = sys.modules["gamma_gpt35"].LLMClient()

    pfc = theta_mod.PFC(dataset_name=args.dataset, llm_client=llm_client)
    subquestions = pfc.plan_subquestions(args.question)

    memory = pfc._global_theta_memory
    if memory is None:
        schema = pfc._ensure_schema(args.question)
        memory = pfc._build_global_theta_memory(args.question, subquestions, schema)

    memory_path = pfc._write_global_theta_memory(
        memory,
        dataset_name=args.dataset,
        example_index=args.example_index,
    )

    print("PFC subquestions:")
    for i, subq in enumerate(subquestions, 1):
        print(f"  {i}. {subq}")
    print(f"Global theta memory path: {memory_path}")
    print(f"Memory file exists: {Path(memory_path).exists()}")
    print(f"Memory subquestion count: {len(memory.get('sub_questions', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
