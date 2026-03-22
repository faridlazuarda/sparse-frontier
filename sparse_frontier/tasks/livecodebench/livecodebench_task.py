# LiveCodeBench code-generation task.
# Task structure follows the existing MathTask pattern in this project.
# Prompt templates sourced from lighteval's LiveCodeBench integration:
#   https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/tasks/lcb/main.py

from typing import List, Dict, Any

from sparse_frontier.tasks.abstract_task import AbstractTask
from sparse_frontier.tasks.abstract_sample import AbstractSample

from sparse_frontier.tasks.livecodebench.livecodebench_data import get_dataset
from sparse_frontier.tasks.livecodebench.livecodebench_utils import extract_code, check_correctness

LIVECODEBENCH_PROMPT_WITH_STARTER = """
You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything besides the program inside markdown code blocks.

Question: {question_content}

You will use the following starter code to write the solution to the problem and enclose your code within delimiters.
```python
{starter_code}
```
""".strip()

LIVECODEBENCH_PROMPT_STDIO = """
You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything besides the program inside markdown code blocks.

Question: {question_content}

Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.
```python
# YOUR CODE HERE
```
""".strip()


class LiveCodeBenchSample(AbstractSample):
    @staticmethod
    def format_sample(question_content: str, starter_code: str) -> str:
        if starter_code:
            return LIVECODEBENCH_PROMPT_WITH_STARTER.format(
                question_content=question_content,
                starter_code=starter_code,
            )
        return LIVECODEBENCH_PROMPT_STDIO.format(
            question_content=question_content,
        )

    def _generate_sample(self):
        dataset = self.task_params["processed_dataset"]
        sample = dataset[self.sample_id]

        question_content = sample["question_content"]
        starter_code = sample["starter_code"]
        gold_answer = ""  # evaluation is execution-based, not answer-matching
        input_text = LiveCodeBenchSample.format_sample(question_content, starter_code)

        extra = {
            "question_id": sample["question_id"],
            "platform": sample["platform"],
            "difficulty": sample["difficulty"],
            "starter_code": starter_code,
            "public_test_cases": sample["public_test_cases"],
            "private_test_cases": sample["private_test_cases"],
            "metadata": sample["metadata"],
        }
        return input_text, gold_answer, extra


class LiveCodeBenchTask(AbstractTask):
    def __init__(self, version: str = "release_v6", **kwargs) -> None:
        super().__init__(**kwargs)
        self._load_and_process_dataset(version)
        self.check_params()

    def _load_and_process_dataset(self, version: str) -> None:
        dataset = get_dataset(version, max_samples=self.num_samples)
        self.task_params["processed_dataset"] = dataset
        self.task_params["version"] = version

    def check_params(self) -> None:
        if not self.task_params.get("processed_dataset"):
            raise ValueError("Dataset not loaded")

    def check_sample_length(self, input_text: str, gold_answer: str) -> None:
        return

    @property
    def sample_class(self):
        return LiveCodeBenchSample

    @staticmethod
    def evaluate(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not predictions:
            return {"pass@1": 0.0}

        passed = 0
        for item in predictions:
            raw_pred = item.get("pred", "")
            pred_str = raw_pred[0] if isinstance(raw_pred, list) else raw_pred
            code = extract_code(pred_str) if isinstance(pred_str, str) else ""

            test_cases = item.get("private_test_cases", [])
            metadata = item.get("metadata", {})

            result = check_correctness(code, test_cases, metadata)
            if result["passed"]:
                passed += 1

        pass_rate = passed / len(predictions)
        return {"pass@1": pass_rate}
