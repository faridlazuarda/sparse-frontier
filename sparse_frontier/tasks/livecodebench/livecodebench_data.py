# Data loader for the LiveCodeBench code-generation benchmark.
# Dataset: https://huggingface.co/datasets/lighteval/code_generation_lite
# Paper:   https://arxiv.org/abs/2403.07974
# The encoding scheme for private_test_cases (base64 -> zlib -> pickle -> JSON)
# mirrors how lighteval stores them in the HuggingFace dataset.

import base64
import json
import pickle
import zlib
from typing import List, Dict

from datasets import load_dataset

AVAILABLE_VERSIONS = [
    "release_v1",
    "release_v2",
    "release_v3",
    "release_v4",
    "release_v5",
    "release_v6",
    "release_latest",
]


def _decode_private_test_cases(encoded: str) -> list:
    """Decode private_test_cases: base64 -> zlib -> pickle -> JSON string."""
    raw = base64.b64decode(encoded)
    decompressed = zlib.decompress(raw)
    json_str = pickle.loads(decompressed)
    return json.loads(json_str)


def get_dataset(version: str = "release_v6", max_samples: int | None = None) -> List[Dict]:
    if version not in AVAILABLE_VERSIONS:
        raise ValueError(f"Unknown version: {version}. Available: {AVAILABLE_VERSIONS}")

    ds = load_dataset("lighteval/code_generation_lite", version, split="test")

    samples = []
    for row in ds:
        samples.append({
            "question_content": row["question_content"],
            "starter_code": row["starter_code"],
            "public_test_cases": json.loads(row["public_test_cases"]),
            "private_test_cases": _decode_private_test_cases(row["private_test_cases"]),
            "difficulty": row["difficulty"],
            "platform": row["platform"],
            "question_id": row["question_id"],
            "metadata": json.loads(row["metadata"]),
        })
        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples
