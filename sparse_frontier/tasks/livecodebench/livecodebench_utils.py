# Code extraction and sandboxed execution for LiveCodeBench evaluation.
#
# Grading logic (call-based vs stdio, reliability_guard, subprocess isolation)
# adapted from:
#   - lighteval:     https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/tasks/lcb/codegen_metrics.py
#   - LiveCodeBench: https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/compute_code_generation_metrics.py
#
# Default timeout of 6s per test case matches the LiveCodeBench standard.

import builtins
import io
import json
import multiprocessing
import os
import re
import shutil
import signal
import subprocess
import sys


# ---------------------------------------------------------------------------
# 1. Code extraction
# ---------------------------------------------------------------------------

_PYTHON_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_GENERIC_BLOCK_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def extract_code(model_output: str) -> str:
    """Extract code from the last markdown code block in model_output."""
    python_blocks = _PYTHON_BLOCK_RE.findall(model_output)
    if python_blocks:
        return python_blocks[-1].strip()

    generic_blocks = _GENERIC_BLOCK_RE.findall(model_output)
    if generic_blocks:
        return generic_blocks[-1].strip()

    return model_output.strip()


# ---------------------------------------------------------------------------
# 2. Reliability guard
# ---------------------------------------------------------------------------

def reliability_guard():
    """Override destructive operations so sandboxed code cannot cause harm."""

    def _noop(*args, **kwargs):
        return None

    os_disable = [
        "kill", "system", "putenv", "remove", "removedirs", "rmdir",
        "fchdir", "setuid", "fork", "forkpty", "killpg", "rename",
        "renames", "truncate", "replace", "unlink", "fchmod", "fchown",
        "chmod", "chown", "chroot", "lchflags", "lchmod", "lchown",
    ]
    for name in os_disable:
        if hasattr(os, name):
            setattr(os, name, _noop)

    for name in ("rmtree", "move", "chown"):
        if hasattr(shutil, name):
            setattr(shutil, name, _noop)

    subprocess.Popen = _noop  # type: ignore[assignment]

    builtins.help = _noop  # type: ignore[assignment]
    builtins.exit = _noop  # type: ignore[assignment]
    builtins.quit = _noop  # type: ignore[assignment]

    sys.setrecursionlimit(50000)
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(50000)


# ---------------------------------------------------------------------------
# 3. Call-based grading (LeetCode-style)
# ---------------------------------------------------------------------------

def _timeout_handler(signum, frame):
    raise TimeoutError("Timed out")


def _parse_call_inputs(input_str: str) -> list:
    """Parse newline-separated JSON values into a list of function arguments.

    LCB encodes each function argument as a separate JSON value on its own line.
    E.g. '"leetscode"\\n["leet", "code"]' -> ["leetscode", ["leet", "code"]]
    """
    args = []
    for line in input_str.strip().splitlines():
        line = line.strip()
        if line:
            args.append(json.loads(line))
    return args


def grade_call_based(code: str, test_cases: list, func_name: str, timeout: int = 6) -> list:
    """Grade code that defines a Solution class with method func_name."""
    namespace: dict = {}
    try:
        compiled = compile(code, "<submission>", "exec")
        exec(compiled, namespace)
    except Exception:
        return [False] * len(test_cases)

    solution_cls = namespace.get("Solution")
    if solution_cls is None:
        return [False] * len(test_cases)

    try:
        instance = solution_cls()
        fn = getattr(instance, func_name, None)
    except Exception:
        return [False] * len(test_cases)

    if fn is None:
        return [False] * len(test_cases)

    signal.signal(signal.SIGALRM, _timeout_handler)
    results: list[bool] = []
    for tc in test_cases:
        signal.alarm(timeout)
        try:
            args = _parse_call_inputs(tc["input"])
            expected = json.loads(tc["output"])
            actual = fn(*args)
            results.append(actual == expected)
        except TimeoutError:
            results.append(False)
        except Exception:
            results.append(False)
        finally:
            signal.alarm(0)

    return results


# ---------------------------------------------------------------------------
# 4. Stdio-based grading (Codeforces / AtCoder style)
# ---------------------------------------------------------------------------

def grade_stdio(code: str, test_cases: list, timeout: int = 6) -> list:
    """Grade code that reads from stdin and writes to stdout."""
    compiled = compile(code, "<submission>", "exec")

    signal.signal(signal.SIGALRM, _timeout_handler)
    results: list[bool] = []
    for tc in test_cases:
        signal.alarm(timeout)
        try:
            input_str = tc["input"]
            expected_str = tc["output"]

            old_stdin = sys.stdin
            old_stdout = sys.stdout
            sys.stdin = io.StringIO(input_str)
            capture = io.StringIO()
            sys.stdout = capture

            try:
                exec(compiled, {})
            finally:
                sys.stdin = old_stdin
                sys.stdout = old_stdout

            actual_lines = capture.getvalue().strip().splitlines()
            expected_lines = expected_str.strip().splitlines()
            match = (
                len(actual_lines) == len(expected_lines)
                and all(
                    a.strip() == e.strip()
                    for a, e in zip(actual_lines, expected_lines)
                )
            )
            results.append(match)
        except TimeoutError:
            results.append(False)
        except Exception:
            results.append(False)
        finally:
            signal.alarm(0)

    return results


# ---------------------------------------------------------------------------
# 5. Top-level correctness checker (runs grading in a subprocess)
# ---------------------------------------------------------------------------

def _worker(code: str, test_cases: list, metadata: dict, timeout: int, result_list):
    """Target for multiprocessing.Process — writes results into result_list."""
    reliability_guard()
    try:
        func_name = metadata.get("func_name")
        if func_name:
            outcomes = grade_call_based(code, test_cases, func_name, timeout=timeout)
        else:
            outcomes = grade_stdio(code, test_cases, timeout=timeout)
        result_list.extend(outcomes)
    except Exception:
        pass


def check_correctness(
    code: str,
    test_cases: list,
    metadata: dict,
    timeout: int = 6,
) -> dict:
    """Run sandboxed grading in a child process with a global timeout.

    Returns {"passed": bool, "results": list[bool], "error": str | None}
    """
    manager = multiprocessing.Manager()
    result_list = manager.list()

    proc = multiprocessing.Process(
        target=_worker,
        args=(code, test_cases, metadata, timeout, result_list),
    )
    global_timeout = (timeout + 1) * len(test_cases) + 5
    proc.start()
    try:
        proc.join(timeout=global_timeout)

        if proc.is_alive():
            proc.kill()
            proc.join()
            return {"passed": False, "results": [], "error": "global_timeout"}

        outcomes = list(result_list)
        if not outcomes:
            return {"passed": False, "results": [], "error": "no_results"}

        return {
            "passed": all(outcomes),
            "results": outcomes,
            "error": None,
        }
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join()
        manager.shutdown()
