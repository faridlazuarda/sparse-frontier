#!/usr/bin/env python3
"""
Colab test script for both feature branches.

Usage (single command in Colab):
    !wget -qO test.py https://raw.githubusercontent.com/faridlazuarda/sparse-frontier/<branch>/test_branches_colab.py && python test.py

Or after cloning:
    !python test_branches_colab.py

Requires: Colab GPU runtime (T4 is fine).
Set HF_TOKEN env var or pass --hf-token for gated models.
"""

import argparse
import os
import subprocess
import sys
import shutil
from pathlib import Path

REPO_URL = "https://github.com/faridlazuarda/sparse-frontier.git"
WORK_DIR = Path("/content/sparse-frontier-test")

# Small model that fits on T4 (16GB) without gating
TEST_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_OVERRIDES = (
    "model.name=qwen_1_5b "
    "model.path=Qwen/Qwen2.5-1.5B-Instruct "
    "model.hf_repo=Qwen/Qwen2.5-1.5B-Instruct "
    "model.num_q_heads=12 "
    "model.num_kv_heads=2 "
    "model.num_layers=28"
)

COMMON_ARGS = (
    f"{MODEL_OVERRIDES} "
    "samples=1 max_input_tokens=4096 max_output_tokens=512 gpus=1 tp=1"
)


def run(cmd, cwd=None, check=True):
    print(f"\n{'='*60}")
    print(f"$ {cmd}")
    print('='*60)
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        stdout=sys.stdout, stderr=sys.stderr,
    )
    if check and result.returncode != 0:
        print(f"\n✗ FAILED (exit {result.returncode}): {cmd}")
        return False
    return True


def setup_repo(branch):
    """Clone or reset repo and checkout the given branch."""
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    run(f"git clone --branch {branch} --single-branch {REPO_URL} {WORK_DIR}")
    run("pip install --no-cache-dir -e . 2>&1 | tail -5", cwd=WORK_DIR)


def test_vllm_compat():
    """Test feat/update-vllm-compat — the main deliverable."""
    print("\n" + "#"*60)
    print("# BRANCH: feat/update-vllm-compat")
    print("#"*60)

    setup_repo("feat/update-vllm-compat")
    results = {}

    # --- Test 1: Import validation ---
    print("\n--- Test 1: Compat imports ---")
    ok = run(
        'python -c "'
        "from sparse_frontier.modelling.attention.compat import ("
        "    get_vllm_version, flash_attn_varlen_func, get_open_port,"
        "    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,"
        "    tensor_model_parallel_all_gather, get_extra_llm_kwargs,"
        "    configure_vllm_environment,"
        ");"
        "v = get_vllm_version();"
        "print(f'vLLM {v}');"
        "print(f'extra kwargs: {get_extra_llm_kwargs()}');"
        "configure_vllm_environment();"
        "import os;"
        "print(f'VLLM_USE_V1={os.environ.get(chr(86)+chr(76)+chr(76)+chr(77)+chr(95)+chr(85)+chr(83)+chr(69)+chr(95)+chr(86)+chr(49), None)}');"
        "print('OK')"
        '"',
        cwd=WORK_DIR,
    )
    results["1_imports"] = ok

    # --- Test 2: flash_attn_with_kvcache lazy import ---
    print("\n--- Test 2: flash_attn_with_kvcache lazy import resolves ---")
    ok = run(
        'python -c "'
        "from sparse_frontier.modelling.attention.compat import flash_attn_with_kvcache;"
        "print(f'Type: {type(flash_attn_with_kvcache)}');"
        "print('OK — lazy wrapper loaded, will resolve on first CUDA call')"
        '"',
        cwd=WORK_DIR,
    )
    results["2_lazy_import"] = ok

    # --- Test 3: Dense attention end-to-end ---
    print("\n--- Test 3: Dense attention (prep → predict → eval) ---")
    ok = run(
        f"python -m sparse_frontier.main mode=all task=ruler_niah "
        f"attention=dense use_attention_patch=true {COMMON_ARGS}",
        cwd=WORK_DIR,
    )
    results["3_dense_e2e"] = ok

    # --- Test 4: Quest sparse decode (exercises flash_attn_with_kvcache) ---
    print("\n--- Test 4: Quest attention (sparse decode) ---")
    ok = run(
        f"python -m sparse_frontier.main mode=all task=ruler_niah "
        f"attention=quest attention.args.token_budget=2048 "
        f"attention.args.page_size=16 attention.args.share_pages=false "
        f"use_attention_patch=true {COMMON_ARGS}",
        cwd=WORK_DIR,
    )
    results["4_quest_decode"] = ok

    # --- Test 5: SnapKV compression ---
    print("\n--- Test 5: SnapKV (KV compression) ---")
    ok = run(
        f"python -m sparse_frontier.main mode=all task=ruler_niah "
        f"attention=snapkv attention.args.kernel_size=5 "
        f"attention.args.approximation_window=16 attention.args.token_capacity=2048 "
        f"use_attention_patch=true {COMMON_ARGS}",
        cwd=WORK_DIR,
    )
    results["5_snapkv"] = ok

    # --- Test 6: No-patch mode (vanilla vLLM, no attention interception) ---
    print("\n--- Test 6: Vanilla vLLM (patch disabled) ---")
    ok = run(
        f"python -m sparse_frontier.main mode=all task=ruler_niah "
        f"attention=dense use_attention_patch=false {COMMON_ARGS}",
        cwd=WORK_DIR,
    )
    results["6_no_patch"] = ok

    return results


def test_livecodebench():
    """Test feat/livecodebench-eval — LCB task integration."""
    print("\n" + "#"*60)
    print("# BRANCH: feat/livecodebench-eval")
    print("#"*60)

    setup_repo("feat/livecodebench-eval")
    results = {}

    # --- Test 1: Import validation ---
    print("\n--- Test 1: LCB task imports ---")
    ok = run(
        'python -c "'
        "from sparse_frontier.tasks.livecodebench.livecodebench_task import LiveCodeBenchTask;"
        "from sparse_frontier.tasks.registry import TASK_REGISTRY;"
        "assert 'livecodebench' in TASK_REGISTRY;"
        "print('OK')"
        '"',
        cwd=WORK_DIR,
    )
    results["1_lcb_imports"] = ok

    # --- Test 2: Data preparation ---
    print("\n--- Test 2: LCB data preparation ---")
    ok = run(
        f"python -m sparse_frontier.main mode=prep task=livecodebench "
        f"{COMMON_ARGS}",
        cwd=WORK_DIR,
    )
    results["2_lcb_prep"] = ok

    # --- Test 3: Full pipeline ---
    print("\n--- Test 3: LCB full pipeline (dense) ---")
    ok = run(
        f"python -m sparse_frontier.main mode=all task=livecodebench "
        f"attention=dense use_attention_patch=false {COMMON_ARGS}",
        cwd=WORK_DIR,
    )
    results["3_lcb_e2e"] = ok

    return results


def install_deps():
    """Install base dependencies."""
    print("\n--- Installing dependencies ---")
    run("pip install --no-cache-dir torch==2.10.0 2>&1 | tail -3")
    run("pip install --no-cache-dir vllm==0.18.0 2>&1 | tail -3")
    run("pip install --no-cache-dir flash-attn --no-build-isolation 2>&1 | tail -3")


def print_report(all_results):
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    all_pass = True
    for branch, results in all_results.items():
        print(f"\n  {branch}:")
        for name, ok in results.items():
            status = "✓ PASS" if ok else "✗ FAIL"
            print(f"    {name}: {status}")
            if not ok:
                all_pass = False

    print("\n" + "="*60)
    if all_pass:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60)
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Test sparse-frontier branches on Colab")
    parser.add_argument("--hf-token", help="HuggingFace token for gated models")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--branch", choices=["compat", "lcb", "both"], default="both",
                        help="Which branch(es) to test")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        run(f"huggingface-cli login --token {args.hf_token}")

    if not args.skip_deps:
        install_deps()

    all_results = {}

    if args.branch in ("compat", "both"):
        all_results["feat/update-vllm-compat"] = test_vllm_compat()

    if args.branch in ("lcb", "both"):
        all_results["feat/livecodebench-eval"] = test_livecodebench()

    ok = print_report(all_results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
