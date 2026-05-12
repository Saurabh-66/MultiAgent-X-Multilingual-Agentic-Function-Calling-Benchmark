#!/usr/bin/env python3
"""
Start vLLM server for Qwen3.6-27B-FP8 on saxa (H200, 16GB/71GB) or crannog (A40, 46GB).
Run this first, wait for 'Application startup complete', then run generate_seeds.py
"""
import subprocess
import sys

cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", "/home/s2892267/models/Qwen3.6-27B-FP8",
    "--served-model-name", "qwen3.6-27b",
    "--port", "8101",
    "--gpu-memory-utilization", "0.90",
    "--max-model-len", "16384",
    "--dtype", "float16",
    "--trust-remote-code",
    "--max-num-seqs", "8",
]

print("Starting vLLM server for Qwen3.6-27B-FP8...")
print("Command:", " ".join(cmd))
print("\nWait for: 'Application startup complete' before running seeds script\n")

subprocess.run(cmd)
