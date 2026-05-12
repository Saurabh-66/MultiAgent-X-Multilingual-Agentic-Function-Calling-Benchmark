#!/usr/bin/env python3
"""
Start vLLM server for Qwen3.6-35B-A3B-FP8 on saxa (H200, 16GB/71GB) or crannog (A40, 46GB).
Aftering running this, wait for 'Application startup complete',
then run expand_multilingual.py pointing to the correct port.
"""
import subprocess
import sys

cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", "/home/s2892267/models/Qwen3.6-35B-A3B-FP8",
    "--served-model-name", "qwen3.6-35b",
    "--port", "8102",
    "--gpu-memory-utilization", "0.90",
    "--max-model-len", "8192",
    "--dtype", "float16",
    "--trust-remote-code",
    "--max-num-seqs", "4",
]

print("Starting vLLM server for Qwen3.6-35B-A3B-FP8...")
print("Command:", " ".join(cmd))
print("\nWait for: 'Application startup complete' before running expansion script\n")

subprocess.run(cmd)
