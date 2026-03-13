"""
run_validation.py — Project-level launcher for the validation pipeline.

Usage (from the project root):
    python3 run_validation.py
"""
import subprocess
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
report = os.path.join(HERE, "validation", "report.py")

result = subprocess.run([sys.executable, report], cwd=os.path.join(HERE, "validation"))
sys.exit(result.returncode)
