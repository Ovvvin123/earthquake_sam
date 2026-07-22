"""Project-root entry point for the maintained evaluator."""

from pathlib import Path
import runpy
import sys


CODE_DIR = Path(__file__).resolve().parent / "SAM_code_extracted" / "SAM code"
sys.path.insert(0, str(CODE_DIR))
runpy.run_path(str(CODE_DIR / "evaluate.py"), run_name="__main__")
