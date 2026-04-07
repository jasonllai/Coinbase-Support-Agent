from pathlib import Path

from app.eval.runner import run_all

if __name__ == "__main__":
    run_all(Path(__file__).resolve().parents[2] / "data" / "eval")
