from pathlib import Path


def append_log(run_dir: Path, message: str) -> None:
    path = run_dir / "train.log"
    with path.open("a") as f:
        f.write(message.rstrip() + "\n")
        f.flush()
