import sys

from sku_clf.experiment import main as train_main
from sku_clf.infer import main as infer_main


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"infer", "inference", "predict"}:
        sys.argv.pop(1)
        infer_main()
        return
    train_main()
