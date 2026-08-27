from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from apps.pyqt_production.__main__ import main as production_main

    return int(production_main(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
