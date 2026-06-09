from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--worker"]:
        from apps.pyqt_production.worker import main as worker_main

        return int(worker_main(args[1:]))

    from apps.pyqt_production.app import run

    return int(run())


if __name__ == "__main__":
    raise SystemExit(main())
