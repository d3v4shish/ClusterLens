from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--check-launch"]:
        print("ClusterLens launcher is ready.")
        return 0
    if args[:1] == ["--worker"]:
        from apps.pyqt_production.worker import main as worker_main

        return int(worker_main(args[1:]))

    new_instance = "--new-instance" in args
    if new_instance:
        args.remove("--new-instance")

    from apps.pyqt_production.app import run

    return int(run(new_instance=new_instance))


if __name__ == "__main__":
    raise SystemExit(main())
