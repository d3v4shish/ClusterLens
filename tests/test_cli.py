from __future__ import annotations

import json
from pathlib import Path

from apps import cli


def _seed_data_home(root: Path) -> None:
    (root / "cache").mkdir(parents=True)
    (root / "cache" / "preview.bin").write_bytes(b"generated-preview")


def test_cli_storage_inventory_is_read_only_and_machine_readable(tmp_path: Path, capsys) -> None:
    runtime = tmp_path / "runtime"
    _seed_data_home(runtime)

    assert cli.main(["--data-home", str(runtime), "--json", "storage", "inventory"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["data_home"] == str(runtime.absolute())
    assert payload["files"] == 1
    assert payload["categories"]["cache"] == len(b"generated-preview")
    assert (runtime / "cache" / "preview.bin").read_bytes() == b"generated-preview"


def test_cli_storage_backup_and_verify_are_explicit_managed_data_actions(tmp_path: Path, capsys) -> None:
    runtime = tmp_path / "runtime"
    backups = tmp_path / "backups"
    _seed_data_home(runtime)

    assert cli.main([
        "--data-home", str(runtime), "--json", "storage", "backup", "--destination", str(backups),
    ]) == 0
    backup = json.loads(capsys.readouterr().out)
    assert Path(backup["backup_root"]).is_dir()

    assert cli.main([
        "--data-home", str(runtime), "--json", "storage", "verify", "--backup", backup["backup_root"],
    ]) == 0
    assert json.loads(capsys.readouterr().out) == {"failures": [], "valid": True}
