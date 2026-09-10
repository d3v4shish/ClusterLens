# Benchmarks

No performance benchmark was run for the current UI/model-discovery change. Do not infer a performance improvement from this change.

For a release benchmark, use a fixed local photo fixture and record the fixture manifest, hardware, runtime mode, cold/warm cache state, command, wall time, peak RSS/VRAM, first visible gallery paint, and cache growth:

```bash
.venv/bin/python -m apps.pyqt_production.release_gates \
  --folder /absolute/path/to/fixed/photo-fixture \
  --report-dir benchmarks/release_gates
```

Run the same fixture at least twice for cold and warm-cache measurements. Never compare results from different fixtures or unseeded/randomized workloads.
