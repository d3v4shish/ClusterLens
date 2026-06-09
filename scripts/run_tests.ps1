$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "Missing venv python at $python. Create the venv first."
}

& $python -m pytest (Join-Path $PSScriptRoot "..\tests")
