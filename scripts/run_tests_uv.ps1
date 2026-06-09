$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

$cacheRoot = Join-Path $repoRoot ".runtime"
$uvCache = Join-Path $cacheRoot "uv"
$tmp = Join-Path $cacheRoot "tmp"
New-Item -ItemType Directory -Force -Path $uvCache | Out-Null
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

$env:UV_CACHE_DIR = $uvCache
$env:TEMP = $tmp
$env:TMP = $tmp

Push-Location $repoRoot
try {
  uv --no-cache run --with pytest python -m pytest tests
} finally {
  Pop-Location
}
