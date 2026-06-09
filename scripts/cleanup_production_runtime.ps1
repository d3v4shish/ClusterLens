param(
    [string]$RuntimeRoot = "",
    [switch]$Yes,
    [switch]$Logs,
    [switch]$Crash,
    [switch]$Support,
    [switch]$Benchmarks,
    [switch]$ModelAssets,
    [switch]$AllUserData,
    [switch]$IUnderstandThisDeletesUserData,
    [switch]$ForceRuntimeRoot,

    # Legacy compatibility with the previous wrapper.
    [switch]$RemoveAll,
    [switch]$PreserveModels,
    [switch]$WhatIfOnly
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$cleanupScript = Join-Path $scriptRoot "cleanup_production_runtime.py"

$cleanupArgs = @($cleanupScript)
if ($RuntimeRoot) {
    $cleanupArgs += @("--runtime-root", $RuntimeRoot)
}
if ($ForceRuntimeRoot) {
    $cleanupArgs += "--force-runtime-root"
}
if ($AllUserData) {
    $cleanupArgs += "--all-user-data"
}
if ($IUnderstandThisDeletesUserData) {
    $cleanupArgs += "--i-understand-this-deletes-user-data"
}
if ($Logs) {
    $cleanupArgs += "--logs"
}
if ($Crash) {
    $cleanupArgs += "--crash"
}
if ($Support) {
    $cleanupArgs += "--support"
}
if ($Benchmarks) {
    $cleanupArgs += "--benchmarks"
}
if ($ModelAssets) {
    $cleanupArgs += "--model-assets"
}

if ($RemoveAll) {
    $cleanupArgs += @("--logs", "--crash", "--support", "--benchmarks")
    if (-not $PreserveModels) {
        $cleanupArgs += "--model-assets"
    }
}

if (($Yes -or $RemoveAll) -and -not $WhatIfOnly) {
    $cleanupArgs += "--yes"
}

$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
    & $python.Source @cleanupArgs
} else {
    & py -3 @cleanupArgs
}
exit $LASTEXITCODE
