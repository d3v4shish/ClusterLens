param(
    [string]$OutputDir = "dist\production",
    [switch]$SkipReleaseGates,
    [switch]$Unsigned
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Manifest = Join-Path $RepoRoot "packaging\production_release_manifest.json"

if (-not (Test-Path $Manifest)) {
    throw "Missing production release manifest: $Manifest"
}

$manifestJson = Get-Content $Manifest -Raw | ConvertFrom-Json
$installer = $manifestJson.installer
$requiredPackagingFiles = @(
    $installer.variant_build_script,
    $installer.pyinstaller_spec,
    $installer.windows_installer_script,
    $installer.icon_png,
    $installer.icon_ico
)
foreach ($variant in $installer.build_variants) {
    if ($variant.requirements) {
        $requiredPackagingFiles += $variant.requirements
    }
}
foreach ($relativePath in $requiredPackagingFiles) {
    if ([string]::IsNullOrWhiteSpace($relativePath)) {
        continue
    }
    $candidate = Join-Path $RepoRoot $relativePath
    if (-not (Test-Path $candidate)) {
        throw "Packaging asset missing: $candidate"
    }
}

if (-not $SkipReleaseGates) {
    python -m apps.pyqt_production.release_gates --report-dir (Join-Path $RepoRoot "dist\release_gates")
}

$cert = $env:IMAGE_CLUSTERING_SIGNING_CERT
$timestamp = $env:IMAGE_CLUSTERING_TIMESTAMP_SERVER
if (-not $Unsigned) {
    if (-not $cert) {
        throw "IMAGE_CLUSTERING_SIGNING_CERT is required unless -Unsigned is supplied."
    }
    if (-not $timestamp) {
        throw "IMAGE_CLUSTERING_TIMESTAMP_SERVER is required unless -Unsigned is supplied."
    }
}

Write-Host "Packaging manifest verified: $Manifest"
Write-Host "Output directory: $OutputDir"
Write-Host "Variant build script: $(Join-Path $RepoRoot $installer.variant_build_script)"
Write-Host "PyInstaller spec: $(Join-Path $RepoRoot $installer.pyinstaller_spec)"
Write-Host "Windows installer script: $(Join-Path $RepoRoot $installer.windows_installer_script)"
Write-Host "Icon assets: $(Join-Path $RepoRoot $installer.icon_png) | $(Join-Path $RepoRoot $installer.icon_ico)"
Write-Host "Next step: run scripts\build_pyqt_binary.py -variant cpu or -variant gpu-cu121, then sign the generated executable."
Write-Host "This script intentionally fails early when signing inputs or release gates are missing."
