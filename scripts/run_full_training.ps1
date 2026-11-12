<#
.SYNOPSIS
    Runs full CPU production training for agricultural ML models in sequence.

.DESCRIPTION
    Runs in order:
      1. Fertilizer Recommendation (~2-4 minutes)
      2. Crop Yield Prediction (~50-75 minutes, with stage caching in reports/yield_prediction/cache/)
      3. (Optional) Crop Recommendation (~13-15 minutes, only if -IncludeCrop is specified)

    Uses 'python -u' for unbuffered stdout, logs real-time output via Tee-Object to
    reports/<model>/console.log, stops on the first error, and outputs formatted start/end timestamps.

.PARAMETER IncludeCrop
    Switch parameter to include full re-training of the Crop Recommendation model.
#>

[CmdletBinding()]
param(
    [switch]$IncludeCrop
)

$ErrorActionPreference = "Stop"

function Run-ModelTraining {
    param(
        [string]$ModelName,
        [string]$ScriptPath,
        [string[]]$ScriptArgs,
        [string]$OutputDir,
        [string]$ArtifactsDir
    )

    $startTime = Get-Date
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host " [START] Training Model: $ModelName" -ForegroundColor Cyan
    Write-Host " Timestamp : $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Cyan
    Write-Host " Script    : $ScriptPath" -ForegroundColor Cyan
    Write-Host " Output Dir: $OutputDir" -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan

    # Ensure output and artifact directories exist before process and Tee-Object start
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    New-Item -ItemType Directory -Force -Path $ArtifactsDir | Out-Null
    $consoleLogFile = Join-Path $OutputDir "console.log"

    $cmdArgs = @("-u", $ScriptPath) + $ScriptArgs
    Write-Host "Executing: python $($cmdArgs -join ' ')" -ForegroundColor DarkGray

    # Run unbuffered Python process and pipe console output to console.log (train.log is managed exclusively by Python's logger).
    # Temporarily relax ErrorActionPreference so that ErrorRecord objects produced by PowerShell's 2>&1 merge
    # (e.g. FutureWarning lines written to stderr by sklearn) do not trigger a Stop before ForEach-Object
    # can coerce them to plain strings. Exit code is checked manually immediately after.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & python @cmdArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $consoleLogFile
    $cmdExitCode = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP

    if ($cmdExitCode -ne 0) {
        $failTime = Get-Date
        Write-Host ""
        Write-Host " [FAILED] Model $ModelName exited with error code $LASTEXITCODE at $($failTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Red
        throw "Training failed for $ModelName with exit code $cmdExitCode. Halting execution."
    }

    $endTime = Get-Date
    $duration = $endTime - $startTime
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Green
    Write-Host " [SUCCESS] Model $ModelName Completed Successfully!" -ForegroundColor Green
    Write-Host " Start Time   : $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Green
    Write-Host " End Time     : $($endTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Green
    Write-Host " Duration     : $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
    Write-Host " Console Log  : $consoleLogFile" -ForegroundColor Green
    Write-Host " Training Log : $(Join-Path $OutputDir 'train.log')" -ForegroundColor Green
    Write-Host "================================================================================" -ForegroundColor Green
}

$suiteStartTime = Get-Date
Write-Host "################################################################################" -ForegroundColor Yellow
Write-Host " STARTING AGRICULTURAL ML SUITE FULL TRAINING" -ForegroundColor Yellow
Write-Host " Suite Start Time: $($suiteStartTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Yellow
Write-Host " Include Crop    : $(if ($IncludeCrop) {'YES'} else {'NO (Already current)'})" -ForegroundColor Yellow
Write-Host "################################################################################" -ForegroundColor Yellow

# 1. Fertilizer Recommendation (~2-4 mins)
Run-ModelTraining `
    -ModelName "Fertilizer Recommendation (Model 4)" `
    -ScriptPath "ml/fertilizer_recommendation/train.py" `
    -ScriptArgs @(
        "--data-dir", "data/fertilizer_recommendation",
        "--output-dir", "reports/fertilizer_recommendation",
        "--artifacts-dir", "artifacts/fertilizer_recommendation",
        "--seed", "42",
        "--n-trials", "20"
    ) `
    -OutputDir "reports/fertilizer_recommendation" `
    -ArtifactsDir "artifacts/fertilizer_recommendation"

# 2. Crop Yield Prediction (~50-75 mins on CPU, fully resumable via stage cache)
Run-ModelTraining `
    -ModelName "Crop Yield Prediction (Model 3)" `
    -ScriptPath "ml/yield_prediction/train.py" `
    -ScriptArgs @(
        "--data-dir", "data/yield_prediction",
        "--output-dir", "reports/yield_prediction",
        "--artifacts-dir", "artifacts/yield_prediction",
        "--seed", "42",
        "--export-variant", "B",
        "--n-trials", "20"
    ) `
    -OutputDir "reports/yield_prediction" `
    -ArtifactsDir "artifacts/yield_prediction"

# 3. Crop Recommendation (Optional, ~13-15 mins)
if ($IncludeCrop) {
    Run-ModelTraining `
        -ModelName "Crop Recommendation (Model 1)" `
        -ScriptPath "ml/crop_recommendation/train.py" `
        -ScriptArgs @(
            "--data-dir", "data/crop_recommendation",
            "--output-dir", "reports/crop_recommendation",
            "--artifacts-dir", "artifacts/crop_recommendation",
            "--seed", "42"
        ) `
        -OutputDir "reports/crop_recommendation" `
        -ArtifactsDir "artifacts/crop_recommendation"
}

$suiteEndTime = Get-Date
$suiteDuration = $suiteEndTime - $suiteStartTime
Write-Host ""
Write-Host "################################################################################" -ForegroundColor Yellow
Write-Host " ALL REQUESTED MODELS SUCCESSFULLY TRAINED AND EXPORTED!" -ForegroundColor Yellow
Write-Host " Total Suite Duration: $($suiteDuration.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
Write-Host " Finished At         : $($suiteEndTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Yellow
Write-Host "################################################################################" -ForegroundColor Yellow
