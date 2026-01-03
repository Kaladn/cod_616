<#
.SYNOPSIS
    Chain Reaction Smoke Test Runner for Windows

.DESCRIPTION
    Runs the comprehensive smoke test suite for CompuCog.
    Tests each module in isolation, then validates the complete chain.

.PARAMETER Quick
    Run quick subset of tests

.PARAMETER ChainOnly
    Only run chain reaction tests

.PARAMETER Failures
    Only run failure injection tests

.PARAMETER Verbose
    Show verbose output

.PARAMETER NoStop
    Don't stop on first failure

.EXAMPLE
    .\run_smoke_tests.ps1
    
.EXAMPLE
    .\run_smoke_tests.ps1 -Quick

.EXAMPLE
    .\run_smoke_tests.ps1 -ChainOnly -Verbose
#>

param(
    [switch]$Quick,
    [switch]$ChainOnly,
    [switch]$Failures,
    [switch]$Verbose,
    [switch]$NoStop
)

$ErrorActionPreference = "Continue"
$script:StartTime = Get-Date

# Colors
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) { Write-Output $args }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Banner($text) {
    Write-Host ""
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host $text -ForegroundColor Cyan -NoNewline
    Write-Host ""
    Write-Host ("=" * 80) -ForegroundColor Cyan
}

function Write-Pass($text) { Write-Host $text -ForegroundColor Green }
function Write-Fail($text) { Write-Host $text -ForegroundColor Red }
function Write-Info($text) { Write-Host $text -ForegroundColor Yellow }

# Navigate to test directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Banner "CHAIN REACTION SMOKE TEST SUITE"
Write-Host "Start Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Test Directory: $ScriptDir"

# Build pytest arguments
$args_list = @()

if ($Quick) {
    $args_list += "--quick"
}
if ($ChainOnly) {
    $args_list += "--chain-only"
}
if ($Failures) {
    $args_list += "--failures"
}
if ($Verbose) {
    $args_list += "--verbose"
}
if ($NoStop) {
    $args_list += "--no-stop"
}

# Run Python test runner
$pythonCmd = "python"
$runnerScript = Join-Path $ScriptDir "run_smoke_tests.py"

Write-Host "`nRunning: $pythonCmd $runnerScript $($args_list -join ' ')"
Write-Host ""

& $pythonCmd $runnerScript @args_list
$exitCode = $LASTEXITCODE

# Summary
$Duration = (Get-Date) - $script:StartTime
Write-Host ""
Write-Host "Total Duration: $($Duration.TotalSeconds.ToString('F1')) seconds"

if ($exitCode -eq 0) {
    Write-Pass "`n ALL TESTS PASSED"
} else {
    Write-Fail "`n TESTS FAILED"
}

exit $exitCode
