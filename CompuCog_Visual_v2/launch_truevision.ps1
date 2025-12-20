#!/usr/bin/env powershell
<#
.SYNOPSIS
    Launch TrueVision v2.0 with full telemetry pipeline

.DESCRIPTION
    Starts all loggers (input, activity, process, network) then launches TrueVision live capture

.PARAMETER Duration
    Capture duration in minutes (default: 15)

.PARAMETER SkipLoggers
    Skip logger startup (use if already running)

.EXAMPLE
    .\launch_truevision.ps1 -Duration 10
    .\launch_truevision.ps1 -Duration 5 -SkipLoggers
#>

param(
    [int]$Duration = 15,
    [switch]$SkipLoggers
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "="*80
Write-Host "TrueVision v2.0 Launch Sequence"
Write-Host "="*80
Write-Host ""

$TRUEVISION_ROOT = Split-Path -Parent $PSCommandPath

if (-not $SkipLoggers) {
    Write-Host "[1/3] Starting telemetry loggers..." -ForegroundColor Cyan
    
    # Start input logger
    Write-Host "  -> Input logger (keyboard/mouse)..." -NoNewline
    Start-Process python -ArgumentList "$TRUEVISION_ROOT\loggers\input_logger.py" -WindowStyle Hidden -PassThru | Out-Null
    Write-Host " STARTED" -ForegroundColor Green
    
    # Start activity logger
    Write-Host "  -> Activity logger (windows/processes)..." -NoNewline
    Start-Process python -ArgumentList "$TRUEVISION_ROOT\loggers\activity_logger.py" -WindowStyle Hidden -PassThru | Out-Null
    Write-Host " STARTED" -ForegroundColor Green
    
    # Start process logger
    Write-Host "  -> Process logger (system resources)..." -NoNewline
    Start-Process python -ArgumentList "$TRUEVISION_ROOT\loggers\process_logger.py" -WindowStyle Hidden -PassThru | Out-Null
    Write-Host " STARTED" -ForegroundColor Green
    
    # Start network logger
    Write-Host "  -> Network logger (packets/connections)..." -NoNewline
    Start-Process powershell -ArgumentList "-File $TRUEVISION_ROOT\loggers\network_logger.ps1" -WindowStyle Hidden -PassThru | Out-Null
    Write-Host " STARTED" -ForegroundColor Green
    
    # Start gamepad logger
    Write-Host "  -> Gamepad logger (controller input)..." -NoNewline
    Start-Process python -ArgumentList "$TRUEVISION_ROOT\loggers\gamepad_logger.py" -WindowStyle Hidden -PassThru | Out-Null
    Write-Host " STARTED" -ForegroundColor Green
    
    Write-Host "  [OK] All loggers running" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "[2/3] Waiting 3 seconds for logger initialization..." -ForegroundColor Cyan
    Start-Sleep -Seconds 3
} else {
    Write-Host "[1/3] Skipping logger startup (assumed already running)" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "[3/3] Launching TrueVision live capture (${Duration} minutes)..." -ForegroundColor Cyan
Write-Host ""
Write-Host "="*80
Write-Host ""

Set-Location "$TRUEVISION_ROOT\gaming"
python truevision_live_run.py --duration $Duration

Write-Host ""
Write-Host "="*80
Write-Host "TrueVision capture complete!" -ForegroundColor Green
Write-Host "="*80
Write-Host ""
Write-Host "Output files:"
Write-Host "  Visual telemetry: gaming\telemetry\truevision_live_*.jsonl" -ForegroundColor Cyan
Write-Host "  Input logs:       logs\input\input_activity_*.jsonl" -ForegroundColor Cyan
Write-Host "  Activity logs:    logs\activity\user_activity_*.jsonl" -ForegroundColor Cyan
Write-Host "  Process logs:     logs\process\process_activity_*.jsonl" -ForegroundColor Cyan
Write-Host "  Network logs:     logs\network\telemetry_*.jsonl" -ForegroundColor Cyan
Write-Host "  Gamepad logs:     logs\gamepad\gamepad_activity_*.jsonl" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Extract baselines: python gaming\extract_baselines.py telemetry\<file>.jsonl"
Write-Host "  2. Fuse telemetry:    python gaming\fuse_telemetry.py telemetry\<file>.jsonl"
Write-Host ""
