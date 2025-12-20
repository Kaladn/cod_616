# CompuCog Combat Telemetry — Live Session
# 
# Launch this while in-game to watch the cognitive organism perceive combat
#
# What you'll see:
# - Real-time EOMM detection
# - Event firing on manipulation spikes
# - 6-1-6 capsules for high-priority events
# - Operator triggers (aim resistance, ghost bullets, insta-melt, spawn manipulation)
#
# Usage:
#   1. Start your game
#   2. Run: .\launch_combat_session.ps1
#   3. Play for 2-5 minutes
#   4. Review the capsules and events after

param(
    [int]$Duration = 120  # 2 minutes default
)

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79)
Write-Host "CompuCog Combat Telemetry — Cognitive Organism Live Session" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79)
Write-Host ""

Write-Host "Duration: " -NoNewline -ForegroundColor White
Write-Host "$Duration seconds ($([math]::Round($Duration/60, 1)) minutes)" -ForegroundColor Green
Write-Host ""

Write-Host "Instructions:" -ForegroundColor Cyan
Write-Host "  1. Get into a match" -ForegroundColor White
Write-Host "  2. Press ENTER when ready to start capture" -ForegroundColor White
Write-Host "  3. Play normally - the organism will perceive and remember" -ForegroundColor White
Write-Host "  4. Review events and capsules after session" -ForegroundColor White
Write-Host ""

Read-Host "Press ENTER to begin capture"

Write-Host ""
Write-Host "🔥 STARTING COGNITIVE CAPTURE..." -ForegroundColor Red
Write-Host ""

# Run the live integration
python truevision_event_live.py --duration $Duration

Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79)
Write-Host "Session Complete — Reviewing Memory..." -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79)
Write-Host ""

# Show Forge records
Write-Host "📋 Forge Memory Contents:" -ForegroundColor Cyan
python ..\memory\forge_inspect.py --data-dir forge_data --limit 10

Write-Host ""
Write-Host "✅ Combat session recorded in forge_data/" -ForegroundColor Green
Write-Host ""
