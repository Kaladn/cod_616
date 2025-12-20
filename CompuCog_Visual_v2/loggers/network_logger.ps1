# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                              ║
# ║     CompuCog — Sovereign Cognitive Defense System                           ║
# ║     Intellectual Property of Cortex Evolved / L.A. Mercey                   ║
# ║                                                                              ║
# ║     Copyright © 2025 Cortex Evolved. All Rights Reserved.                   ║
# ║                                                                              ║
# ║     "We use unconventional digital wisdom —                                  ║
# ║        because conventional digital wisdom doesn't protect anyone."         ║
# ║                                                                              ║
# ║     This software is proprietary and confidential.                           ║
# ║     Unauthorized access, copying, modification, or distribution             ║
# ║     is strictly prohibited and may violate applicable laws.                  ║
# ║                                                                              ║
# ║     File automatically watermarked on: 2025-11-29 19:21:12                           ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ========= network_logger.ps1 =========
# CompuCogLogger - Network Telemetry Module
# Captures TCP/UDP connections with process information every 3 seconds
# Schema: compucog_schema.py::NetworkEvent

param(
    [string]$LogDir = "$PSScriptRoot\..\logs\network",
    [int]$SampleIntervalSeconds = 3
)

# Create log directory if it doesn't exist
if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

Write-Host "CompuCogLogger - Network Telemetry Started"
Write-Host "Log Directory: $LogDir"
Write-Host "Sample Interval: $SampleIntervalSeconds seconds"
Write-Host "Press Ctrl+C to stop"
Write-Host ""

$eventCount = 0

while ($true) {
    $timestamp = (Get-Date).ToString("o")
    $data = @()

    try {
        # Capture TCP connections
        $conns = Get-NetTCPConnection -ErrorAction SilentlyContinue | Where-Object {
            $_.RemoteAddress -and $_.RemoteAddress -ne "0.0.0.0" -and $_.RemoteAddress -ne "::"
        }
        
        foreach ($c in $conns) {
            $pname = ""
            try { 
                $pname = (Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue).ProcessName 
            } catch {}
            
            $data += [PSCustomObject]@{
                Timestamp     = $timestamp
                LocalAddress  = $c.LocalAddress
                LocalPort     = $c.LocalPort
                RemoteAddress = $c.RemoteAddress
                RemotePort    = $c.RemotePort
                State         = $c.State.ToString()
                Protocol      = "TCP"
                PID           = $c.OwningProcess
                ProcessName   = $pname
            }
        }

        # Capture UDP endpoints
        $udpConns = Get-NetUDPEndpoint -ErrorAction SilentlyContinue | Where-Object {
            $_.LocalAddress -and $_.LocalAddress -ne "0.0.0.0" -and $_.LocalAddress -ne "::"
        }
        
        foreach ($u in $udpConns) {
            $pname = ""
            try { 
                $pname = (Get-Process -Id $u.OwningProcess -ErrorAction SilentlyContinue).ProcessName 
            } catch {}
            
            $data += [PSCustomObject]@{
                Timestamp     = $timestamp
                LocalAddress  = $u.LocalAddress
                LocalPort     = $u.LocalPort
                RemoteAddress = "0.0.0.0"  # UDP doesn't track remote endpoint
                RemotePort    = 0
                State         = "Listen"
                Protocol      = "UDP"
                PID           = $u.OwningProcess
                ProcessName   = $pname
            }
        }
    } catch {
        Write-Host "WARNING: Error capturing network data: $_"
    }

    # Write to JSONL file - ONE event per line (no array wrapper)
    $file = "$LogDir\telemetry_$((Get-Date).ToString('yyyyMMdd')).jsonl"
    
    foreach ($event in $data) {
        try {
            $event | ConvertTo-Json -Depth 2 -Compress | Out-File -Append -Encoding UTF8 -FilePath $file
            $eventCount++
        } catch {
            Write-Host "ERROR: Failed to write event: $_"
        }
    }

    # Status update every 60 seconds
    if ($eventCount % 1000 -eq 0 -and $eventCount -gt 0) {
        Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] Collected $eventCount events"
    }

    Start-Sleep -Seconds $SampleIntervalSeconds
}
