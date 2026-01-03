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
    [string]$OutputPath = "",
    [double]$SessionEpoch = 0,
    [string]$SessionId = "",
    [int]$SampleIntervalSeconds = 3
)

# Determine output file
if ($OutputPath -ne "") {
    $file = $OutputPath
    $LogDir = Split-Path $OutputPath -Parent
} else {
    $file = "$LogDir\network_state_$((Get-Date).ToString('yyyyMMdd')).jsonl"
}

# Create log directory if it doesn't exist
if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# Function to get Unix epoch with millisecond precision
function Get-UnixEpoch {
    $now = [DateTime]::UtcNow
    $epoch = [DateTime]::new(1970, 1, 1, 0, 0, 0, [DateTimeKind]::Utc)
    return ($now - $epoch).TotalSeconds
}

# Function to get offset from session start
function Get-SessionOffsetMs {
    param([double]$SessionStart)
    if ($SessionStart -eq 0) { return 0 }
    $now = Get-UnixEpoch
    return [math]::Round(($now - $SessionStart) * 1000, 3)
}

# Function to cleanup old log files (older than 1 day)
function Remove-OldLogs {
    param([string]$LogDirectory)
    try {
        $cutoff = (Get-Date).AddDays(-1)
        $oldFiles = Get-ChildItem -Path $LogDirectory -Filter "*.jsonl" -ErrorAction SilentlyContinue | 
                    Where-Object { $_.LastWriteTime -lt $cutoff }
        foreach ($oldFile in $oldFiles) {
            try {
                Remove-Item $oldFile.FullName -Force -ErrorAction SilentlyContinue
                Write-Host "[Cleanup] Deleted old log: $($oldFile.Name)"
            } catch {}
        }
    } catch {}
}

# Auto-cleanup old logs on startup (older than 1 day)
Write-Host "[Cleanup] Removing logs older than 1 day..."
Remove-OldLogs -LogDirectory $LogDir

Write-Host "CompuCogLogger - Network Telemetry [IMMORTAL]"
Write-Host "Log File: $file"
if ($SessionId -ne "") {
    Write-Host "Session ID: $SessionId"
    Write-Host "Session Epoch: $SessionEpoch"
}
Write-Host "Sample Interval: $SampleIntervalSeconds seconds"
Write-Host "IMMORTAL mode - will never exit"
Write-Host ""

$eventCount = 0

# IMMORTAL LOOP - while $true FOREVER
while ($true) {
    try {
        $timestamp = (Get-Date).ToString("o")
        $eventEpoch = Get-UnixEpoch
        $offsetMs = Get-SessionOffsetMs -SessionStart $SessionEpoch
        $data = @()

        # Capture TCP connections (safe)
        try {
            $conns = Get-NetTCPConnection -ErrorAction SilentlyContinue | Where-Object {
                $_.RemoteAddress -and $_.RemoteAddress -ne "0.0.0.0" -and $_.RemoteAddress -ne "::"
            }
            
            foreach ($c in $conns) {
                $pname = ""
                try { 
                    $pname = (Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue).ProcessName 
                } catch {}
                
                $record = [ordered]@{
                    session_id      = $SessionId
                    session_epoch   = $SessionEpoch
                    event_epoch     = $eventEpoch
                    event_offset_ms = $offsetMs
                    Timestamp       = $timestamp
                    LocalAddress    = $c.LocalAddress
                    LocalPort       = $c.LocalPort
                    RemoteAddress   = $c.RemoteAddress
                    RemotePort      = $c.RemotePort
                    State           = $c.State.ToString()
                    Protocol        = "TCP"
                    PID             = $c.OwningProcess
                    ProcessName     = $pname
                }
                $data += [PSCustomObject]$record
            }
        } catch {
            # Ignore TCP errors, continue
        }

        # Capture UDP endpoints (safe)
        try {
            $udpConns = Get-NetUDPEndpoint -ErrorAction SilentlyContinue | Where-Object {
                $_.LocalAddress -and $_.LocalAddress -ne "0.0.0.0" -and $_.LocalAddress -ne "::"
            }
            
            foreach ($u in $udpConns) {
                $pname = ""
                try { 
                    $pname = (Get-Process -Id $u.OwningProcess -ErrorAction SilentlyContinue).ProcessName 
                } catch {}
                
                $record = [ordered]@{
                    session_id      = $SessionId
                    session_epoch   = $SessionEpoch
                    event_epoch     = $eventEpoch
                    event_offset_ms = $offsetMs
                    Timestamp       = $timestamp
                    LocalAddress    = $u.LocalAddress
                    LocalPort       = $u.LocalPort
                    RemoteAddress   = "0.0.0.0"
                    RemotePort      = 0
                    State           = "Listen"
                    Protocol        = "UDP"
                    PID             = $u.OwningProcess
                    ProcessName     = $pname
                }
                $data += [PSCustomObject]$record
            }
        } catch {
            # Ignore UDP errors, continue
        }

        # Write to JSONL (immortal - retry 3 times per event)
        foreach ($event in $data) {
            $maxRetries = 3
            $retryCount = 0
            $success = $false
            
            while (-not $success -and $retryCount -lt $maxRetries) {
                try {
                    $jsonLine = $event | ConvertTo-Json -Depth 2 -Compress
                    $sw = [System.IO.StreamWriter]::new($file, $true, [System.Text.Encoding]::UTF8)
                    try {
                        $sw.WriteLine($jsonLine)
                        $sw.Flush()
                        $success = $true
                        $eventCount++
                    } finally {
                        $sw.Close()
                    }
                } catch {
                    $retryCount++
                    if ($retryCount -lt $maxRetries) {
                        Start-Sleep -Milliseconds 50
                    }
                    # Don't log errors - just continue
                }
            }
        }

        # Status update every 1000 events
        if ($eventCount % 1000 -eq 0 -and $eventCount -gt 0) {
            Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] $eventCount events"
        }

    } catch {
        # ANY error in loop body - ignore and continue
        # Don't even log it - just keep going
    }

    # ALWAYS sleep (even after errors)
    Start-Sleep -Seconds $SampleIntervalSeconds
}

# This line will NEVER be reached (IMMORTAL)
