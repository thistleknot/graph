# run.ps1 -- pidfile-tracked lifecycle for the walker (streamlit) process.
#
# Spec: playbook.md T34 (the zombie fix -- prior manual `streamlit run` launches
# left orphaned processes with no record of their PID, so `stop` had no way to
# find them). This script is the sole start/stop/status entrypoint; a Makefile
# was considered and cut (T34 _Notes:).
#
# Usage:
#   ./run.ps1 start   # launch src/walker_app.py under streamlit, write .tmp/walker.pid
#   ./run.ps1 stop     # kill the pidfile's process tree, remove the pidfile
#   ./run.ps1 status   # report pidfile + health

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("start", "stop", "status")]
    [string]$Action
)

$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
$PidFile = Join-Path $RepoRoot ".tmp\walker.pid"
$Python = "C:\Users\user\py310\Scripts\python.exe"
$Port = 8501
$HealthUrl = "http://localhost:$Port/_stcore/health"

function Get-LiveProcess {
    if (-not (Test-Path $PidFile)) { return $null }
    $procId = Get-Content $PidFile -Raw | ForEach-Object { $_.Trim() }
    if (-not $procId) { return $null }
    $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
    if ($proc) { return $proc }
    return $null
}

function Test-Health {
    try {
        $resp = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2
        return $resp.StatusCode -eq 200
    } catch {
        return $false
    }
}

switch ($Action) {
    "start" {
        $existing = Get-LiveProcess
        if ($existing) {
            Write-Error "walker already running (pid $($existing.Id)); refusing to start a second copy. Run './run.ps1 stop' first."
            exit 1
        }

        $tmpDir = Join-Path $RepoRoot ".tmp"
        if (-not (Test-Path $tmpDir)) { New-Item -ItemType Directory -Path $tmpDir | Out-Null }

        if (-not $env:CHUNKGRAPH_MODEL_DIR) {
            $env:CHUNKGRAPH_MODEL_DIR = "C:/Users/user/models/m2v-minilm-l6-256"
        }

        $appPath = Join-Path $RepoRoot "src\walker_app.py"
        $argList = @("-m", "streamlit", "run", $appPath,
                     "--server.port", "$Port", "--server.headless", "true")

        $proc = Start-Process -FilePath $Python -ArgumentList $argList `
            -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru
        Set-Content -Path $PidFile -Value $proc.Id -NoNewline

        $deadline = (Get-Date).AddSeconds(30)
        $healthy = $false
        while ((Get-Date) -lt $deadline) {
            if (Test-Health) { $healthy = $true; break }
            Start-Sleep -Seconds 1
        }

        if ($healthy) {
            Write-Output "walker started, pid $($proc.Id), health ok at $HealthUrl"
        } else {
            Write-Error "walker started (pid $($proc.Id)) but did not become healthy within 30s"
            exit 1
        }
    }

    "stop" {
        if (-not (Test-Path $PidFile)) {
            Write-Output "no pidfile at $PidFile; nothing to stop"
            exit 0
        }
        $procId = (Get-Content $PidFile -Raw).Trim()
        $root = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if (-not $root) {
            Write-Output "pidfile names pid $procId but it is not running; removing stale pidfile"
            Remove-Item $PidFile -Force
            exit 0
        }

        # Kill the whole process tree rooted at the recorded pid (streamlit
        # spawns children; killing only the root leaves orphans) -- never by
        # process name, only by the PID this script itself recorded.
        $toKill = [System.Collections.Generic.Queue[int]]::new()
        $toKill.Enqueue([int]$procId)
        $allPids = New-Object System.Collections.Generic.List[int]
        while ($toKill.Count -gt 0) {
            $current = $toKill.Dequeue()
            $allPids.Add($current)
            $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$current" -ErrorAction SilentlyContinue
            foreach ($child in $children) { $toKill.Enqueue($child.ProcessId) }
        }
        foreach ($p in $allPids) {
            Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
        }
        Remove-Item $PidFile -Force
        Write-Output "stopped walker tree (pids: $($allPids -join ', '))"
    }

    "status" {
        $proc = Get-LiveProcess
        if (-not $proc) {
            Write-Output "walker: not running (no live pid in $PidFile)"
            exit 0
        }
        $healthy = Test-Health
        Write-Output "walker: running (pid $($proc.Id)), health=$healthy ($HealthUrl)"
    }
}
