<#
Register the graph-to-BM25 watcher as a hidden logon task for the current user.

| Stage | Mechanism | Tag | Ref |
|---|---|---|---|
| VALIDATE | Check the dedicated graph interpreter and absence of a duplicate task. | [mandatory] | R1 |
| REGISTER | Create a current-user interactive logon task with restart-on-failure settings. | [mandatory] | R2 |
| START | Start the worker now and report task state and log path. | [mandatory] | R3 |

R1 The task uses the F: graph environment and repository root; database DSNs come from the user's environment or `src/config.py` defaults.
R2 The task runs only after this user logs on; the user account must have read/write access to the repository and database endpoints.
R3 The worker polls every 60 seconds, logs to `.tmp/arxiv_bm25_worker.log`, and rebuilds only when the BM25 source graph build is stale or incomplete.

CLOSED: NSSM service hosting is removed because Task Scheduler is the selected runtime manager. A boot-time service account is not used; the existing graph tasks use this user's interactive logon.

Preconditions: run from the repository on the graph host; `F:\Python\mcp\graph\Scripts\python.exe` exists. Failure modes: task registration errors stop the script; database/build errors are logged and retried by the worker.
#>
param(
    [string]$PythonPath = "F:\Python\mcp\graph\Scripts\python.exe",
    [string]$TaskName = "arxiv-bm25-watch",
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$worker = Join-Path $repoRoot "src\arxiv_bm25_worker.py"
$logPath = Join-Path $repoRoot ".tmp\arxiv_bm25_worker.log"
$userId = "$env:USERDOMAIN\$env:USERNAME"

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Graph Python interpreter not found: $PythonPath"
}
if (-not (Test-Path -LiteralPath $worker -PathType Leaf)) {
    throw "BM25 worker not found: $worker"
}
if ($PollSeconds -lt 10) {
    throw "PollSeconds must be at least 10."
}
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "Scheduled task '$TaskName' already exists; inspect it before changing its registration."
}

New-Item -ItemType Directory -Path (Split-Path -Parent $logPath) -Force | Out-Null
$escapedRoot = $repoRoot.Replace("'", "''")
$escapedPython = $PythonPath.Replace("'", "''")
$escapedLog = $logPath.Replace("'", "''")
$command = "Set-Location '$escapedRoot'; & '$escapedPython' -u src\arxiv_bm25_worker.py --watch --poll-seconds $PollSeconds *>> '$escapedLog'"
$windowsPowerShell = (Get-Command powershell.exe -ErrorAction Stop).Source
$action = New-ScheduledTaskAction -Execute $windowsPowerShell `
    -Argument "-NoProfile -WindowStyle Hidden -Command `"$command`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $userId
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Description "Keep the arXiv BM25 build synchronized with the active section graph." | Out-Null
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 2
$task = Get-ScheduledTask -TaskName $TaskName
$taskInfo = Get-ScheduledTaskInfo -TaskName $TaskName
if ($task.State -ne "Running") {
    throw "Task registered but did not remain running (state=$($task.State), last result=$($taskInfo.LastTaskResult)). See $logPath"
}

[pscustomobject]@{
    TaskName = $TaskName
    State = $task.State
    User = $userId
    Python = $PythonPath
    PollSeconds = $PollSeconds
    Log = $logPath
}
