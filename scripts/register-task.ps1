<#
.SYNOPSIS
  Registers the weekly "arXiv Research Digest" task in Windows Task Scheduler (Monday 08:00 local time).

.NOTES
  - Runs as the current user, only when logged on (no stored password). The workstation must be powered on.
  - StartWhenAvailable: if the PC was off/asleep at 08:00 Monday, the task runs as soon as possible afterwards.
  - Task Scheduler uses local time; the pipeline computes windows in America/New_York regardless.
  - Unregister with:  Unregister-ScheduledTask -TaskName 'arXiv Research Digest' -Confirm:$false
#>
param(
    [string]$TaskName = 'arXiv Research Digest',
    [string]$At = '08:00',
    [ValidateSet('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')]
    [string]$Day = 'Monday',
    [switch]$WakeToRun
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Script = Join-Path $PSScriptRoot 'run-weekly.ps1'

$tz = (Get-TimeZone).Id
if ($tz -ne 'Eastern Standard Time') {
    Write-Warning "System time zone is '$tz'. The trigger fires at $At local time; collection windows are still computed in America/New_York."
}

$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Script`"" -WorkingDirectory $Root
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Day -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -WakeToRun:$WakeToRun
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
    -Description 'Local-first arXiv research digest (discovery, Ollama analysis, Obsidian report). Exit codes: 0 success, 2 partial, 1 failed.' -Force |
    Out-Null

Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State, @{n='Next run';e={(Get-ScheduledTaskInfo $_).NextRunTime}} | Format-List
