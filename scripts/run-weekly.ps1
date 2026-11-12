<#
.SYNOPSIS
  Weekly entry point used by Windows Task Scheduler.
  Ensures Ollama is serving, runs the full pipeline, and returns its exit code:
    0 = SUCCESS (or nothing to do), 1 = FAILED, 2 = PARTIAL (report kept, window NOT advanced).
#>
param(
    [switch]$CollectOnly,
    [int]$OllamaWaitSeconds = 120
)

$ErrorActionPreference = 'Stop'
# Python's console output contains non-ASCII characters; without UTF-8 a cp1252 console can crash the final print.
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$LogDir = Join-Path $Root 'data\logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Transcript = Join-Path $LogDir ("scheduler-{0:yyyyMMdd-HHmmss}.log" -f (Get-Date))
Start-Transcript -Path $Transcript | Out-Null

try {
    if (-not (Test-Path $Python)) { throw "virtual environment not found: $Python (run: python -m venv .venv; .venv\Scripts\pip install -e .)" }

    if (-not $CollectOnly) {
        $ollamaUp = $false
        try { Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 5 | Out-Null; $ollamaUp = $true } catch {}
        if (-not $ollamaUp) {
            Write-Host 'Ollama not reachable; starting "ollama serve"...'
            $ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
            if (-not $ollama) { $ollama = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe' }
            Start-Process -FilePath $ollama -ArgumentList 'serve' -WindowStyle Hidden
            $deadline = (Get-Date).AddSeconds($OllamaWaitSeconds)
            while ((Get-Date) -lt $deadline -and -not $ollamaUp) {
                Start-Sleep -Seconds 3
                try { Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 5 | Out-Null; $ollamaUp = $true } catch {}
            }
            if (-not $ollamaUp) { Write-Warning 'Ollama did not start; the run will be recorded as FAILED at the ANALYZED stage.' }
        }
    }

    Set-Location $Root
    $cmd = if ($CollectOnly) { 'collect' } else { 'run' }
    & $Python -m arxiv_digest $cmd
    $code = $LASTEXITCODE
    Write-Host "digest $cmd exited with code $code"
}
catch {
    Write-Error $_
    $code = 1
}
finally {
    Stop-Transcript | Out-Null
}
exit $code
