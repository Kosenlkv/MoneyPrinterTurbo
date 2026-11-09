param(
    [Parameter(Mandatory=$true)][string]$Script,
    [Parameter(Mandatory=$true)][string]$LogFile,
    [string]$WorkDir = (Split-Path -Parent $PSScriptRoot)
)
$python = Join-Path $WorkDir '.venv\Scripts\python.exe'
$log = Join-Path $WorkDir $LogFile
if (Test-Path $log) { Remove-Item $log -Force }
$p = Start-Process -FilePath $python -ArgumentList $Script `
    -WorkingDirectory $WorkDir -WindowStyle Hidden `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru
Write-Output "PID=$($p.Id)"
