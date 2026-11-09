$ErrorActionPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSScriptRoot
$conns = Get-NetTCPConnection -LocalPort 8501 -State Listen
foreach ($c in $conns) { Stop-Process -Id $c.OwningProcess -Force }
Start-Sleep -Seconds 2
Start-Process -FilePath (Join-Path $root '.venv\Scripts\streamlit.exe') `
    -ArgumentList 'run','webui\Main.py','--server.port','8501','--server.headless','true' `
    -WorkingDirectory $root -WindowStyle Hidden
Start-Sleep -Seconds 10
try {
    (Invoke-WebRequest -Uri 'http://127.0.0.1:8501/' -UseBasicParsing -TimeoutSec 10).StatusCode
} catch {
    'ERR: ' + $_.Exception.Message
}
