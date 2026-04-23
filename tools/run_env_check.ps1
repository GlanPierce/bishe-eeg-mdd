$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Error 'Missing .venv\\Scripts\\python.exe. Set up the local environment first.'
}

Write-Host 'Running environment check...'
& $venvPython (Join-Path $PSScriptRoot 'env_check.py')
if ($LASTEXITCODE -ne 0) {
    Write-Error 'Environment check failed. Fix the missing dependencies first.'
}

Write-Host ''
Write-Host 'Launch JupyterLab with:'
Write-Host ('.\\.venv\\Scripts\\python.exe -m jupyter lab --notebook-dir "{0}"' -f $repoRoot)
