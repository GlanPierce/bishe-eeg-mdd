$ErrorActionPreference = 'Stop'

$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Error '未找到 .venv\Scripts\python.exe，请先完成环境部署。'
}

Write-Host 'Running environment check...'
& $venvPython "$PSScriptRoot\env_check.py"
if ($LASTEXITCODE -ne 0) {
    Write-Error '环境检查失败，请先修复依赖问题。'
}

Write-Host "\n启动 JupyterLab："
Write-Host "& .\.venv\Scripts\python.exe -m jupyter lab --notebook-dir `"$PSScriptRoot`""
