# Interview Playground verified HR2 startup from this script's own folder.
# Run in PowerShell: powershell -ExecutionPolicy Bypass -File .\Start-InterviewPlayground.ps1
param(
    [int]$Port = 8765,
    [switch]$InstallDependencies,
    [switch]$SkipApiPrompt
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host 'Creating isolated Python virtual environment...'
    if (Get-Command py -ErrorAction SilentlyContinue) { & py -3 -m venv '.venv' }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { & python -m venv '.venv' }
    else { throw 'Python 3 not found. Install Python 3 and retry.' }
    $InstallDependencies = $true
}
if (-not (Test-Path -LiteralPath $Python)) { throw 'Virtual environment creation failed.' }
if ($InstallDependencies) {
    & $Python -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
if (-not $env:INTERVIEW_SERVER_BASE_URL) { $env:INTERVIEW_SERVER_BASE_URL = 'https://api.shubiaobiao.cn/v1' }
if (-not $env:QWEN_MAAS_BASE_URL) { $env:QWEN_MAAS_BASE_URL = 'https://maas.qianwenaiapi.com/compatible-mode/v1' }
if (-not $env:INTERVIEW_SERVER_MODEL) { $env:INTERVIEW_SERVER_MODEL = 'qwen3.5-flash' }
if (-not $env:QWEN_MAAS_MODEL) { $env:QWEN_MAAS_MODEL = 'qwen3.8-2.4t-a95b' }
# Interviewer MaaS and Shubiaobiao are separate configurations.
# Never assume a key for one URL authorizes another provider.
# Respect original InterviewAgent LLM_* settings. They may identify a
# completely different provider; do not silently overwrite its credentials.
if (-not $SkipApiPrompt -and -not $env:INTERVIEW_SERVER_API_KEY -and -not $env:QWEN_MAAS_API_KEY -and -not $env:LLM_API_KEY) {
    Write-Host 'No interviewer key found. Configure the active provider in the browser.'
}
Write-Host "Interview provider shubi: $env:INTERVIEW_SERVER_BASE_URL ; model: $env:INTERVIEW_SERVER_MODEL"
Write-Host "Interview provider maas: $env:QWEN_MAAS_BASE_URL ; model: $env:QWEN_MAAS_MODEL"
Write-Host 'The active provider depends on the Interviewer mode selected in the webpage.'
Write-Host 'Original LLM_* settings preserved. Do not reuse keys across providers.'
Write-Host 'API keys will not be printed.'
& $Python 'server.py' --port $Port --open-browser
if ($LASTEXITCODE -ne 0) { throw "Server exited with code $LASTEXITCODE" }
