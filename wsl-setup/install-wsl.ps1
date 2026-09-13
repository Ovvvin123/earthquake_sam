$ErrorActionPreference = 'Continue'
$OutputEncoding = [System.Text.Encoding]::Unicode
[Console]::OutputEncoding = [System.Text.Encoding]::Unicode
$taskLog = Join-Path $PSScriptRoot 'install.log'
& "$env:SystemRoot\System32\wsl.exe" --install -d Ubuntu-22.04 --no-launch 2>&1 | Out-File -LiteralPath $taskLog -Encoding utf8
$taskExit = $LASTEXITCODE
Set-Content -LiteralPath (Join-Path $PSScriptRoot 'exit-code.txt') -Value $taskExit -Encoding utf8
exit $taskExit
