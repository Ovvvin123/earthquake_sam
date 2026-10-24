param(
    [switch]$Execute,
    [string]$ReportDir = ''
)
$ErrorActionPreference = 'Stop'
$wslProject = (& wsl -d Ubuntu-22.04 -- wslpath -a $PSScriptRoot)
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve project path in WSL' }
$wslProject = $wslProject.Trim()
$launchArguments = @('-u', 'run_experiment.py', '--sequence',
    'experiments/california_ds5.json', 'experiments/greece_nz_original.json')
if ($Execute) { $launchArguments += '--execute' }
if ($ReportDir) { $launchArguments += @('--reportDir', $ReportDir) }
& wsl -d Ubuntu-22.04 --cd $wslProject -- env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg /home/ovvvin/.venvs/earthquake-sam/bin/python @launchArguments
if ($LASTEXITCODE -ne 0) { throw "Experiment check/queue stopped (exit $LASTEXITCODE). Inspect the output or console.log; do not blindly restart." }
