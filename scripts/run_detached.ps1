# Start a long command so it keeps running after the terminal (or Claude Code session) closes.
# Windows has no nohup/tmux; a process created through WMI is not part of the caller's process
# tree or job, so closing the caller does not kill it. Output goes to outputs/logs/<Name>.log.
#
#   powershell -File scripts/run_detached.ps1 -Name final_sweep `
#       -Command ".venv\Scripts\python.exe scripts\run_e2e.py --config configs\e2e_final_noise_cnn.json"
#
# Chain several with "&&". run_e2e.py is resumable: re-running the same command after a crash or
# reboot continues where it stopped. Check progress with:  Get-Content outputs\logs\<Name>.log -Tail 3
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Command,
    [int]$Threads = 0  # >0 caps torch/OpenMP threads (use when several jobs share the CPU)
)
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$logDir = Join-Path $root "outputs\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir "$Name.log"
$envs = "set PYTHONUTF8=1&& "
if ($Threads -gt 0) { $envs += "set OMP_NUM_THREADS=$Threads&& set MKL_NUM_THREADS=$Threads&& " }
$cmdline = "cmd.exe /c `"$envs$Command > `"$log`" 2>&1`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
    -Arguments @{ CommandLine = $cmdline; CurrentDirectory = $root }
if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create failed with code $($r.ReturnValue)" }
"started pid $($r.ProcessId): $Command`nlog: $log"
