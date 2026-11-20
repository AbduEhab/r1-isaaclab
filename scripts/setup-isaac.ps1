<#
.SYNOPSIS
    One-shot setup for the R1 Isaac Lab stack + the labd queue daemon, all under one root.

.DESCRIPTION
    Creates a self-contained tree and wires the pieces together:

        <Root>\
          IsaacSim\        Isaac Sim 6.2 (binary install; extracted from -IsaacSimZip,
                           or the path given with -IsaacSimPath)
          IsaacLab\        Isaac Lab v3.0.0 from the repository, with _isaac_sim -> IsaacSim
          r1-isaaclab\     this repository (task, asset, scripts, docs)
          labd\labd.exe    the training-queue daemon
          artifacts\       finished runs copied here by labd
          logs\            setup and install logs
          start-labd.cmd   starts the daemon with the right paths

    The Python environment is Isaac Sim's bundled interpreter, used through Isaac Lab's own
    wrapper (isaaclab.bat -p). That is NVIDIA's documented pairing for a binary Isaac Sim install;
    combining a binary install with a conda/uv/venv environment is explicitly unsupported. A venv
    is only an option when Isaac Sim itself is installed from pip.

.PARAMETER Root
    Where everything goes. Default C:\isaac.

.PARAMETER IsaacSimPath
    An existing Isaac Sim install to link (contains python.bat). Default: <Root>\IsaacSim.

.PARAMETER IsaacSimZip
    A downloaded Isaac Sim 6.2 zip to extract into <Root>\IsaacSim.

.PARAMETER IsaacLabPath
    An existing IsaacLab checkout to reuse instead of cloning. Its _isaac_sim link is still created.

.PARAMETER IsaLabTag
    Isaac Lab tag to clone. Default v3.0.0.

.PARAMETER RepoSource
    Where r1-isaaclab comes from: a git URL, or a local folder to copy. Default: the checkout this
    script belongs to.

.PARAMETER SkipIsaacLabInstall
    Skip `isaaclab.bat --install` (it downloads torch and the RL frameworks; 30-60 min first time).

.PARAMETER SkipVerify
    Skip the two-iteration smoke run at the end.

.PARAMETER OpenFirewall
    Add an inbound rule for TCP 8080 on private networks (needs an elevated shell; otherwise add it
    by hand when Windows asks on the first run).

.EXAMPLE
    .\setup-isaac.ps1 -IsaacSimZip D:\downloads\isaac-sim-6.2.0.zip

.EXAMPLE
    # reuse an Isaac Sim already installed at C:\isaac-sim
    .\setup-isaac.ps1 -IsaacSimPath C:\isaac-sim
#>
[CmdletBinding()]
param(
    [string]$Root = "C:\isaac",
    [string]$IsaacSimPath = "",
    [string]$IsaacSimZip = "",
    [string]$IsaacLabPath = "",
    [string]$IsaacLabTag = "v3.0.0",
    [string]$RepoSource = "",
    [switch]$SkipIsaacLabInstall,
    [switch]$SkipVerify,
    [switch]$OpenFirewall
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Say([string]$Message) { Write-Host "==> $Message" -ForegroundColor Cyan }
function Note([string]$Message) { Write-Host "    $Message" -ForegroundColor DarkGray }
function Fail([string]$Message) { Write-Host "ERROR: $Message" -ForegroundColor Red; exit 1 }

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments, [string]$WorkDir = "", [string]$LogFile = "")
    $line = "$Exe $($Arguments -join ' ')"
    Note $line
    $pushed = $false
    if ($WorkDir) { Push-Location $WorkDir; $pushed = $true }
    try {
        if ($LogFile) {
            & $Exe @Arguments 2>&1 | Tee-Object -FilePath $LogFile
        } else {
            & $Exe @Arguments
        }
        $code = $LASTEXITCODE
    } finally {
        if ($pushed) { Pop-Location }
    }
    if ($code -ne 0) { Fail "command failed (exit $code): $line" }
}

# ----------------------------------------------------------------------------- root and logging
$root = (New-Item -ItemType Directory -Force -Path $Root).FullName
$logs = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$artifacts = Join-Path $root "artifacts"
New-Item -ItemType Directory -Force -Path $artifacts | Out-Null

Say "root: $root"
$freeGb = [math]::Round((Get-Item $root).PSDrive.Free / 1GB, 1)
Note "free space on $((Get-Item $root).PSDrive.Name): ${freeGb} GB"
if ($freeGb -lt 60) {
    Fail "only ${freeGb} GB free: Isaac Sim, Isaac Lab and torch need about 60 GB"
}

# Kit and Isaac Sim prompt for the NVIDIA EULA on a first start; accept it from the environment so
# an unattended setup does not block on a dialog.
$env:OMNI_KIT_ACCEPT_EULA = "YES"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "git is required: install Git for Windows, or copy IsaacLab and r1-isaaclab in by hand"
}

# ----------------------------------------------------------------------------- Isaac Sim
if ($IsaacSimZip) {
    $sim = Join-Path $root "IsaacSim"
    if (-not (Test-Path (Join-Path $sim "python.bat"))) {
        Say "extracting Isaac Sim: $IsaacSimZip"
        New-Item -ItemType Directory -Force -Path $sim | Out-Null
        Expand-Archive -Path $IsaacSimZip -DestinationPath $sim -Force
    }
} elseif ($IsaacSimPath) {
    if (-not (Test-Path $IsaacSimPath)) { Fail "IsaacSimPath does not exist: $IsaacSimPath" }
    $sim = (Resolve-Path $IsaacSimPath).Path
} else {
    $sim = Join-Path $root "IsaacSim"
}

if (-not (Test-Path (Join-Path $sim "python.bat"))) {
    Fail @"
Isaac Sim 6.2 not found at '$sim' (no python.bat).
Download the 6.2 binary zip from https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html
then either pass -IsaacSimZip <zip> or -IsaacSimPath <existing install>.
"@
}
Say "Isaac Sim: $sim"

# ----------------------------------------------------------------------------- Isaac Lab
if ($IsaacLabPath) {
    $lab = (Resolve-Path $IsaacLabPath).Path
} else {
    $lab = Join-Path $root "IsaacLab"
    if (-not (Test-Path (Join-Path $lab "isaaclab.bat"))) {
        Say "cloning Isaac Lab $IsaacLabTag"
        New-Item -ItemType Directory -Force -Path (Split-Path $lab) | Out-Null
        Invoke-Checked git @("clone", "--depth", "1", "--branch", $IsaacLabTag,
            "https://github.com/isaac-sim/IsaacLab.git", $lab) "" (Join-Path $logs "isaaclab-clone.log")
    }
}
if (-not (Test-Path (Join-Path $lab "isaaclab.bat"))) { Fail "not an IsaacLab checkout: $lab" }
Say "Isaac Lab: $lab"

$link = Join-Path $lab "_isaac_sim"
if (-not (Test-Path $link)) {
    Say "linking $link -> $sim"
    # /J is a directory junction: works without administrator rights, unlike /D.
    $out = cmd /c "mklink /J `"$link`" `"$sim`"" 2>&1
    Note ($out -join ' ')
    if (-not (Test-Path $link)) { Fail "could not create the _isaac_sim junction" }
} else {
    Note "_isaac_sim already linked"
}

if (-not $SkipIsaacLabInstall) {
    Say "installing Isaac Lab (torch 2.11 + RL frameworks; 30-60 min the first time)"
    Invoke-Checked cmd @("/c", "isaaclab.bat --install") $lab (Join-Path $logs "isaaclab-install.log")
    Say "Isaac Lab installed"
} else {
    Note "skipping isaaclab.bat --install"
}

# ----------------------------------------------------------------------------- this repository
$repo = Join-Path $root "r1-isaaclab"
if (-not (Test-Path (Join-Path $repo "scripts\train.py"))) {
    if ($RepoSource -and ($RepoSource -match '^(https?://|git@)')) {
        Say "cloning r1-isaaclab from $RepoSource"
        Invoke-Checked git @("clone", $RepoSource, $repo) "" (Join-Path $logs "repo-clone.log")
    } else {
        $source = if ($RepoSource) { (Resolve-Path $RepoSource).Path } else { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path }
        if (-not (Test-Path (Join-Path $source "scripts\train.py"))) { Fail "r1-isaaclab not found at $source" }
        Say "copying r1-isaaclab from $source"
        $null = New-Item -ItemType Directory -Force -Path $repo
        $exclude = @(".git", "logs", "labd-artifacts", "target", "__pycache__", ".venv")
        robocopy $source $repo /E /NFL /NDL /NJH /NJS /NP /XD @exclude /XF "*.pyc" | Out-Null
        if ($LASTEXITCODE -ge 8) { Fail "robocopy failed copying $source" }
    }
}
Say "repository: $repo"

# ----------------------------------------------------------------------------- labd
$labdDir = Join-Path $root "labd"
New-Item -ItemType Directory -Force -Path $labdDir | Out-Null
$labdExe = Join-Path $labdDir "labd.exe"
$built = Join-Path $repo "labd\dist\labd.exe"
if (Test-Path $built) {
    Copy-Item $built $labdExe -Force
} elseif (Get-Command cargo -ErrorAction SilentlyContinue) {
    Say "building labd (no prebuilt binary in the repository)"
    Invoke-Checked cargo @("build", "--release") (Join-Path $repo "labd") (Join-Path $logs "labd-build.log")
    Copy-Item (Join-Path $repo "labd\target\release\labd.exe") $labdExe -Force
} else {
    Note "no labd binary and no cargo; the daemon will be skipped"
}
if (Test-Path $labdExe) { Say "labd: $labdExe" }

# ----------------------------------------------------------------------------- launcher
$cmd = @"
@echo off
rem Starts the R1 training queue on this machine; open http://<this-machine>:8080
"%~dp0labd\labd.exe" --repo "%~dp0r1-isaaclab" --isaaclab "%~dp0IsaacLab" --port 8080 --artifacts "%~dp0artifacts"
pause
"@
Set-Content -Path (Join-Path $root "start-labd.cmd") -Value $cmd -Encoding ascii
Say "launcher: $(Join-Path $root 'start-labd.cmd')"

# ----------------------------------------------------------------------------- verify
if (-not $SkipVerify) {
    Say "verifying: two training iterations of Isaac-Velocity-Flat-R1 (headless)"
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    Invoke-Checked cmd @("/c", "isaaclab.bat -p `"$repo\scripts\train.py`" --rl_library rsl_rl --task Isaac-Velocity-Flat-R1 --viz none --num_envs 8 --max_iterations 2") $lab (Join-Path $logs "verify.log")
    Say "verification passed"
}

# ----------------------------------------------------------------------------- firewall
if ($OpenFirewall) {
    try {
        New-NetFirewallRule -DisplayName "labd TCP 8080" -Direction Inbound -Protocol TCP `
            -LocalPort 8080 -Action Allow -Profile Private -ErrorAction Stop | Out-Null
        Say "firewall: inbound TCP 8080 allowed on private networks"
    } catch {
        Note "could not add the firewall rule (needs an elevated shell): $($_.Exception.Message)"
    }
}

# ----------------------------------------------------------------------------- summary
Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "  Isaac Sim    $sim"
Write-Host "  Isaac Lab    $lab"
Write-Host "  repository   $repo"
Write-Host "  labd         $labdExe"
Write-Host "  artifacts    $artifacts"
Write-Host ""
Write-Host "  Start the queue:   $(Join-Path $root 'start-labd.cmd')"
Write-Host "  Then open:         http://<this-machine>:8080"
Write-Host ""
Write-Host "  Allow TCP 8080 through Windows Firewall on the first run (private networks)." -ForegroundColor DarkGray
Write-Host "  Logs of this setup: $logs" -ForegroundColor DarkGray

