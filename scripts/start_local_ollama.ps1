#Requires -Version 7.4
<#
.SYNOPSIS
Starts the optional workspace-local Ollama service on Windows.
.DESCRIPTION
The standalone runtime must already exist at .venv/tools/ollama/ollama.exe.
This helper does not download software or models, request completions, change
global environment settings, or require administrator privileges. It starts a
hidden loopback-only service and stores its models, identity, temporary files,
and logs under .venv/tools. An existing service is reused only when its saved
PID, executable, loopback listener, and health response match this installation.
.EXAMPLE
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/start_local_ollama.ps1
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not $IsWindows) {
    throw 'This optional standalone-runtime helper supports Windows only.'
}

$taskProject = Split-Path -Parent $PSScriptRoot
$taskTools = Join-Path $taskProject '.venv/tools'
$taskExecutable = Join-Path $taskTools 'ollama/ollama.exe'
$taskRuntime = Join-Path $taskTools 'ollama-runtime'
$taskModels = Join-Path $taskTools 'ollama-models'
$taskHomeDirectory = Join-Path $taskRuntime 'home'
$taskTempDirectory = Join-Path $taskRuntime 'tmp'
$taskStdout = Join-Path $taskRuntime 'server.stdout.log'
$taskStderr = Join-Path $taskRuntime 'server.stderr.log'
$taskMetadataPath = Join-Path $taskRuntime 'service.json'
$taskServerUrl = 'http://127.0.0.1:11434'

if (-not (Test-Path -LiteralPath $taskExecutable -PathType Leaf)) {
    throw "Optional Ollama runtime is missing: $taskExecutable. Install the official standalone runtime before using this helper."
}

$taskEnvironment = @{
    USERPROFILE = $taskHomeDirectory
    TEMP = $taskTempDirectory
    TMP = $taskTempDirectory
    OLLAMA_HOST = '127.0.0.1:11434'
    OLLAMA_MODELS = $taskModels
    OLLAMA_NO_CLOUD = '1'
    OLLAMA_NOHISTORY = '1'
    OLLAMA_FLASH_ATTENTION = '1'
    OLLAMA_KV_CACHE_TYPE = 'q8_0'
    OLLAMA_NUM_PARALLEL = '1'
}
$taskBindings = @(Get-NetTCPConnection -State Listen -LocalPort 11434 -ErrorAction SilentlyContinue)
$taskServiceId = $null
$taskStartedAt = $null
$taskStatus = 'started'

if ($taskBindings.Count -gt 0) {
    $taskProcessIds = @($taskBindings | Select-Object -ExpandProperty OwningProcess -Unique)
    $taskOtherBindings = @($taskBindings | Where-Object LocalAddress -NE '127.0.0.1')
    if ($taskProcessIds.Count -ne 1 -or $taskOtherBindings.Count -gt 0) {
        throw 'Port 11434 has an unexpected listener. This helper will not stop or adopt it.'
    }
    if (-not (Test-Path -LiteralPath $taskMetadataPath -PathType Leaf)) {
        throw 'Port 11434 is already in use without workspace service metadata. This helper will not stop or adopt it.'
    }
    $taskSaved = Get-Content -LiteralPath $taskMetadataPath -Raw | ConvertFrom-Json
    $taskExistingProcess = Get-Process -Id $taskProcessIds[0]
    if ($taskSaved.Pid -ne $taskExistingProcess.Id -or
        -not [string]::Equals($taskExistingProcess.Path, $taskExecutable, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Port 11434 belongs to another process. This helper will not stop or adopt it.'
    }
    $taskHealth = Invoke-RestMethod -Uri "$taskServerUrl/api/version" -NoProxy -TimeoutSec 5
    if (-not $taskHealth.version) {
        throw 'The existing workspace service did not return a valid Ollama version.'
    }
    $taskServiceId = $taskExistingProcess.Id
    $taskStartedAt = $taskSaved.StartedAt
    $taskStatus = 'already_running'
} else {
    New-Item -ItemType Directory -Force -Path $taskRuntime, $taskModels, $taskHomeDirectory, $taskTempDirectory | Out-Null
    $taskService = Start-Process -FilePath $taskExecutable -ArgumentList 'serve' -WorkingDirectory $taskRuntime -Environment $taskEnvironment -WindowStyle Hidden -RedirectStandardOutput $taskStdout -RedirectStandardError $taskStderr -PassThru
    $taskServiceId = $taskService.Id
    $taskStartedAt = [DateTime]::UtcNow.ToString('o')
    [pscustomobject]@{
        Pid = $taskServiceId
        Executable = $taskExecutable
        StartedAt = $taskStartedAt
        ServerUrl = $taskServerUrl
        Status = 'starting'
    } | ConvertTo-Json | Set-Content -LiteralPath $taskMetadataPath -Encoding utf8
    $taskDeadline = [DateTime]::UtcNow.AddSeconds(120)
    $taskHealth = $null
    while ([DateTime]::UtcNow -lt $taskDeadline) {
        $taskService.Refresh()
        if ($taskService.HasExited) {
            throw "Ollama exited with code $($taskService.ExitCode). Check $taskStderr."
        }
        try {
            $taskHealth = Invoke-RestMethod -Uri "$taskServerUrl/api/version" -NoProxy -TimeoutSec 2
            if ($taskHealth.version) {
                break
            }
        } catch {
            Start-Sleep -Milliseconds 300
        }
    }
    if (-not $taskHealth -or -not $taskHealth.version) {
        throw "Ollama did not become healthy within 120 seconds. Check $taskStderr; process $taskServiceId was left available for diagnosis."
    }
    $taskStartedBindings = @(Get-NetTCPConnection -State Listen -LocalPort 11434 -ErrorAction SilentlyContinue)
    $taskUnexpectedBindings = @($taskStartedBindings | Where-Object {
        $_.OwningProcess -ne $taskServiceId -or $_.LocalAddress -ne '127.0.0.1'
    })
    if ($taskStartedBindings.Count -eq 0 -or $taskUnexpectedBindings.Count -gt 0) {
        throw "Ollama has an unexpected listener. Check process $taskServiceId and $taskStderr."
    }
}

$taskMetadata = [pscustomobject]@{
    Pid = $taskServiceId
    Version = $taskHealth.version
    Status = $taskStatus
    Executable = $taskExecutable
    Models = $taskModels
    Profile = $taskHomeDirectory
    Stdout = $taskStdout
    Stderr = $taskStderr
    StartedAt = $taskStartedAt
    ServerUrl = $taskServerUrl
    Environment = $taskEnvironment
}
$taskMetadata | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $taskMetadataPath -Encoding utf8
$taskMetadata | ConvertTo-Json -Depth 3 -Compress
