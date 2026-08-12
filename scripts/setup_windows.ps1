[CmdletBinding()]
param(
    [switch]$InstallTools,
    [switch]$AllVoices,
    [string[]]$Voice,
    [switch]$WithBookNLP
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$locationPushed = $false

function Find-Application {
    param([Parameter(Mandatory)][string]$Name)

    $command = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $command) {
        return $null
    }
    return $command.Path
}

function Refresh-ProcessPath {
    $currentPath = [string]$env:Path
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $parts = @($currentPath -split ";") + @([string]$userPath -split ";") + @([string]$machinePath -split ";")
    $env:Path = ($parts | Where-Object { $_ } | Select-Object -Unique) -join ";"
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter()][string[]]$Arguments = @()
    )

    Write-Host (">> {0} {1}" -f $FilePath, ($Arguments -join " "))
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath"
    }
}

function Test-VersionCommand {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string]$Label
    )

    $output = & $FilePath -version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "$Label is present but could not run -version."
    }
    $firstLine = $output | Select-Object -First 1
    Write-Host ("{0}: {1}" -f $Label, $firstLine)
}

try {
    $root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    if ($AllVoices -and $null -ne $Voice -and $Voice.Count -gt 0) {
        throw "Use either -AllVoices or -Voice, not both."
    }
    $expectedFiles = @(
        "pyproject.toml",
        "uv.lock",
        "benchmark\environments\kokoro\pyproject.toml",
        "scripts\download_kokoro_assets.py"
    )
    if ($WithBookNLP) {
        $expectedFiles += "analyzer_envs\booknlp\pyproject.toml"
    }
    foreach ($relativePath in $expectedFiles) {
        if (-not (Test-Path (Join-Path $root $relativePath) -PathType Leaf)) {
            throw "Expected project file is missing: $relativePath"
        }
    }

    Push-Location $root
    $locationPushed = $true
    Write-Host "Repository: $root"

    $uv = Find-Application "uv"
    if ($null -eq $uv) {
        if (-not $InstallTools) {
            Write-Host "uv was not found on PATH. Re-run with -InstallTools, or install it with the official command:"
            Write-Host 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
            throw "uv is required to continue."
        }
        $winget = Find-Application "winget"
        if ($null -eq $winget) {
            throw "WinGet is unavailable. Install uv manually from https://docs.astral.sh/uv/getting-started/installation/ and re-run."
        }
        Invoke-CheckedCommand $winget @(
            "install", "--id", "astral-sh.uv", "--exact", "--source", "winget",
            "--accept-source-agreements", "--accept-package-agreements"
        )
        Refresh-ProcessPath
        $uv = Find-Application "uv"
        if ($null -eq $uv) {
            throw "WinGet installed uv, but uv is still not on this process PATH. Start a new PowerShell window and re-run."
        }
    }
    Write-Host "uv: $uv"
    Invoke-CheckedCommand $uv @("--version")

    $ffmpeg = Find-Application "ffmpeg"
    $ffprobe = Find-Application "ffprobe"
    if ($null -eq $ffmpeg -or $null -eq $ffprobe) {
        if (-not $InstallTools) {
            if ($null -eq $ffmpeg) {
                $configuredFfmpeg = [Environment]::GetEnvironmentVariable("PDF_AUDIOBOOK_FFMPEG")
                if (-not [string]::IsNullOrWhiteSpace($configuredFfmpeg) -and [System.IO.File]::Exists($configuredFfmpeg)) {
                    $ffmpeg = (Resolve-Path -LiteralPath $configuredFfmpeg).Path
                    Write-Host "Using PDF_AUDIOBOOK_FFMPEG: $ffmpeg"
                }
            }
            if ($null -eq $ffprobe) {
                $configuredFfprobe = [Environment]::GetEnvironmentVariable("PDF_AUDIOBOOK_FFPROBE")
                if (-not [string]::IsNullOrWhiteSpace($configuredFfprobe) -and [System.IO.File]::Exists($configuredFfprobe)) {
                    $ffprobe = (Resolve-Path -LiteralPath $configuredFfprobe).Path
                    Write-Host "Using PDF_AUDIOBOOK_FFPROBE: $ffprobe"
                }
            }
            if ($null -ne $ffmpeg -and $null -ne $ffprobe) {
                Write-Host "Using configured FFmpeg executables."
            }
        }
        if ($null -eq $ffmpeg -or $null -eq $ffprobe) {
            if (-not $InstallTools) {
                Write-Host "FFmpeg and ffprobe must both be available on PATH or configured as existing executable files."
                Write-Host "Manual Windows downloads: https://www.gyan.dev/ffmpeg/builds/ or https://ffmpeg.org/download.html"
                Write-Host 'After extraction, add the bin directory to PATH, or set:'
                Write-Host '$env:PDF_AUDIOBOOK_FFMPEG = ''C:\path\to\ffmpeg.exe'''
                Write-Host '$env:PDF_AUDIOBOOK_FFPROBE = ''C:\path\to\ffprobe.exe'''
                throw "ffmpeg and/or ffprobe was not found."
            }
            $winget = Find-Application "winget"
            if ($null -eq $winget) {
                throw "WinGet is unavailable. Install FFmpeg manually from https://www.gyan.dev/ffmpeg/builds/ and ensure ffmpeg.exe and ffprobe.exe are available."
            }
            Invoke-CheckedCommand $winget @(
                "install", "--id", "Gyan.FFmpeg", "--exact", "--source", "winget",
                "--accept-source-agreements", "--accept-package-agreements"
            )
            Refresh-ProcessPath
            $ffmpeg = Find-Application "ffmpeg"
            $ffprobe = Find-Application "ffprobe"
        }
    }
    if ($null -eq $ffmpeg -or $null -eq $ffprobe) {
        throw "FFmpeg installation completed, but both ffmpeg and ffprobe are not available on PATH."
    }
    Test-VersionCommand $ffmpeg "ffmpeg"
    Test-VersionCommand $ffprobe "ffprobe"

    Invoke-CheckedCommand $uv @("python", "install", "3.11")
    Invoke-CheckedCommand $uv @("sync", "--group", "test")
    Invoke-CheckedCommand $uv @("sync", "--project", "benchmark/environments/kokoro", "--python", "3.11")
    if ($WithBookNLP) {
        Write-Host "Installing optional Interactive Voices analyzer (BookNLP)..."
        Invoke-CheckedCommand $uv @("sync", "--project", "analyzer_envs/booknlp", "--python", "3.11")
    }

    $rootPython = Join-Path $root ".venv\Scripts\python.exe"
    $kokoroPython = Join-Path $root "benchmark\environments\kokoro\.venv\Scripts\python.exe"
    if (-not (Test-Path $rootPython -PathType Leaf)) {
        throw "Root environment was not created: $rootPython"
    }
    if (-not (Test-Path $kokoroPython -PathType Leaf)) {
        throw "Kokoro environment was not created: $kokoroPython"
    }

    Invoke-CheckedCommand $rootPython @(
        "-c",
        "import sys, fastapi, pdfplumber, pypdf, pytest; print(f'root Python {sys.version.split()[0]}: packages ready')"
    )
    Invoke-CheckedCommand $kokoroPython @(
        "-c",
        "import sys, importlib.metadata as metadata; from kokoro import KPipeline; print(f'Kokoro Python {sys.version.split()[0]}: kokoro {metadata.version(`"kokoro`")}, KPipeline ready')"
    )
    if ($WithBookNLP) {
        $bookNlpPython = Join-Path $root "analyzer_envs\booknlp\.venv\Scripts\python.exe"
        if (-not (Test-Path $bookNlpPython -PathType Leaf)) {
            throw "BookNLP environment was not created: $bookNlpPython"
        }
        Invoke-CheckedCommand $bookNlpPython @(
            "-c",
            "import sys, importlib.metadata as metadata; print(f'BookNLP Python {sys.version.split()[0]}: booknlp {metadata.version(`"booknlp`")}, transformers {metadata.version(`"transformers`")}')"
        )
    }

    $assetArguments = @()
    if ($AllVoices) {
        Write-Host "Downloading all approved Kokoro voices (large download)."
        $assetArguments += "--all"
    } elseif ($null -ne $Voice -and $Voice.Count -gt 0) {
        Write-Host ("Downloading selected Kokoro voices: {0}" -f ($Voice -join ", "))
        foreach ($voiceId in $Voice) {
            $assetArguments += @("--voice", $voiceId)
        }
    } else {
        Write-Host "Downloading the default Kokoro voice: af_heart"
        $assetArguments += @("--voice", "af_heart")
    }
    $assetHelper = Join-Path $root "scripts\download_kokoro_assets.py"
    $assetArgumentsWithScript = @($assetHelper) + $assetArguments
    Invoke-CheckedCommand $kokoroPython $assetArgumentsWithScript

    Write-Host ""
    Write-Host "Setup complete. The root and isolated Kokoro environments are ready."
    Write-Host "Next: set PDF_AUDIOBOOK_KOKORO_PYTHON only if you need a non-default Kokoro interpreter."
    Write-Host "Next: use the project README for the normal application launch command."
    if (-not $WithBookNLP) {
        Write-Host "Interactive Voices remains optional; re-run with -WithBookNLP when needed."
    }
    if (-not $AllVoices) {
        Write-Host "Additional voice assets remain optional; re-run with -AllVoices or -Voice <id>."
    }
} catch {
    Write-Error ("Setup failed: {0}" -f $_.Exception.Message)
    exit 1
} finally {
    if ($locationPushed) {
        Pop-Location
    }
}
