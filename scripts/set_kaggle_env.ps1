param(
    [string]$Username = "alexproger23",
    [string]$Key = "KGAT_c9612e657d1e16c3210cea192e15cb2e",
    [switch]$Persist
)

$ErrorActionPreference = "Stop"

if (-not $Username) {
    throw "Kaggle username is required."
}

if (-not $Key) {
    throw "Kaggle API key is required."
}

if ($Key -eq "KGAT_PASTE_YOUR_KEY_HERE") {
    throw "Replace KGAT_PASTE_YOUR_KEY_HERE with your real Kaggle API key."
}

$env:KAGGLE_USERNAME=$Username
$env:KAGGLE_API_TOKEN=$Key

Write-Host "KAGGLE_USERNAME and KAGGLE_KEY set for current PowerShell session."

