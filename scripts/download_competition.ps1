$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$downloadDir = Join-Path $root "data\raw"
New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null

kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -p $downloadDir
