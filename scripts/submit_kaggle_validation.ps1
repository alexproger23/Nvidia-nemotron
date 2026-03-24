$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python scripts\prepare_kaggle_validation_kernel.py
kaggle kernels push -p build\kaggle_validation_kernel
