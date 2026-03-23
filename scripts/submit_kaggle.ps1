$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python scripts\prepare_kaggle_kernel.py
kaggle kernels push -p build\kaggle_kernel
