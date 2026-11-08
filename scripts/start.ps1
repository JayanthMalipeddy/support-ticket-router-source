$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not $env:ROUTER_API_TOKEN) { throw 'Set ROUTER_API_TOKEN to a private random token first.' }
& .\.venv\Scripts\python.exe -m uvicorn ticket_router.api:app --host 127.0.0.1 --port 8025
