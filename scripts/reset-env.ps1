Write-Host "🔄 Resetting virtual environment..." -ForegroundColor Cyan

# Deactivate if active
if (Get-Command deactivate -ErrorAction SilentlyContinue) {
    deactivate
}

# Remove existing .venv
if (Test-Path ".venv") {
    Write-Host "🗑 Removing existing .venv..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .venv
}

# Create new venv
Write-Host "🐍 Creating new virtual environment..." -ForegroundColor Green
python -m venv .venv

# Activate venv
Write-Host "⚡ Activating virtual environment..." -ForegroundColor Green
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "⬆ Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# Install project with dev deps
Write-Host "📦 Installing project (editable + dev)..." -ForegroundColor Green
pip install -e ".[dev]"

# Final check
Write-Host "✅ Checking installation..." -ForegroundColor Green
python -c "import sys; print('Python:', sys.executable)"
python -c "import mlb; print('MLB version:', mlb.__version__)"

Write-Host "🎉 Environment reset complete!" -ForegroundColor Cyan

"""
   В PowerShell:
.\scripts\reset-env.ps1

   Если будет ошибка политики выполнения:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
"""