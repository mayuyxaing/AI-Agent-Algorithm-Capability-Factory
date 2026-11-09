@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "FORGE_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%FORGE_PYTHON%" goto ready
set "FORGE_PYTHON=%~dp0..\..\work\.venv\Scripts\python.exe"
if exist "%FORGE_PYTHON%" goto ready
set "FORGE_PYTHON=python"
:ready
"%FORGE_PYTHON%" scripts\run_glm.py %*
set "FORGE_RESULT=%ERRORLEVEL%"
echo.
echo Exit code: %FORGE_RESULT%
pause
exit /b %FORGE_RESULT%
