@echo off
call "%~dp0check-glm.cmd" --run --repair-demo
exit /b %ERRORLEVEL%
