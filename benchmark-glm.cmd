@echo off
call "%~dp0check-glm.cmd" --benchmark
exit /b %ERRORLEVEL%
