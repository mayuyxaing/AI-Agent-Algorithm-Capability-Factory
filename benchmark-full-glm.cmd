@echo off
call "%~dp0check-glm.cmd" --full-benchmark
exit /b %ERRORLEVEL%
