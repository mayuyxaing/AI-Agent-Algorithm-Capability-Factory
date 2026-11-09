@echo off
call "%~dp0check-openai.cmd" --run
exit /b %ERRORLEVEL%
