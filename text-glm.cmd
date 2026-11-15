@echo off
call "%~dp0check-glm.cmd" --run --task text_classification
exit /b %ERRORLEVEL%
