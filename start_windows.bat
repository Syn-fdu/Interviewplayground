@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo InterviewPlayground v1.9.3 - Interview and Evaluation
echo Starting the local research workspace...
python server.py --open-browser %*
if errorlevel 1 (
  echo.
  echo Startup failed. Verify Python, the server port, and your active provider settings.
)
pause
