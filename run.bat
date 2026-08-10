@echo off
REM run.bat — double-click this to launch everything.
REM Activates the venv, then runs start.py (which launches llama-server
REM and the agent server together).

cd /d "%~dp0"
call venv\Scripts\activate.bat
python start.py

REM Keep the window open if something crashes, so you can read the error
REM instead of the window closing instantly.
pause