@echo off
setlocal
set PY312=C:\venvs\gan312\Scripts\python.exe
if not exist "%PY312%" (
  echo [ERROR] Missing interpreter: %PY312%
  echo Create it first, then retry.
  exit /b 1
)
"%PY312%" "%~dp0main.py" %*
endlocal
