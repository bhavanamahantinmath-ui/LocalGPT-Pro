@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo Starting LocalGPT Pro (Multi-User Local AI Assistant)
echo ========================================================

:: Route Streamlit home to workspace .home to avoid permission errors on ~/.streamlit
set "USERPROFILE=%~dp0..\.home"
set "STREAMLIT_SERVER_SHOW_EMAIL_PROMPT=false"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"

set "PY_EXE=C:\Users\BHAVANA M K\AppData\Local\Programs\Python\Python314\python.exe"
if exist "%PY_EXE%" goto RUN_PY

for /f "tokens=*" %%i in ('where python 2^>nul') do (
    "%%i" --version >nul 2>nul
    if not errorlevel 1 (
        set "PY_EXE=%%i"
        goto RUN_PY
    )
)

echo [ERROR] Working Python interpreter could not be found!
pause
exit /b 1

:RUN_PY
echo Using Python at: "%PY_EXE%"
echo Launching LocalGPT Pro at http://localhost:8501 ...

:: Open browser automatically as soon as port 8501 accepts connections
start "" powershell -WindowStyle Hidden -Command "$s = New-Object System.Net.Sockets.TcpClient; for ($i=0; $i -lt 60; $i++) { Start-Sleep -Milliseconds 500; try { $s.Connect('127.0.0.1', 8501); if ($s.Connected) { Start-Process 'http://localhost:8501'; break } } catch {} }"

"%PY_EXE%" -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --browser.gatherUsageStats false
pause
