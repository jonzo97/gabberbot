@echo off
REM BMAD Music Production Expansion Pack Initialization
REM Provides the /init command functionality referenced in CLAUDE.md

echo.
echo 🎵 BMAD Music Production System Initialization
echo ===============================================
echo.

REM Run the BMAD initialization script
python bmad_init.py %*

REM Check if Python script was successful
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ BMAD initialization completed successfully!
    echo 🎯 Ready for hardcore music production with specialized agents!
    echo.
) else (
    echo.
    echo ❌ BMAD initialization failed!
    echo 💡 Check the error messages above and try again.
    echo.
    exit /b 1
)