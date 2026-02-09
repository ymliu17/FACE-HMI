@echo off
cd /d "%~dp0"

:: Initialize conda (tries Miniconda first, then Anaconda)
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" 2>nul || call "%USERPROFILE%\anaconda3\Scripts\activate.bat" 2>nul

:: Activate target environment
call conda activate face

echo ==============================
echo      FACE2 API Launcher
echo ==============================
echo.

set /p SESSION=Enter session ID (example: 77777): 
set /p GROUP=Enter group index (0 = control, 1 = FACE): 

echo.
echo Executing: python src\face2_api.py %SESSION% %GROUP%
echo.

python src\face2_api.py %SESSION% %GROUP%

echo.
echo Finished. Press any key to exit...
pause >nul
