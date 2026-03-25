@echo off
REM APEX Running Analysis - Full Pipeline Launcher (Windows Batch)
REM Simplified execution of the full pipeline

setlocal enabledelayedexpansion

REM Get script directory
set SCRIPT_DIR=%~dp0

REM Display header
cls
echo.
echo ================================================================================
echo  APEX Running Analysis - Full Pipeline Orchestrator
echo ================================================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8+ and add it to your system PATH
    pause
    exit /b 1
)

REM Parse command line arguments
set SKIP_ETL=0
set SKIP_DASHBOARD=0
set VERBOSE=0

:parse_args
if "%1"=="" goto args_done
if "%1"=="--skip-etl" set SKIP_ETL=1
if "%1"=="--skip-dashboard" set SKIP_DASHBOARD=1
if "%1"=="--verbose" set VERBOSE=1
if "%1"=="-h" goto show_help
if "%1"=="--help" goto show_help
shift
goto parse_args

:args_done
REM Build command
set CMD=python "%SCRIPT_DIR%run_full_pipeline.py"
if %SKIP_ETL%==1 set CMD=!CMD! --skip-etl
if %SKIP_DASHBOARD%==1 set CMD=!CMD! --skip-dashboard
if %VERBOSE%==1 set CMD=!CMD! --verbose

REM Run pipeline
echo [INFO] Starting pipeline...
echo.
call !CMD!

if errorlevel 1 (
    echo.
    echo [ERROR] Pipeline failed
    pause
    exit /b 1
)

exit /b 0

:show_help
echo Usage: run_pipeline.bat [OPTIONS]
echo.
echo Options:
echo   --skip-etl          Skip ETL step (use existing Apex.db)
echo   --skip-dashboard    Skip dashboard launch
echo   --verbose           Show detailed output
echo   --help              Show this help message
echo.
echo Examples:
echo   run_pipeline.bat
echo   run_pipeline.bat --skip-etl
echo   run_pipeline.bat --skip-etl --skip-dashboard
echo.
pause
exit /b 0
