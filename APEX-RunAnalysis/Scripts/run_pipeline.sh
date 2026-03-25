#!/bin/bash
# APEX Running Analysis - Full Pipeline Launcher (Unix/Linux/macOS)
# Simplified execution of the full pipeline

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Display header
clear
echo ""
echo "================================================================================"
echo " APEX Running Analysis - Full Pipeline Orchestrator"
echo "================================================================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH"
    echo ""
    echo "Please install Python 3.8+ using your package manager:"
    echo "  - macOS: brew install python3"
    echo "  - Ubuntu/Debian: sudo apt-get install python3"
    echo "  - Fedora: sudo dnf install python3"
    exit 1
fi

# Parse command line arguments
SKIP_ETL=0
SKIP_DASHBOARD=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-etl)
            SKIP_ETL=1
            shift
            ;;
        --skip-dashboard)
            SKIP_DASHBOARD=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Build command
CMD="python3 \"$SCRIPT_DIR/run_full_pipeline.py\""
[[ $SKIP_ETL -eq 1 ]] && CMD="$CMD --skip-etl"
[[ $SKIP_DASHBOARD -eq 1 ]] && CMD="$CMD --skip-dashboard"
[[ $VERBOSE -eq 1 ]] && CMD="$CMD --verbose"

# Run pipeline
echo "[INFO] Starting pipeline..."
echo ""

eval $CMD
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "[ERROR] Pipeline failed with exit code $EXIT_CODE"
    exit 1
fi

exit 0

show_help() {
    echo "Usage: run_pipeline.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --skip-etl          Skip ETL step (use existing Apex.db)"
    echo "  --skip-dashboard    Skip dashboard launch"
    echo "  --verbose           Show detailed output"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_pipeline.sh"
    echo "  ./run_pipeline.sh --skip-etl"
    echo "  ./run_pipeline.sh --skip-etl --skip-dashboard"
    echo ""
}
