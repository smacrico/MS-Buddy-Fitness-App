#!/usr/bin/env python3
# Debug script to log ALL field names from session messages in FIT files
"""Debug FIT session fields - stelios (c) 2026"""

import os
import logging
from fitparse import FitFile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('c:/temp/logsFitnessApp/fit_session_debug.log'),
        logging.StreamHandler()
    ]
)

def debug_fit_session_fields(folder_path):
    """Log all field names from session messages in all .fit files"""
    print(f"🔍 Scanning folder: {folder_path}")
    logging.info(f"Starting debug scan of {folder_path}")
    
    for filename
