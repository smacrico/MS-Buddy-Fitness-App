#!/usr/bin/env python3
"""
APEX Running Analysis - Pipeline Quick Start Guide
Run this script to install dependencies and verify setup
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8+"""
    print("\n[1/4] Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  ✗ Python 3.8+ required, but found {version.major}.{version.minor}")
        return False
    print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_required_packages():
    """Check and install required packages"""
    print("\n[2/4] Checking Python packages...")
    
    required = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'plotly': 'plotly',
        'streamlit': 'streamlit',
    }
    
    missing = []
    for package_name, import_name in required.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - missing")
            missing.append(package_name)
    
    if missing:
        print(f"\n  Installing missing packages: {', '.join(missing)}")
        cmd = [sys.executable, '-m', 'pip', 'install'] + missing
        try:
            subprocess.run(cmd, check=True)
            print("  ✓ Packages installed")
            return True
        except subprocess.CalledProcessError:
            print("  ✗ Failed to install packages")
            print("    Try manually: pip install " + " ".join(missing))
            return False
    
    return True

def check_required_files():
    """Check if required scripts exist"""
    print("\n[3/4] Checking required files...")
    
    script_dir = Path(__file__).parent
    required_files = [
        'run_full_pipeline.py',
        'createRunAnalDB - v6.26.py',
        'RunningAnalysis_v6.26.py',
        'app.py',
        'test_metrics_db.py',
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = script_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} - missing")
            all_exist = False
    
    return all_exist

def check_databases():
    """Check if required database paths exist"""
    print("\n[4/4] Checking databases...")
    
    paths = {
        'artemis.db': r'c:/smakrykoDBs/artemis.db',
        'garmin_activities.db': r'c:/smakrykoDBs/garmin_activities.db',
    }
    
    all_exist = True
    for name, path in paths.items():
        db_path = Path(path)
        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {name} ({size_mb:.1f} MB)")
        else:
            print(f"  ⚠ {name} - not found at {path}")
            all_exist = False
    
    return all_exist

def print_next_steps():
    """Print instructions for running the pipeline"""
    print("\n" + "="*75)
    print("SETUP COMPLETE! Next Steps:")
    print("="*75)
    
    print("\n1. Run the full pipeline:")
    print("     Windows: run_pipeline.bat")
    print("     Linux/Mac: ./run_pipeline.sh")
    print("     Or: python run_full_pipeline.py")
    
    print("\n2. If ETL fails (databases not found):")
    print("     - Check artemis.db and garmin_activities.db exist")
    print("     - Update paths in createRunAnalDB - v6.26.py if needed")
    print("     - Or run: python run_full_pipeline.py --skip-etl")
    
    print("\n3. Access the dashboard:")
    print("     Open browser to: http://localhost:8501")
    
    print("\n4. Customize the pipeline:")
    print("     - Edit pipeline_config.ini for settings")
    print("     - Edit RunningAnalysis_v6.26.py for user metrics")
    
    print("\n" + "="*75)
    print("Documentation: See PIPELINE_README.md for detailed information")
    print("="*75 + "\n")

def main():
    """Run all checks"""
    print("\n" + "="*75)
    print("APEX Running Analysis - Pipeline Setup Verification")
    print("="*75)
    
    checks = [
        ("Python Version", check_python_version),
        ("Python Packages", check_required_packages),
        ("Required Files", check_required_files),
        ("Databases", check_databases),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Check failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*75)
    print("SETUP SUMMARY")
    print("="*75)
    
    critical_checks = results[:2]  # Python version and packages are critical
    optional_checks = results[2:]  # Files and databases are informational
    
    critical_passed = all(r[1] for r in critical_checks)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    if not critical_passed:
        print("\n✗ Critical checks failed. Cannot proceed.")
        print("Please fix the issues above and try again.")
        return False
    
    print("\n✓ Setup verification complete!")
    print_next_steps()
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
