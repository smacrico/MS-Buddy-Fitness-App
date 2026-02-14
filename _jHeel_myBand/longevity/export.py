"""
Data export module for longevity metrics.
"""

from __future__ import annotations
import pandas as pd
import json
from pathlib import Path
from typing import Union, Dict, Any, Optional
import logging


logger = logging.getLogger(__name__)


def export_to_csv(
    data: Union[pd.DataFrame, Dict[str, Any]],
    output_path: Union[str, Path],
    index: bool = False
) -> bool:
    """
    Export data to CSV format.
    
    Args:
        data: DataFrame or dict to export
        output_path: Output file path
        index: Whether to include index column
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=index)
        logger.info(f"Exported {len(df)} records to CSV: {output_path}")
        print(f"✓ Exported to CSV: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to export CSV: {e}")
        print(f"✗ Export failed: {e}")
        return False


def export_to_json(
    data: Union[pd.DataFrame, Dict[str, Any]],
    output_path: Union[str, Path],
    orient: str = 'records',
    indent: int = 2
) -> bool:
    """
    Export data to JSON format.
    
    Args:
        data: DataFrame or dict to export
        output_path: Output file path
        orient: JSON orientation ('records', 'index', 'columns', etc.)
        indent: JSON indentation level
    
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            data.to_json(output_path, orient=orient, indent=indent)
        else:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=indent, default=str)
        
        logger.info(f"Exported to JSON: {output_path}")
        print(f"✓ Exported to JSON: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to export JSON: {e}")
        print(f"✗ Export failed: {e}")
        return False


def export_to_excel(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    output_path: Union[str, Path],
    sheet_name: str = 'Sheet1'
) -> bool:
    """
    Export data to Excel format.
    
    Args:
        data: DataFrame or dict of DataFrames (for multiple sheets)
        output_path: Output file path
        sheet_name: Sheet name (if single DataFrame)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, dict):
            # Multiple sheets
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for name, df in data.items():
                    if isinstance(df, pd.DataFrame):
                        df.to_excel(writer, sheet_name=name, index=False)
        else:
            # Single sheet
            data.to_excel(output_path, sheet_name=sheet_name, index=False)
        
        logger.info(f"Exported to Excel: {output_path}")
        print(f"✓ Exported to Excel: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to export Excel: {e}")
        print(f"✗ Export failed: {e}")
        return False


def export_to_parquet(
    data: pd.DataFrame,
    output_path: Union[str, Path],
    compression: str = 'snappy'
) -> bool:
    """
    Export data to Parquet format (efficient for large datasets).
    
    Args:
        data: DataFrame to export
        output_path: Output file path
        compression: Compression algorithm ('snappy', 'gzip', 'brotli', None)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_parquet(output_path, compression=compression, index=False)
        
        logger.info(f"Exported to Parquet: {output_path}")
        print(f"✓ Exported to Parquet: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to export Parquet: {e}")
        print(f"✗ Export failed: {e}")
        return False


def export_metrics_summary(
    metrics: Dict[str, Any],
    output_dir: Union[str, Path],
    formats: list = ['csv', 'json']
) -> Dict[str, bool]:
    """
    Export metrics to multiple formats.
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Output directory
        formats: List of formats ('csv', 'json', 'excel', 'parquet')
    
    Returns:
        Dictionary of format: success status
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Convert to DataFrame if dict
    if isinstance(metrics, dict) and not isinstance(metrics, pd.DataFrame):
        # Check if it's a simple dict or nested
        if all(isinstance(v, (list, tuple)) for v in metrics.values()):
            df = pd.DataFrame(metrics)
        else:
            df = pd.DataFrame([metrics])
    else:
        df = metrics
    
    for fmt in formats:
        if fmt == 'csv':
            results['csv'] = export_to_csv(df, output_dir / 'metrics.csv')
        elif fmt == 'json':
            results['json'] = export_to_json(df, output_dir / 'metrics.json')
        elif fmt == 'excel':
            results['excel'] = export_to_excel(df, output_dir / 'metrics.xlsx')
        elif fmt == 'parquet':
            results['parquet'] = export_to_parquet(df, output_dir / 'metrics.parquet')
    
    return results


def create_report(
    metrics: Dict[str, Any],
    output_path: Union[str, Path],
    title: str = "Longevity Metrics Report"
) -> bool:
    """
    Create a formatted text report.
    
    Args:
        metrics: Dictionary of metrics to report
        output_path: Output file path
        title: Report title
    
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(f"{title}\n")
            f.write("=" * len(title) + "\n\n")
            
            for key, value in metrics.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for subkey, subvalue in value.items():
                        f.write(f"  {subkey}: {subvalue}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            f.write(f"\n{'=' * len(title)}\n")
            f.write("Report generated successfully\n")
        
        logger.info(f"Created report: {output_path}")
        print(f"✓ Created report: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to create report: {e}")
        print(f"✗ Report creation failed: {e}")
        return False
