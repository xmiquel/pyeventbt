"""
Data loader for PyEventBT tests.
Loads CSV data for integration tests.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


# Root of the tests directory
_TESTS_DIR = Path(__file__).parent.parent.resolve()


def load_ndx_csv(nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load NDX.csv test data.
    
    The CSV format is:
        date,time,open,high,low,close,tickvol,volume,spread
    
    Args:
        nrows: Number of rows to load. If None, loads all.
    
    Returns:
        DataFrame with the NDX data.
    
    Note:
        The actual NDX.csv file is gitignored. This function uses the example file
        for documentation purposes. Tests should use fixture data or mock data.
    """
    csv_path = _TESTS_DIR / "data" / "NDX.csv.example"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Test data not found: {csv_path}")
    
    df = pd.read_csv(csv_path, nrows=nrows)
    return df


def load_csv_as_arrays(csv_path: str | Path, nrows: Optional[int] = None) -> dict:
    """
    Load CSV data and return as numpy arrays.
    
    Args:
        csv_path: Path to the CSV file.
        nrows: Number of rows to load.
    
    Returns:
        Dictionary with 'open', 'high', 'low', 'close', 'volume' arrays.
    """
    df = pd.read_csv(csv_path, nrows=nrows)
    
    return {
        'open':   df['open'].values.astype(float),
        'high':   df['high'].values.astype(float),
        'low':    df['low'].values.astype(float),
        'close':  df['close'].values.astype(float),
        'volume': df['volume'].values.astype(float),
    }