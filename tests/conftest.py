"""
PyEventBT Test Configuration
Shared fixtures for all tests.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from decimal import Decimal


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_ohlc():
    """Standard OHLC data for indicator tests."""
    return {
        'open':    np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109], dtype=np.float64),
        'high':    np.array([102, 104, 103, 105, 107, 106, 108, 110, 109, 111], dtype=np.float64),
        'low':     np.array([99,  100, 99,  101, 103, 102, 104, 106, 105, 107], dtype=np.float64),
        'close':   np.array([101, 103, 102, 104, 106, 105, 107, 109, 108, 110], dtype=np.float64),
        'volume':  np.array([1000, 1200, 900, 1100, 1300, 1000, 1200, 1400, 1100, 1200], dtype=np.float64),
    }


@pytest.fixture
def sample_ohlc_long():
    """Long OHLC data for indicators requiring more bars (30 bars)."""
    base = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                    110, 108, 109, 111, 113, 112, 114, 116, 115, 117,
                    118, 120, 119, 121, 123, 122, 124, 126, 125, 127], dtype=np.float64)
    return {
        'open':   base - 1,
        'high':   base + 2,
        'low':    base - 3,
        'close':  base,
        'volume': np.array([1000]*30, dtype=np.float64),
    }


@pytest.fixture
def sample_bar():
    """Sample bar data for events tests."""
    return {
        'open': 100,
        'high': 105,
        'low': 98,
        'close': 103,
        'tickvol': 100,
        'volume': 50000,
        'spread': 10,
        'digits': 1,
    }


@pytest.fixture
def sample_datetime():
    """Sample datetime for events tests."""
    return datetime(2024, 1, 15, 10, 30, 0)


@pytest.fixture
def sample_decimal_balance():
    """Sample decimal balance for portfolio tests."""
    return Decimal('10000.00')