# Test Data Format Specification

## Overview

This document describes the CSV format expected for test data in pyeventbt.

## Expected Format

CSV files should follow this structure (as exported from MetaTrader/ClickHouse):

| Column   | Type   | Description                    |
|----------|--------|--------------------------------|
| date     | string | Date in YYYY.MM.DD format      |
| time     | string | Time in HH:MM:SS format        |
| open     | float  | Opening price                  |
| high     | float  | Highest price in the period   |
| low      | float  | Lowest price in the period    |
| close    | float  | Closing price                  |
| tickvol  | int    | Number of ticks               |
| volume   | int    | Trading volume                 |
| spread   | int    | Spread in points               |

## Requirements

- **Header**: First line must contain column names
- **Encoding**: UTF-8
- **Delimiter**: Comma (,)
- **Date Format**: YYYY.MM.DD (separated by dots)
- **Time Format**: HH:MM:SS (24-hour format)

## Example

```csv
date,time,open,high,low,close,tickvol,volume,spread
2019.01.02,01:00:00,6332.8,6336.0,6327.5,6335.0,118,4768,91
2019.01.02,01:01:00,6333.9,6347.8,6333.9,6345.1,168,7086,52
```

## Usage in Tests

```python
import pandas as pd

def load_test_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(
        filepath,
        parse_dates={'date': '%Y.%m.%d'},
        encoding='utf-8'
    )
```

## Notes

- Compatible with MetaTrader 5 export format
- Compatible with ClickHouse export format
- The `date` and `time` columns should be combined for full datetime parsing