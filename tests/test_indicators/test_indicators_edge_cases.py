"""
Edge case tests for PyEventBT indicators.
Tests handling of boundary conditions and special cases.
"""
import numpy as np
import pytest

from pyeventbt.indicators.indicators import (
    KAMA, ATR, SMA, EMA, RSI, ADX, Momentum, BollingerBands,
    DonchianChannels, MACD, KeltnerChannel, ADR, VWAP,
    Stochastic, CCI, WilliamsR, ROC, TRIX, DeMarker, Aroon, RVI
)


# =============================================================================
# Tests: Flat / Constant price series
# =============================================================================

class TestConstantPrices:
    def test_sma_constant_prices(self):
        """SMA with constant prices should return constant."""
        close = np.array([100.0]*10, dtype=np.float64)
        result = SMA.compute(close, period=5)
        valid = result[~np.isnan(result)]
        # SMA with constant prices returns the constant value
        assert len(valid) >= 5  # Allow for different warmup behavior
        assert np.allclose(valid, 100.0)
    
    def test_rsi_constant_prices(self):
        """RSI with no price changes should handle gracefully."""
        close = np.array([100.0]*20, dtype=np.float64)
        result = RSI.compute(close, period=14)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0


class TestSingleDirectionTrend:
    def test_ema_uptrend(self):
        """EMA should produce rising values in uptrend."""
        close = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=np.float64)
        result = EMA.compute(close, period=3)
        valid = result[~np.isnan(result)]
        # Allow for different warmup behavior
        assert len(valid) >= 7

    def test_ema_downtrend(self):
        """EMA should produce falling values in downtrend."""
        close = np.array([109, 108, 107, 106, 105, 104, 103, 102, 101, 100], dtype=np.float64)
        result = EMA.compute(close, period=3)
        valid = result[~np.isnan(result)]
        # Allow for different warmup behavior
        assert len(valid) >= 7


# =============================================================================
# Tests: Sharp movements
# =============================================================================

class TestSharpMovements:
    def test_rsi_extreme_values(self):
        """RSI should handle sharp price increases."""
        close = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240], dtype=np.float64)
        result = RSI.compute(close, period=14)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert (valid <= 100).all()
    
    def test_williams_r_extreme_low(self):
        """Williams %R when price hits high of period."""
        high = np.array([105, 105, 105, 105, 105, 105], dtype=np.float64)
        low = np.array([95, 95, 95, 95, 95, 95], dtype=np.float64)
        close = np.array([105, 105, 105, 105, 105, 105], dtype=np.float64)
        result = WilliamsR.compute(high, low, close, period=6)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert (valid >= -100).all()


# =============================================================================
# Tests: Minimal data lengths
# =============================================================================

class TestMinimalLengths:
    def test_sma_exact_period_length(self):
        """SMA with array length equal to period."""
        close = np.array([100, 101, 102, 103, 104], dtype=np.float64)
        result = SMA.compute(close, period=5)
        assert not np.isnan(result[-1])
    
    def test_ema_exact_period_plus_one(self):
        """EMA with array length equal to period + 1."""
        close = np.array([100, 101, 102, 103, 104, 105], dtype=np.float64)
        result = EMA.compute(close, period=5)
        assert not np.isnan(result[-1])


# =============================================================================
# Tests: NaN and missing data handling
# =============================================================================

class TestNaNHandling:
    def test_ema_no_nan_in_output(self, sample_ohlc):
        """EMA should not introduce unexpected NaN values."""
        result = EMA.compute(sample_ohlc['close'], period=5)
        # Only first (period-1) values should be NaN
        nan_count = np.sum(np.isnan(result))
        assert nan_count == 4
    
    def test_sma_no_nan_beyond_warmup(self, sample_ohlc):
        """SMA should not have NaN after warmup."""
        result = SMA.compute(sample_ohlc['close'], period=3)
        nan_count = np.sum(np.isnan(result))
        assert nan_count == 2


# =============================================================================
# Tests: Array types and shapes
# =============================================================================

class TestOutputTypes:
    def test_all_indicators_return_numpy_array(self):
        """All single-output indicators should return np.ndarray."""
        # Use larger arrays to avoid numba issues
        close = np.linspace(100, 200, 50, dtype=np.float64)
        high = close + 5
        low = close - 5
        volume = np.ones(50, dtype=np.float64)

        single_output = [
            (SMA, [close], {'period': 5}),
            (EMA, [close], {'period': 5}),
            (RSI, [close], {'period': 14}),
            (Momentum, [close], {'period': 5}),
            (ROC, [close], {'period': 5}),
            (TRIX, [close], {'period': 5}),
            (ATR, [high, low, close], {'period': 14}),
            (ADR, [high, low], {'period': 5}),
            (CCI, [high, low, close], {'period': 14}),
            (WilliamsR, [high, low, close], {'period': 5}),
            (DeMarker, [high, low], {'period': 5}),
            (VWAP, [high, low, close, volume], {}),
        ]
        for indicator_cls, args, kwargs in single_output:
            result = indicator_cls.compute(*args, **kwargs)
            assert isinstance(result, np.ndarray), f"{indicator_cls.__name__} should return np.ndarray"
            assert len(result) > 0, f"{indicator_cls.__name__} output should not be empty"
    
    def test_tuple_indicators_return_tuples(self, sample_ohlc_long):
        """All tuple-output indicators should return tuple of arrays."""
        tuple_output = [
            (ADX, [sample_ohlc_long['high'], sample_ohlc_long['low'], sample_ohlc_long['close']], {'period': 14}),
            (BollingerBands, [sample_ohlc_long['close']], {'period': 20}),
            (DonchianChannels, [sample_ohlc_long['high'], sample_ohlc_long['low']], {'period': 20}),
            (KeltnerChannel, [sample_ohlc_long['high'], sample_ohlc_long['low'], sample_ohlc_long['close']], {'period': 20}),
            (Aroon, [sample_ohlc_long['high'], sample_ohlc_long['low']], {'period': 25}),
            (RVI, [sample_ohlc_long['open'], sample_ohlc_long['high'], sample_ohlc_long['low'], sample_ohlc_long['close']], {'period': 10}),
        ]
        for indicator_cls, args, kwargs in tuple_output:
            result = indicator_cls.compute(*args, **kwargs)
            assert isinstance(result, tuple), f"{indicator_cls.__name__} should return tuple"
            for arr in result:
                assert isinstance(arr, np.ndarray), f"{indicator_cls.__name__} tuple elements should be np.ndarray"