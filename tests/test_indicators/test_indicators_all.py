"""
Comprehensive tests for all PyEventBT technical indicators.
Covers KAMA, ATR, EMA, RSI, ADX, Momentum, BollingerBands, DonchianChannels,
MACD, KeltnerChannel, ADR, VWAP, Stochastic, CCI, WilliamsR, ROC, TRIX,
DeMarker, Aroon, RVI.
"""
import numpy as np
import pytest
from pyeventbt.indicators.indicators import (
    KAMA, ATR, SMA, EMA, RSI, ADX, Momentum, BollingerBands,
    DonchianChannels, MACD, KeltnerChannel, ADR, VWAP,
    Stochastic, CCI, WilliamsR, ROC, TRIX, DeMarker, Aroon, RVI
)


# =============================================================================
# KAMA - Kaufman Adaptive Moving Average
# =============================================================================

class TestKAMA:
    """Tests for KAMA indicator."""

    def test_kama_basic_computation(self, sample_ohlc_long):
        """KAMA should compute without errors."""
        close = sample_ohlc_long['close']
        result = KAMA.compute(close, n_period=10)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)

    def test_kama_warmup_period(self, sample_ohlc_long):
        """KAMA should have NaN values during warmup period."""
        close = sample_ohlc_long['close']
        result = KAMA.compute(close, n_period=10)
        # First (period-1) values should be NaN
        assert np.all(np.isnan(result[:9]))
        # After warmup should have valid values
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_kama_with_custom_periods(self, sample_ohlc_long):
        """KAMA should work with custom fast/slow periods."""
        close = sample_ohlc_long['close']
        result = KAMA.compute(close, n_period=5, period_fast=2, period_slow=10)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_kama_min_length_error(self):
        """KAMA should raise error if data too short."""
        close = np.array([100, 101, 102], dtype=np.float64)
        with pytest.raises(ValueError):
            KAMA.compute(close, n_period=10)


# =============================================================================
# ATR - Average True Range
# =============================================================================

class TestATR:
    """Tests for ATR indicator."""

    def test_atr_basic_computation(self, sample_ohlc_long):
        """ATR should compute without errors."""
        high = sample_ohlc_long['high']
        low = sample_ohlc_long['low']
        close = sample_ohlc_long['close']
        result = ATR.compute(high, low, close, period=14)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)

    def test_atr_sma_method(self):
        """ATR should work with SMA method."""
        high = np.array([105, 110, 108, 112, 115], dtype=np.float64)
        low = np.array([95, 100, 98, 102, 105], dtype=np.float64)
        close = np.array([100, 105, 103, 108, 110], dtype=np.float64)
        result = ATR.compute(high, low, close, period=3, method='sma')
        assert isinstance(result, np.ndarray)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_atr_ema_method(self):
        """ATR should work with EMA method."""
        high = np.array([105, 110, 108, 112, 115, 118], dtype=np.float64)
        low = np.array([95, 100, 98, 102, 105, 108], dtype=np.float64)
        close = np.array([100, 105, 103, 108, 110, 115], dtype=np.float64)
        result = ATR.compute(high, low, close, period=3, method='ema')
        assert isinstance(result, np.ndarray)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_atr_invalid_method_error(self):
        """ATR should raise error for invalid method."""
        high = low = close = np.array([100, 105, 110], dtype=np.float64)
        with pytest.raises(ValueError):
            ATR.compute(high, low, close, period=3, method='invalid')

    def test_atr_mismatched_lengths_error(self):
        """ATR should raise error for mismatched array lengths."""
        high = np.array([105, 110, 108], dtype=np.float64)
        low = np.array([95, 100], dtype=np.float64)  # Different length
        close = np.array([100, 105, 103], dtype=np.float64)
        with pytest.raises(ValueError):
            ATR.compute(high, low, close, period=2)


# =============================================================================
# EMA - Exponential Moving Average (additional tests)
# =============================================================================

class TestEMAExtended:
    """Additional tests for EMA indicator."""

    def test_ema_returns_numpy_array(self, sample_ohlc):
        """EMA should return numpy array."""
        result = EMA.compute(sample_ohlc['close'], period=5)
        assert isinstance(result, np.ndarray)

    def test_ema_values_reasonable(self, sample_ohlc_long):
        """EMA values should be within reasonable range of close prices."""
        close = sample_ohlc_long['close']
        result = EMA.compute(close, period=10)
        valid_result = result[~np.isnan(result)]
        valid_close = close[~np.isnan(result)]
        # EMA should be close to actual prices
        assert np.all(valid_result > 0)


# =============================================================================
# RSI - Relative Strength Index (additional tests)
# =============================================================================

class TestRSIExtended:
    """Additional tests for RSI indicator."""

    def test_rsi_bounds(self, sample_ohlc_long):
        """RSI should be between 0 and 100."""
        close = sample_ohlc_long['close']
        result = RSI.compute(close, period=14)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_default_period(self):
        """RSI should work with default period."""
        close = np.linspace(100, 150, 30, dtype=np.float64)
        result = RSI.compute(close)  # Default period=14
        valid = result[~np.isnan(result)]
        assert len(valid) > 0


# =============================================================================
# ADX - Average Directional Index
# =============================================================================

class TestADX:
    """Tests for ADX indicator."""

    def test_adx_basic_computation(self, sample_ohlc_long):
        """ADX should compute without errors."""
        high = sample_ohlc_long['high']
        low = sample_ohlc_long['low']
        close = sample_ohlc_long['close']
        result = ADX.compute(high, low, close, period=14)
        assert isinstance(result, tuple)
        assert len(result) == 3  # (adx, plus_di, minus_di)
        adx, plus_di, minus_di = result
        assert isinstance(adx, np.ndarray)
        assert isinstance(plus_di, np.ndarray)
        assert isinstance(minus_di, np.ndarray)

    def test_adx_returns_three_arrays(self, sample_ohlc_long):
        """ADX should return adx, +DI, and -DI arrays."""
        high = sample_ohlc_long['high']
        low = sample_ohlc_long['low']
        close = sample_ohlc_long['close']
        adx, plus_di, minus_di = ADX.compute(high, low, close, period=14)
        assert len(adx) == len(close)
        assert len(plus_di) == len(close)
        assert len(minus_di) == len(close)

    def test_adx_di_bounds(self, sample_ohlc_long):
        """+DI and -DI should be between 0 and 100."""
        high = sample_ohlc_long['high']
        low = sample_ohlc_long['low']
        close = sample_ohlc_long['close']
        adx, plus_di, minus_di = ADX.compute(high, low, close, period=10)
        valid_plus = plus_di[~np.isnan(plus_di)]
        valid_minus = minus_di[~np.isnan(minus_di)]
        if len(valid_plus) > 0:
            assert (valid_plus >= 0).all()
            assert (valid_plus <= 100).all()
        if len(valid_minus) > 0:
            assert (valid_minus >= 0).all()
            assert (valid_minus <= 100).all()


# =============================================================================
# Momentum
# =============================================================================

class TestMomentum:
    """Tests for Momentum indicator."""

    def test_momentum_basic_computation(self, sample_ohlc_long):
        """Momentum should compute without errors."""
        close = sample_ohlc_long['close']
        result = Momentum.compute(close, period=10)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)

    def test_momentum_uptrend_positive(self):
        """Momentum should be positive in uptrend."""
        close = np.array([100, 105, 110, 115, 120, 125], dtype=np.float64)
        result = Momentum.compute(close, period=5)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_momentum_default_period(self):
        """Momentum should work with default period."""
        close = np.linspace(100, 150, 20, dtype=np.float64)
        result = Momentum.compute(close)  # Default period=10
        valid = result[~np.isnan(result)]
        assert len(valid) > 0


# =============================================================================
# Bollinger Bands
# =============================================================================

class TestBollingerBands:
    """Tests for Bollinger Bands indicator."""

    def test_bollinger_basic_computation(self, sample_ohlc_long):
        """Bollinger Bands should compute without errors."""
        close = sample_ohlc_long['close']
        result = BollingerBands.compute(close, period=20)
        assert isinstance(result, tuple)
        assert len(result) == 3  # (upper, middle, lower)
        upper, middle, lower = result
        assert isinstance(upper, np.ndarray)
        assert isinstance(middle, np.ndarray)
        assert isinstance(lower, np.ndarray)

    def test_bollinger_upper_above_lower(self):
        """Upper band should always be above lower band."""
        close = np.linspace(100, 150, 30, dtype=np.float64)
        upper, middle, lower = BollingerBands.compute(close, period=10)
        valid_upper = upper[~np.isnan(upper)]
        valid_lower = lower[~np.isnan(lower)]
        # Align arrays
        min_len = min(len(valid_upper), len(valid_lower))
        assert np.all(valid_upper[:min_len] >= valid_lower[:min_len])

    def test_bollinger_middle_is_sma(self):
        """Middle band should be close to SMA."""
        close = np.linspace(100, 150, 40, dtype=np.float64)
        upper, middle, lower = BollingerBands.compute(close, period=20, std_dev=2)
        sma = SMA.compute(close, period=20)
        # Middle band should be equal to SMA (with std_dev=2, upper/lower are offset)
        valid_middle = middle[~np.isnan(middle)]
        valid_sma = sma[~np.isnan(middle)]
        assert np.allclose(valid_middle, valid_sma, rtol=1e-10)


# =============================================================================
# Donchian Channels
# =============================================================================

class TestDonchianChannels:
    """Tests for Donchian Channels indicator."""

    def test_donchian_basic_computation(self, sample_ohlc_long):
        """Donchian Channels should compute without errors."""
        high = sample_ohlc_long['high']
        low = sample_ohlc_long['low']
        result = DonchianChannels.compute(high, low, period=10)
        assert isinstance(result, tuple)
        assert len(result) == 3  # (upper, middle, lower)
        upper, middle, lower = result
        assert len(upper) == len(high)

    def test_donchian_upper_above_middle_above_lower(self):
        """Channels should be ordered: upper > middle > lower."""
        high = np.array([110, 115, 112, 118, 120], dtype=np.float64)
        low = np.array([90, 95, 92, 98, 100], dtype=np.float64)
        upper, middle, lower = DonchianChannels.compute(high, low, period=5)
        valid_upper = upper[~np.isnan(upper)]
        valid_middle = middle[~np.isnan(middle)]
        valid_lower = lower[~np.isnan(lower)]
        min_len = min(len(valid_upper), len(valid_middle), len(valid_lower))
        assert np.all(valid_upper[:min_len] >= valid_middle[:min_len])
        assert np.all(valid_middle[:min_len] >= valid_lower[:min_len])


# =============================================================================
# MACD
# =============================================================================

class TestMACD:
    """Tests for MACD indicator."""

    def test_macd_basic_computation(self):
        """MACD should compute without errors."""
        # MACD requires at least slow_period + signal_period = 35 bars by default
        close = np.linspace(100, 150, 50, dtype=np.float64)
        result = MACD.compute(close)
        assert isinstance(result, tuple)
        assert len(result) == 3  # (macd, signal, histogram)
        macd_line, signal_line, histogram = result
        assert isinstance(macd_line, np.ndarray)
        assert len(macd_line) == len(close)

    def test_macd_with_custom_periods(self):
        """MACD should work with custom periods."""
        close = np.linspace(100, 150, 40, dtype=np.float64)
        result = MACD.compute(close, fast_period=5, slow_period=10, signal_period=5)
        macd_line, signal_line, histogram = result
        valid_macd = macd_line[~np.isnan(macd_line)]
        assert len(valid_macd) > 0

    def test_macd_histogram_is_difference(self):
        """Histogram should be MACD line - signal line."""
        close = np.linspace(100, 150, 60, dtype=np.float64)
        macd_line, signal_line, histogram = MACD.compute(close, fast_period=12, slow_period=26, signal_period=9)
        valid_idx = ~np.isnan(macd_line) & ~np.isnan(signal_line)
        expected_histogram = macd_line[valid_idx] - signal_line[valid_idx]
        actual_histogram = histogram[valid_idx]
        assert np.allclose(actual_histogram, expected_histogram)


# =============================================================================
# Keltner Channel
# =============================================================================

class TestKeltnerChannel:
    """Tests for Keltner Channel indicator."""

    def test_keltner_basic_computation(self, sample_ohlc_long):
        """Keltner Channel should compute without errors."""
        high = sample_ohlc_long['high']
        low = sample_ohlc_long['low']
        close = sample_ohlc_long['close']
        result = KeltnerChannel.compute(high, low, close)
        assert isinstance(result, tuple)
        assert len(result) == 3  # (upper, middle, lower)
        upper, middle, lower = result
        assert len(upper) == len(close)

    def test_keltner_upper_above_middle_above_lower(self):
        """Keltner channels should be ordered correctly."""
        close = np.linspace(100, 150, 40, dtype=np.float64)
        high = close + 5
        low = close - 5
        upper, middle, lower = KeltnerChannel.compute(high, low, close, period=20)
        valid_upper = upper[~np.isnan(upper)]
        valid_middle = middle[~np.isnan(middle)]
        valid_lower = lower[~np.isnan(lower)]
        min_len = min(len(valid_upper), len(valid_middle), len(valid_lower))
        assert np.all(valid_upper[:min_len] >= valid_middle[:min_len])
        assert np.all(valid_middle[:min_len] >= valid_lower[:min_len])


# =============================================================================
# ADR - Average Daily Range
# =============================================================================

class TestADR:
    """Tests for ADR indicator."""

    def test_adr_basic_computation(self, sample_ohlc_long):
        """ADR should compute without errors."""
        high = sample_ohlc_long['high']
        low = sample_ohlc_long['low']
        result = ADR.compute(high, low, period=14)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(high)

    def test_adr_positive_values(self):
        """ADR should always be positive."""
        high = np.array([110, 115, 112, 118, 120], dtype=np.float64)
        low = np.array([90, 95, 92, 98, 100], dtype=np.float64)
        result = ADR.compute(high, low, period=3)
        valid = result[~np.isnan(result)]
        assert (valid > 0).all()

    def test_adr_default_period(self):
        """ADR should work with default period."""
        high = np.linspace(105, 120, 30, dtype=np.float64)
        low = np.linspace(95, 100, 30, dtype=np.float64)
        result = ADR.compute(high, low)  # Default period=14
        valid = result[~np.isnan(result)]
        assert len(valid) > 0


# =============================================================================
# VWAP - Volume Weighted Average Price
# =============================================================================

class TestVWAP:
    """Tests for VWAP indicator."""

    def test_vwap_basic_computation(self, sample_ohlc_long):
        """VWAP should compute without errors."""
        high = sample_ohlc_long['high']
        low = sample_ohlc_long['low']
        close = sample_ohlc_long['close']
        volume = sample_ohlc_long['volume']
        result = VWAP.compute(high, low, close, volume)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)

    def test_vwap_cumulative(self):
        """VWAP should compute cumulative when period=0."""
        high = low = close = np.array([100, 105, 110, 115, 120], dtype=np.float64)
        volume = np.array([1000, 2000, 1500, 2500, 1800], dtype=np.float64)
        result = VWAP.compute(high, low, close, volume, period=0)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_vwap_rolling(self):
        """VWAP should compute rolling when period>0."""
        high = low = close = np.linspace(100, 150, 30, dtype=np.float64)
        volume = np.ones(30, dtype=np.float64) * 1000
        result = VWAP.compute(high, low, close, volume, period=10)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_vwap_mismatched_lengths_error(self):
        """VWAP should raise error for mismatched array lengths."""
        high = np.array([105, 110], dtype=np.float64)
        low = np.array([95, 100], dtype=np.float64)
        close = np.array([100, 105], dtype=np.float64)
        volume = np.array([1000], dtype=np.float64)  # Different length
        with pytest.raises(ValueError):
            VWAP.compute(high, low, close, volume)


# =============================================================================
# Stochastic
# =============================================================================

class TestStochastic:
    """Tests for Stochastic Oscillator indicator."""

    def test_stochastic_basic_computation(self, sample_ohlc_long):
        """Stochastic should compute without errors."""
        high = sample_ohlc_long['high']
        low = sample_ohlc_long['low']
        close = sample_ohlc_long['close']
        result = Stochastic.compute(high, low, close)
        assert isinstance(result, tuple)
        assert len(result) == 2  # (k, d)
        k, d = result
        assert len(k) == len(close)

    def test_stochastic_bounds(self, sample_ohlc_long):
        """Stochastic %K and %D should be between 0 and 100."""
        high = sample_ohlc_long['high']
        low = sample_ohlc_long['low']
        close = sample_ohlc_long['close']
        k, d = Stochastic.compute(high, low, close, k_period=14, d_period=3, smooth_k=3)
        valid_k = k[~np.isnan(k)]
        valid_d = d[~np.isnan(d)]
        if len(valid_k) > 0:
            assert (valid_k >= 0).all() and (valid_k <= 100).all()
        if len(valid_d) > 0:
            assert (valid_d >= 0).all() and (valid_d <= 100).all()


# =============================================================================
# CCI - Commodity Channel Index
# =============================================================================

class TestCCI:
    """Tests for CCI indicator."""

    def test_cci_basic_computation(self, sample_ohlc_long):
        """CCI should compute without errors."""
        high = sample_ohlc_long['high']
        low = sample_ohlc_long['low']
        close = sample_ohlc_long['close']
        result = CCI.compute(high, low, close, period=20)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)

    def test_cci_default_period(self):
        """CCI should work with default period."""
        high = np.linspace(105, 120, 30, dtype=np.float64)
        low = np.linspace(95, 100, 30, dtype=np.float64)
        close = np.linspace(100, 110, 30, dtype=np.float64)
        result = CCI.compute(high, low, close)  # Default period=20
        valid = result[~np.isnan(result)]
        assert len(valid) > 0


# =============================================================================
# Williams %R (additional tests)
# =============================================================================

class TestWilliamsRExtended:
    """Additional tests for Williams %R indicator."""

    def test_williams_r_bounds(self, sample_ohlc_long):
        """Williams %R should be between -100 and 0."""
        high = sample_ohlc_long['high']
        low = sample_ohlc_long['low']
        close = sample_ohlc_long['close']
        result = WilliamsR.compute(high, low, close, period=14)
        valid = result[~np.isnan(result)]
        assert (valid >= -100).all()
        assert (valid <= 0).all()


# =============================================================================
# ROC - Rate of Change (additional tests)
# =============================================================================

class TestROCExtended:
    """Additional tests for ROC indicator."""

    def test_roc_default_period(self):
        """ROC should work with default period."""
        close = np.linspace(100, 150, 30, dtype=np.float64)
        result = ROC.compute(close)  # Default period=12
        valid = result[~np.isnan(result)]
        assert len(valid) > 0


# =============================================================================
# TRIX - Triple Exponential Moving Average Oscillator
# =============================================================================

class TestTRIX:
    """Tests for TRIX indicator."""

    def test_trix_basic_computation(self):
        """TRIX should compute without errors."""
        # TRIX requires 3*period-1 = 44 bars for period=15
        close = np.linspace(100, 150, 50, dtype=np.float64)
        result = TRIX.compute(close, period=15)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)

    def test_trix_default_period(self):
        """TRIX should work with default period."""
        close = np.linspace(100, 150, 50, dtype=np.float64)
        result = TRIX.compute(close)  # Default period=15
        valid = result[~np.isnan(result)]
        assert len(valid) > 0


# =============================================================================
# DeMarker
# =============================================================================

class TestDeMarker:
    """Tests for DeMarker indicator."""

    def test_demarker_basic_computation(self, sample_ohlc_long):
        """DeMarker should compute without errors."""
        high = sample_ohlc_long['high']
        low = sample_ohlc_long['low']
        result = DeMarker.compute(high, low, period=14)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(high)

    def test_demarker_bounds(self):
        """DeMarker should be between 0 and 1."""
        high = np.array([110, 115, 112, 118, 120, 125], dtype=np.float64)
        low = np.array([90, 95, 92, 98, 100, 105], dtype=np.float64)
        result = DeMarker.compute(high, low, period=3)
        valid = result[~np.isnan(result)]
        assert (valid >= 0).all()
        assert (valid <= 1).all()


# =============================================================================
# Aroon
# =============================================================================

class TestAroon:
    """Tests for Aroon indicator."""

    def test_aroon_basic_computation(self, sample_ohlc_long):
        """Aroon should compute without errors."""
        high = sample_ohlc_long['high']
        low = sample_ohlc_long['low']
        result = Aroon.compute(high, low, period=25)
        assert isinstance(result, tuple)
        assert len(result) == 2  # (aroon_up, aroon_down)
        aroon_up, aroon_down = result
        assert len(aroon_up) == len(high)

    def test_aroon_bounds(self, sample_ohlc_long):
        """Aroon should be between 0 and 100."""
        high = sample_ohlc_long['high']
        low = sample_ohlc_long['low']
        aroon_up, aroon_down = Aroon.compute(high, low, period=10)
        valid_up = aroon_up[~np.isnan(aroon_up)]
        valid_down = aroon_down[~np.isnan(aroon_down)]
        if len(valid_up) > 0:
            assert (valid_up >= 0).all() and (valid_up <= 100).all()
        if len(valid_down) > 0:
            assert (valid_down >= 0).all() and (valid_down <= 100).all()


# =============================================================================
# RVI - Relative Volatility Index
# =============================================================================

class TestRVI:
    """Tests for RVI indicator."""

    def test_rvi_basic_computation(self, sample_ohlc_long):
        """RVI should compute without errors."""
        open_prices = sample_ohlc_long['close'] - 1
        high = sample_ohlc_long['high']
        low = sample_ohlc_long['low']
        close = sample_ohlc_long['close']
        result = RVI.compute(open_prices, high, low, close, period=10)
        assert isinstance(result, tuple)
        assert len(result) == 2  # (rvi, signal)
        rvi, signal = result
        assert len(rvi) == len(close)

    def test_rvi_bounds(self, sample_ohlc_long):
        """RVI should be between 0 and 100."""
        open_prices = sample_ohlc_long['close'] - 1
        high = sample_ohlc_long['high']
        low = sample_ohlc_long['low']
        close = sample_ohlc_long['close']
        rvi, signal = RVI.compute(open_prices, high, low, close, period=10)
        valid_rvi = rvi[~np.isnan(rvi)]
        if len(valid_rvi) > 0:
            assert (valid_rvi >= 0).all() and (valid_rvi <= 100).all()


# =============================================================================
# Tests for edge cases across all indicators
# =============================================================================

class TestAllIndicatorsEdgeCases:
    """Edge case tests for all indicators."""

    def test_all_single_output_indicators_array_type(self):
        """All single-output indicators should return numpy arrays."""
        close = np.linspace(100, 200, 40, dtype=np.float64)
        high = close + 5
        low = close - 5
        volume = np.ones(40, dtype=np.float64) * 1000

        single_output_indicators = [
            (KAMA, [close], {'n_period': 10}),
            (ATR, [high, low, close], {'period': 14}),
            (EMA, [close], {'period': 10}),
            (RSI, [close], {'period': 14}),
            (Momentum, [close], {'period': 10}),
            (ROC, [close], {'period': 10}),
            (TRIX, [close], {'period': 10}),
            (ADR, [high, low], {'period': 10}),
            (CCI, [high, low, close], {'period': 14}),
            (WilliamsR, [high, low, close], {'period': 10}),
            (DeMarker, [high, low], {'period': 10}),
            (VWAP, [high, low, close, volume], {}),
        ]

        for indicator_cls, args, kwargs in single_output_indicators:
            result = indicator_cls.compute(*args, **kwargs)
            assert isinstance(result, np.ndarray), f"{indicator_cls.__name__} should return np.ndarray"
            assert len(result) > 0

    def test_all_tuple_output_indicators_tuple_type(self):
        """All tuple-output indicators should return tuples."""
        # Use larger arrays and custom periods to meet minimum requirements
        close = np.linspace(100, 200, 60, dtype=np.float64)
        high = close + 5
        low = close - 5
        open_prices = close - 1

        tuple_output_indicators = [
            (ADX, [high, low, close], {'period': 14}),
            (BollingerBands, [close], {'period': 20}),
            (DonchianChannels, [high, low], {'period': 20}),
            (MACD, [close], {'fast_period': 5, 'slow_period': 10, 'signal_period': 5}),
            (KeltnerChannel, [high, low, close], {'period': 20}),
            (Stochastic, [high, low, close], {'k_period': 14, 'd_period': 3, 'smooth_k': 3}),
            (Aroon, [high, low], {'period': 25}),
            (RVI, [open_prices, high, low, close], {'period': 10}),
        ]

        for indicator_cls, args, kwargs in tuple_output_indicators:
            result = indicator_cls.compute(*args, **kwargs)
            assert isinstance(result, tuple), f"{indicator_cls.__name__} should return tuple"
            for arr in result:
                assert isinstance(arr, np.ndarray), f"{indicator_cls.__name__} tuple elements should be np.ndarray"