"""
Parameter validation tests for PyEventBT indicators.
Tests that invalid parameters raise appropriate errors.
"""
import numpy as np
import pytest

from pyeventbt.indicators.indicators import (
    KAMA, ATR, SMA, EMA, RSI, ADX, Momentum, BollingerBands,
    DonchianChannels, MACD, KeltnerChannel, ADR, VWAP,
    Stochastic, CCI, WilliamsR, ROC, TRIX, DeMarker, Aroon, RVI
)


# =============================================================================
# Tests: Close-only indicators
# =============================================================================

class TestSMAParamValidation:
    def test_sma_period_too_large(self):
        close = np.array([100, 101, 102], dtype=np.float64)
        with pytest.raises(ValueError):
            SMA.compute(close, period=10)
    
    def test_sma_empty_array(self):
        close = np.array([], dtype=np.float64)
        with pytest.raises(ValueError):
            SMA.compute(close, period=5)


class TestEMAParamValidation:
    def test_ema_period_too_large(self):
        close = np.array([100, 101, 102], dtype=np.float64)
        with pytest.raises(ValueError):
            EMA.compute(close, period=10)


class TestKAMAParamValidation:
    def test_kama_period_too_large(self):
        close = np.array([100, 101, 102], dtype=np.float64)
        with pytest.raises(ValueError):
            KAMA.compute(close, n_period=10)


class TestRSIParamValidation:
    def test_rsi_period_too_large(self):
        close = np.array([100, 101, 102, 103, 104], dtype=np.float64)
        with pytest.raises(ValueError):
            RSI.compute(close, period=14)
    
    def test_rsi_invalid_period_zero(self):
        close = np.array([100, 101, 102, 103, 104], dtype=np.float64)
        # period=0 causes ZeroDivisionError (not ValueError)
        with pytest.raises((ValueError, ZeroDivisionError)):
            RSI.compute(close, period=0)


class TestMomentumParamValidation:
    def test_momentum_period_too_large(self):
        close = np.array([100, 101, 102, 103], dtype=np.float64)
        with pytest.raises(ValueError):
            Momentum.compute(close, period=10)


class TestROCParamValidation:
    def test_roc_period_zero(self):
        # ROC with period=0 returns array with NaN (no error raised)
        close = np.array([100, 101, 102, 103, 104], dtype=np.float64)
        result = ROC.compute(close, period=0)
        assert result is not None
        assert len(result) == len(close)


class TestTRIXParamValidation:
    def test_trix_period_zero(self):
        close = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108], dtype=np.float64)
        with pytest.raises(ValueError):
            TRIX.compute(close, period=0)
    
    def test_trix_array_too_short(self):
        close = np.array([100, 101, 102, 103, 104], dtype=np.float64)
        with pytest.raises(ValueError):
            TRIX.compute(close, period=5)


# =============================================================================
# Tests: High + Low + Close indicators
# =============================================================================

class TestATRParamValidation:
    def test_atr_mismatched_array_lengths(self):
        high = np.array([100, 101, 102], dtype=np.float64)
        low = np.array([98, 99], dtype=np.float64)
        close = np.array([99, 100, 101], dtype=np.float64)
        with pytest.raises(ValueError):
            ATR.compute(high, low, close, period=14)
    
    def test_atr_invalid_method(self):
        high = np.array([100, 101, 102], dtype=np.float64)
        low = np.array([98, 99, 100], dtype=np.float64)
        close = np.array([99, 100, 101], dtype=np.float64)
        with pytest.raises(ValueError):
            ATR.compute(high, low, close, period=14, method='invalid')


class TestADRParamValidation:
    def test_adr_mismatched_array_lengths(self):
        high = np.array([100, 101, 102], dtype=np.float64)
        low = np.array([98, 99], dtype=np.float64)
        with pytest.raises(ValueError):
            ADR.compute(high, low, period=14)


class TestADXParamValidation:
    def test_adx_mismatched_array_lengths(self):
        high = np.array([100, 101, 102], dtype=np.float64)
        low = np.array([98, 99, 100], dtype=np.float64)
        close = np.array([99, 100], dtype=np.float64)
        with pytest.raises(ValueError):
            ADX.compute(high, low, close, period=14)
    
    def test_adx_array_too_short(self):
        high = np.array([100, 101, 102], dtype=np.float64)
        low = np.array([98, 99, 100], dtype=np.float64)
        close = np.array([99, 100, 101], dtype=np.float64)
        with pytest.raises(ValueError):
            ADX.compute(high, low, close, period=14)


class TestBollingerBandsParamValidation:
    def test_bb_period_too_large(self):
        close = np.array([100, 101, 102], dtype=np.float64)
        with pytest.raises(ValueError):
            BollingerBands.compute(close, period=20)


class TestDonchianChannelsParamValidation:
    def test_donchian_mismatched_array_lengths(self):
        high = np.array([100, 101, 102], dtype=np.float64)
        low = np.array([98, 99], dtype=np.float64)
        with pytest.raises(ValueError):
            DonchianChannels.compute(high, low, period=5)


class TestKeltnerChannelParamValidation:
    def test_keltner_mismatched_array_lengths(self):
        high = np.array([100, 101, 102], dtype=np.float64)
        low = np.array([98, 99], dtype=np.float64)
        close = np.array([99, 100, 101], dtype=np.float64)
        with pytest.raises(ValueError):
            KeltnerChannel.compute(high, low, close, period=20)


class TestCCIParamValidation:
    def test_cci_mismatched_array_lengths(self):
        high = np.array([100, 101, 102], dtype=np.float64)
        low = np.array([98, 99, 100], dtype=np.float64)
        close = np.array([99, 100], dtype=np.float64)
        with pytest.raises(ValueError):
            CCI.compute(high, low, close, period=14)


class TestWilliamsRParamValidation:
    def test_williams_r_mismatched_array_lengths(self):
        high = np.array([100, 101, 102], dtype=np.float64)
        low = np.array([98, 99, 100], dtype=np.float64)
        close = np.array([99, 100], dtype=np.float64)
        with pytest.raises(ValueError):
            WilliamsR.compute(high, low, close, period=14)


class TestDeMarkerParamValidation:
    def test_demarker_period_zero(self):
        high = np.array([100, 101, 102], dtype=np.float64)
        low = np.array([98, 99, 100], dtype=np.float64)
        with pytest.raises(ValueError):
            DeMarker.compute(high, low, period=0)
    
    def test_demarker_array_too_short(self):
        high = np.array([100, 101, 102], dtype=np.float64)
        low = np.array([98, 99, 100], dtype=np.float64)
        with pytest.raises(ValueError):
            DeMarker.compute(high, low, period=14)


class TestAroonParamValidation:
    def test_aroon_mismatched_array_lengths(self):
        high = np.array([100, 101, 102], dtype=np.float64)
        low = np.array([98, 99], dtype=np.float64)
        with pytest.raises(ValueError):
            Aroon.compute(high, low, period=5)


# =============================================================================
# Tests: Volume-based indicators
# =============================================================================

class TestVWAPParamValidation:
    def test_vwap_mismatched_array_lengths(self):
        high = np.array([100, 101, 102], dtype=np.float64)
        low = np.array([98, 99, 100], dtype=np.float64)
        close = np.array([99, 100, 101], dtype=np.float64)
        volume = np.array([1000, 2000], dtype=np.float64)
        with pytest.raises(ValueError):
            VWAP.compute(high, low, close, volume)


# =============================================================================
# Tests: Stochastic indicators
# =============================================================================

class TestStochasticParamValidation:
    def test_stochastic_mismatched_array_lengths(self):
        high = np.array([100, 101, 102], dtype=np.float64)
        low = np.array([98, 99], dtype=np.float64)
        close = np.array([99, 100, 101], dtype=np.float64)
        with pytest.raises(ValueError):
            Stochastic.compute(high, low, close)


# =============================================================================
# Tests: RVI
# =============================================================================

class TestRVIParamValidation:
    def test_rvi_period_zero(self):
        open_ = np.array([100, 101, 102, 103, 104, 105, 106], dtype=np.float64)
        high = np.array([102, 103, 104, 105, 106, 107, 108], dtype=np.float64)
        low = np.array([98, 99, 100, 101, 102, 103, 104], dtype=np.float64)
        close = np.array([101, 102, 103, 104, 105, 106, 107], dtype=np.float64)
        with pytest.raises(ValueError):
            RVI.compute(open_, high, low, close, period=0)
    
    def test_rvi_mismatched_array_lengths(self):
        open_ = np.array([100, 101, 102], dtype=np.float64)
        high = np.array([102, 103, 104], dtype=np.float64)
        low = np.array([98, 99, 100], dtype=np.float64)
        close = np.array([101, 102], dtype=np.float64)
        with pytest.raises(ValueError):
            RVI.compute(open_, high, low, close, period=10)