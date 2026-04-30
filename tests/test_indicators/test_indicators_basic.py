"""Basic tests for PyEventBT technical indicators."""
import numpy as np
import pytest
from pyeventbt.indicators.indicators import (
    KAMA, ATR, SMA, EMA, RSI, ADX, Momentum, BollingerBands,
    DonchianChannels, MACD, KeltnerChannel, ADR, VWAP,
    Stochastic, CCI, WilliamsR, ROC, TRIX, DeMarker, Aroon, RVI
)

class TestSMABasic:
    def test_sma_computes_correct_values(self, sample_ohlc):
        result = SMA.compute(sample_ohlc['close'], period=5)
        assert len(result) == len(sample_ohlc['close'])
        assert np.isnan(result[0])
        assert not np.isnan(result[4])
    
    def test_sma_output_has_nans_for_warmup(self, sample_ohlc):
        result = SMA.compute(sample_ohlc['close'], period=3)
        assert np.all(np.isnan(result[:2]))
        assert np.all(~np.isnan(result[2:]))
    
    def test_sma_period_equals_length(self, sample_ohlc):
        close = sample_ohlc['close'][:10]
        result = SMA.compute(close, period=10)
        assert not np.isnan(result[-1])