"""Tests for PyEventBT SignalMACrossover."""
import pytest
from unittest.mock import MagicMock, patch
from pyeventbt.signal_engine.signal_engines.signal_ma_crossover import SignalMACrossover
from pyeventbt.signal_engine.core.configurations.signal_engine_configurations import MACrossoverConfig, MAType
from pyeventbt.events.events import BarEvent, Bar, EventType
from pyeventbt.trading_context.trading_context import TypeContext
from datetime import datetime


class TestSignalMACrossoverInit:  # noqa: N801
    def test_signal_ma_crossover_initializes(self):
        """SignalMACrossover should initialize with config."""
        config = MACrossoverConfig(
            strategy_id="test_strategy",
            signal_timeframe="1min",
            fast_period=5,
            slow_period=20
        )
        signal_engine = SignalMACrossover(configurations=config)

        assert signal_engine.strategy_id == "test_strategy"
        assert signal_engine.signal_timeframe == "1min"
        assert signal_engine.fast_period == 5
        assert signal_engine.slow_period == 20

    def test_signal_ma_crossover_stores_ma_type(self):
        """SignalMACrossover should store ma_type from config."""
        config = MACrossoverConfig(
            strategy_id="test_strategy",
            signal_timeframe="1min",
            ma_type=MAType.EXPONENTIAL,
            fast_period=5,
            slow_period=20
        )
        signal_engine = SignalMACrossover(configurations=config)

        assert signal_engine.ma_type == MAType.EXPONENTIAL


class TestSignalMACrossoverGenerate:  # noqa: N801
    def test_generate_returns_none_on_mismatched_timeframe(self, sample_bar_event):
        """Generate should return None when bar timeframe differs."""
        config = MACrossoverConfig(
            strategy_id="test_strategy",
            signal_timeframe="5min",
            fast_period=5,
            slow_period=20
        )
        signal_engine = SignalMACrossover(configurations=config)

        mock_modules = MagicMock()

        result = signal_engine.generate_signal(sample_bar_event, mock_modules)

        assert result is None

    def test_generate_returns_none_on_insufficient_data(self, sample_bar_event):
        """Generate should return None when insufficient bars."""
        config = MACrossoverConfig(
            strategy_id="test_strategy",
            signal_timeframe="1min",
            fast_period=5,
            slow_period=20
        )
        signal_engine = SignalMACrossover(configurations=config)

        mock_modules = MagicMock()
        # Return empty like mock (less than 2 rows)
        mock_modules.DATA_PROVIDER.get_latest_bars.return_value = MagicMock()
        mock_modules.DATA_PROVIDER.get_latest_bars.return_value.shape = (1,)
        mock_modules.PORTFOLIO.get_number_of_strategy_open_positions_by_symbol.return_value = {'LONG': 0, 'SHORT': 0}

        result = signal_engine.generate_signal(sample_bar_event, mock_modules)

        assert result is None