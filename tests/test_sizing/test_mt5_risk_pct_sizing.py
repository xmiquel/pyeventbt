"""Tests for PyEventBT MT5RiskPctSizing."""
import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch
from pyeventbt.sizing_engine.sizing_engines.mt5_risk_pct_sizing import MT5RiskPctSizing
from pyeventbt.sizing_engine.core.configurations.sizing_engine_configurations import RiskPctSizingConfig
from pyeventbt.events.events import SignalEvent, SignalType, OrderType, EventType
from pyeventbt.portfolio_handler.core.entities.suggested_order import SuggestedOrder
from pyeventbt.trading_context.trading_context import TypeContext
from datetime import datetime


class TestMT5RiskPctSizingInit:  # noqa: N801
    def test_risk_pct_sizing_initializes_with_config(self):
        """MT5RiskPctSizing should initialize with config."""
        with patch.dict('sys.modules', {'MetaTrader5': MagicMock()}):
            config = RiskPctSizingConfig(risk_pct=1.0)
            sizing = MT5RiskPctSizing(configs=config, trading_context=TypeContext.LIVE)
            assert sizing.risk_pct == 1.0

    def test_risk_pct_sizing_stores_risk_value(self):
        """MT5RiskPctSizing should store risk percentage value."""
        with patch.dict('sys.modules', {'MetaTrader5': MagicMock()}):
            config = RiskPctSizingConfig(risk_pct=2.0)
            sizing = MT5RiskPctSizing(configs=config, trading_context=TypeContext.LIVE)
            assert sizing.risk_pct == 2.0


class TestMT5RiskPctSizingMethods:  # noqa: N801
    def test_convert_currency_same_currency_returns_same(self):
        """_convert_currency should return same amount when currencies match."""
        config = RiskPctSizingConfig(risk_pct=1.0)
        sizing = MT5RiskPctSizing(configs=config, trading_context=TypeContext.BACKTEST)

        latest_tick = {'bid': 1.1000}
        result = sizing._convert_currency_amount_to_another_currency(100.0, 'USD', 'USD', latest_tick)

        assert result == 100.0


class TestMT5RiskPctSizingValidation:  # noqa: N801
    def test_get_suggested_order_requires_valid_risk_pct(self):
        """get_suggested_order should raise on invalid risk_pct."""
        config = RiskPctSizingConfig(risk_pct=0.0)
        sizing = MT5RiskPctSizing(configs=config, trading_context=TypeContext.BACKTEST)

        signal = SignalEvent(
            type=EventType.SIGNAL,
            symbol='EURUSD',
            time_generated=datetime(2024, 1, 15, 10, 30, 0),
            strategy_id='test_strategy',
            forecast=10.0,
            signal_type=SignalType.BUY,
            order_type=OrderType.MARKET,
            sl=Decimal('1.0900')
        )

        with pytest.raises(Exception, match="Risk percentage"):
            sizing.get_suggested_order(signal, MagicMock())

    def test_get_suggested_order_requires_stop_loss(self):
        """get_suggested_order should raise when stop loss is 0."""
        config = RiskPctSizingConfig(risk_pct=1.0)
        sizing = MT5RiskPctSizing(configs=config, trading_context=TypeContext.BACKTEST)

        signal = SignalEvent(
            type=EventType.SIGNAL,
            symbol='EURUSD',
            time_generated=datetime(2024, 1, 15, 10, 30, 0),
            strategy_id='test_strategy',
            forecast=10.0,
            signal_type=SignalType.BUY,
            order_type=OrderType.MARKET,
            sl=Decimal('0.0')
        )

        with pytest.raises(Exception, match="Stop loss"):
            sizing.get_suggested_order(signal, MagicMock())