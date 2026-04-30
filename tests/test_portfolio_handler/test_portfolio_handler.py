"""Tests for PyEventBT PortfolioHandler."""
import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch
from queue import Queue
from pyeventbt.portfolio_handler.portfolio_handler import PortfolioHandler
from pyeventbt.portfolio.portfolio import Portfolio
from pyeventbt.trading_context.trading_context import TypeContext
from pyeventbt.events.events import BarEvent, Bar, EventType
from datetime import datetime


class TestPortfolioHandlerInit:  # noqa: N801
    def test_portfolio_handler_initializes(self, mock_sizing_engine, mock_risk_engine, mock_portfolio):
        """PortfolioHandler should initialize with all dependencies."""
        events_queue = Queue()

        handler = PortfolioHandler(
            events_queue=events_queue,
            sizing_engine=mock_sizing_engine,
            risk_engine=mock_risk_engine,
            portfolio=mock_portfolio,
            base_timeframe='1min'
        )

        assert handler.event_queue is not None
        assert handler.POSITION_SIZER is mock_sizing_engine
        assert handler.RISK_ENGINE is mock_risk_engine
        assert handler.PORTFOLIO is mock_portfolio

    def test_portfolio_handler_stores_base_timeframe(self, mock_sizing_engine, mock_risk_engine, mock_portfolio):
        """PortfolioHandler should store base timeframe."""
        events_queue = Queue()

        handler = PortfolioHandler(
            events_queue=events_queue,
            sizing_engine=mock_sizing_engine,
            risk_engine=mock_risk_engine,
            portfolio=mock_portfolio,
            base_timeframe='5min'
        )

        assert handler.base_timeframe == '5min'


class TestPortfolioHandlerProcessBarEvent:  # noqa: N801
    def test_process_bar_event_ignores_non_base_timeframe(self, mock_sizing_engine, mock_risk_engine):
        """process_bar_event should ignore events from non-base timeframe."""
        from unittest.mock import MagicMock
        mock_portfolio = MagicMock()
        
        events_queue = Queue()
        handler = PortfolioHandler(
            events_queue=events_queue,
            sizing_engine=mock_sizing_engine,
            risk_engine=mock_risk_engine,
            portfolio=mock_portfolio,
            base_timeframe='1min'
        )

        bar_event = BarEvent(
            type=EventType.BAR,
            symbol='EURUSD',
            datetime=datetime(2024, 1, 15, 10, 30, 0),
            data=Bar(
                open=10100, high=10500, low=9800, close=10300,
                tickvol=100, volume=50000, spread=10, digits=4
            ),
            timeframe='5min'
        )

        handler.process_bar_event(bar_event)

        # Should not update portfolio for non-matching timeframe
        mock_portfolio._update_portfolio.assert_not_called()


class TestPortfolioHandlerProcessSignalEvent:  # noqa: N801
    def test_process_signal_event_calls_sizing_engine(self, mock_sizing_engine, mock_risk_engine, mock_portfolio, sample_signal_event):
        """process_signal_event should call sizing engine."""
        events_queue = Queue()
        handler = PortfolioHandler(
            events_queue=events_queue,
            sizing_engine=mock_sizing_engine,
            risk_engine=mock_risk_engine,
            portfolio=mock_portfolio,
            base_timeframe='1min'
        )

        handler.process_signal_event(sample_signal_event)

        mock_sizing_engine.get_suggested_order.assert_called_once_with(sample_signal_event)

    def test_process_signal_event_calls_risk_engine(self, mock_sizing_engine, mock_risk_engine, mock_portfolio, sample_signal_event):
        """process_signal_event should call risk engine after sizing."""
        events_queue = Queue()
        handler = PortfolioHandler(
            events_queue=events_queue,
            sizing_engine=mock_sizing_engine,
            risk_engine=mock_risk_engine,
            portfolio=mock_portfolio,
            base_timeframe='1min'
        )

        handler.process_signal_event(sample_signal_event)

        mock_risk_engine.assess_order.assert_called_once()


class TestPortfolioHandlerProcessFillEvent:  # noqa: N801
    def test_process_fill_event_calls_archiver(self, mock_sizing_engine, mock_risk_engine):
        """process_fill_event should call trade archiver."""
        from pyeventbt.events.events import FillEvent, DealType, SignalType
        from unittest.mock import MagicMock, patch
        
        # Use MagicMock for portfolio to avoid issues
        mock_portfolio = MagicMock()
        
        # Patch TradeArchiver to avoid file system issues
        with patch('pyeventbt.portfolio_handler.portfolio_handler.TradeArchiver') as MockArchiver:
            mock_archiver = MagicMock()
            MockArchiver.return_value = mock_archiver
            
            handler = PortfolioHandler(
                events_queue=Queue(),
                sizing_engine=mock_sizing_engine,
                risk_engine=mock_risk_engine,
                portfolio=mock_portfolio,
                base_timeframe='1min'
            )

            fill_event = FillEvent(
                type=EventType.FILL,
                deal=DealType.IN,
                symbol='EURUSD',
                time_generated=datetime(2024, 1, 15, 10, 30, 0),
                position_id=1,
                strategy_id='test_strategy',
                exchange='MetaTrader',
                volume=Decimal('1.0'),
                price=Decimal('1.1000'),
                signal_type=SignalType.BUY,
                commission=Decimal('1.0'),
                swap=Decimal('0.0'),
                fee=Decimal('0.0'),
                gross_profit=Decimal('0.0'),
                ccy='EUR'
            )

            handler.process_fill_event(fill_event)

            mock_archiver.archive_trade.assert_called_once_with(fill_event)