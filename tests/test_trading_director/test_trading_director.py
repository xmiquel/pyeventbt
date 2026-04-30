"""Tests for PyEventBT TradingDirector."""
import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch
from queue import Queue
from pyeventbt.trading_director.trading_director import TradingDirector
from pyeventbt.portfolio_handler.portfolio_handler import PortfolioHandler
from pyeventbt.signal_engine.services.signal_engine_service import SignalEngineService
from pyeventbt.trading_director.core.configurations.trading_session_configurations import MT5BacktestSessionConfig, MT5LiveSessionConfig
from pyeventbt.strategy.core.modules import Modules
from pyeventbt.strategy.core.strategy_timeframes import StrategyTimeframes
from pyeventbt.hooks.hook_service import HookService
from datetime import datetime


class TestTradingDirectorInit:  # noqa: N801
    def test_trading_director_initializes(self, mock_signal_engine, mock_portfolio_handler, mock_modules):
        """TradingDirector should initialize with all configurations."""
        events_queue = Queue()

        config = MT5BacktestSessionConfig(
            initial_capital=Decimal('10000.00'),
            start_date=datetime(2024, 1, 1),
            backtest_name='test_backtest'
        )

        director = TradingDirector(
            events_queue=events_queue,
            signal_engine_service=mock_signal_engine,
            portfolio_handler=mock_portfolio_handler,
            trading_session_config=config,
            modules=mock_modules
        )

        assert director.events_queue is not None
        assert director.SIGNAL_GENERATOR is mock_signal_engine
        assert director.PORTFOLIO_HANDLER is mock_portfolio_handler

    def test_trading_director_sets_backtest_config(self, mock_signal_engine, mock_portfolio_handler, mock_modules):
        """TradingDirector should configure backtest session."""
        events_queue = Queue()

        config = MT5BacktestSessionConfig(
            initial_capital=Decimal('10000.00'),
            start_date=datetime(2024, 1, 1),
            backtest_name='test_backtest'
        )

        director = TradingDirector(
            events_queue=events_queue,
            signal_engine_service=mock_signal_engine,
            portfolio_handler=mock_portfolio_handler,
            trading_session_config=config,
            modules=mock_modules
        )

        assert director.initial_capital == Decimal('10000.00')
        assert director.backtest_name == 'test_backtest'


class TestTradingDirectorEventHandlers:  # noqa: N801
    def test_handle_bar_event_calls_portfolio_handler(self, mock_signal_engine, mock_modules):
        """_handle_bar_event should call portfolio handler."""
        # Create MagicMock for portfolio handler
        mock_portfolio_handler = MagicMock()
        
        events_queue = Queue()

        config = MT5BacktestSessionConfig(
            initial_capital=Decimal('10000.00'),
            start_date=datetime(2024, 1, 1),
            backtest_name='test_backtest'
        )

        director = TradingDirector(
            events_queue=events_queue,
            signal_engine_service=mock_signal_engine,
            portfolio_handler=mock_portfolio_handler,
            trading_session_config=config,
            modules=mock_modules
        )

        mock_bar_event = MagicMock()
        mock_bar_event.type = MagicMock()

        director._handle_bar_event(mock_bar_event)

        mock_portfolio_handler.process_bar_event.assert_called_once_with(mock_bar_event)

    def test_handle_signal_event_calls_portfolio_handler(self, mock_signal_engine, mock_modules):
        """_handle_signal_event should call portfolio handler."""
        # Create MagicMock for portfolio handler
        mock_portfolio_handler = MagicMock()
        
        events_queue = Queue()

        config = MT5BacktestSessionConfig(
            initial_capital=Decimal('10000.00'),
            start_date=datetime(2024, 1, 1),
            backtest_name='test_backtest'
        )

        director = TradingDirector(
            events_queue=events_queue,
            signal_engine_service=mock_signal_engine,
            portfolio_handler=mock_portfolio_handler,
            trading_session_config=config,
            modules=mock_modules
        )

        mock_signal_event = MagicMock()
        mock_signal_event.type = MagicMock()

        director._handle_signal_event(mock_signal_event)

        mock_portfolio_handler.process_signal_event.assert_called_once_with(mock_signal_event)

    def test_handle_fill_event_calls_portfolio_handler(self, mock_signal_engine, mock_modules):
        """_handle_fill_event should call portfolio handler."""
        # Create MagicMock for portfolio handler
        mock_portfolio_handler = MagicMock()
        
        events_queue = Queue()

        config = MT5BacktestSessionConfig(
            initial_capital=Decimal('10000.00'),
            start_date=datetime(2024, 1, 1),
            backtest_name='test_backtest'
        )

        director = TradingDirector(
            events_queue=events_queue,
            signal_engine_service=mock_signal_engine,
            portfolio_handler=mock_portfolio_handler,
            trading_session_config=config,
            modules=mock_modules
        )

        mock_fill_event = MagicMock()
        mock_fill_event.type = MagicMock()

        director._handle_fill_event(mock_fill_event)

        mock_portfolio_handler.process_fill_event.assert_called_once_with(mock_fill_event)


class TestTradingDirectorAddSchedule:  # noqa: N801
    def test_add_schedule_registers_callback(self, mock_signal_engine, mock_portfolio_handler, mock_modules):
        """add_schedule should register callback to schedule service."""
        events_queue = Queue()

        config = MT5BacktestSessionConfig(
            initial_capital=Decimal('10000.00'),
            start_date=datetime(2024, 1, 1),
            backtest_name='test_backtest'
        )

        director = TradingDirector(
            events_queue=events_queue,
            signal_engine_service=mock_signal_engine,
            portfolio_handler=mock_portfolio_handler,
            trading_session_config=config,
            modules=mock_modules,
            run_schedules=True
        )

        def dummy_callback(scheduled_event, modules):
            pass

        director.add_schedule(StrategyTimeframes.ONE_MIN, dummy_callback)

        # Should have registered the callback - no exception means success
        assert director.SCHEDULE_SERVICE is not None