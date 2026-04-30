"""
PyEventBT Test Configuration
Shared fixtures for all tests.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock


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


# =============================================================================
# Mock Fixtures for Framework Components
# =============================================================================

@pytest.fixture
def mock_execution_engine():
    """Mock execution engine for portfolio tests."""
    mock_ee = MagicMock()
    mock_ee._get_legal_buy_size.return_value = Decimal('1.0')
    mock_ee._get_account_balance.return_value = Decimal('10000.00')
    mock_ee._get_account_equity.return_value = Decimal('10000.00')
    mock_ee._get_strategy_positions.return_value = ()
    mock_ee._get_strategy_pending_orders.return_value = ()
    return mock_ee


@pytest.fixture
def mock_signal_engine():
    """Mock signal engine for strategy tests."""
    mock_sig = MagicMock()
    mock_sig.generate_signal.return_value = None
    mock_sig.strategy_id = "test_strategy"
    mock_sig.signal_timeframe = "1min"
    return mock_sig


@pytest.fixture
def mock_sizing_engine():
    """Mock sizing engine for portfolio handler tests."""
    mock_sz = MagicMock()
    mock_sz.get_suggested_order.return_value = MagicMock()
    return mock_sz


@pytest.fixture
def mock_risk_engine():
    """Mock risk engine for portfolio handler tests."""
    mock_risk = MagicMock()
    mock_risk.check_risk_and_size_order.return_value = MagicMock()
    return mock_risk


@pytest.fixture
def mock_portfolio(mock_execution_engine):
    """Mock portfolio for handler/director tests."""
    from pyeventbt.trading_context.trading_context import TypeContext
    from pyeventbt.portfolio.portfolio import Portfolio
    return Portfolio(
        initial_balance=Decimal('10000.00'),
        execution_engine=mock_execution_engine,
        trading_context=TypeContext.BACKTEST,
        base_timeframe='1min'
    )


@pytest.fixture
def mock_data_provider():
    """Mock data provider for signal tests."""
    mock_dp = MagicMock()
    mock_dp.get_latest_bars.return_value = MagicMock()
    return mock_dp


@pytest.fixture
def mock_modules(mock_data_provider, mock_portfolio):
    """Mock modules container for strategy tests."""
    from unittest.mock import MagicMock
    mock_mods = MagicMock()
    mock_mods.DATA_PROVIDER = mock_data_provider
    mock_mods.PORTFOLIO = mock_portfolio
    mock_mods.EXECUTION_ENGINE = MagicMock()
    return mock_mods


@pytest.fixture
def sample_bar_event():
    """Sample bar event for signal/handler tests."""
    from pyeventbt.events.events import BarEvent, Bar, EventType
    return BarEvent(
        type=EventType.BAR,
        symbol='EURUSD',
        datetime=datetime(2024, 1, 15, 10, 30, 0),
        data=Bar(
            open=10100,
            high=10500,
            low=9800,
            close=10300,
            tickvol=100,
            volume=50000,
            spread=10,
            digits=4
        ),
        timeframe='1min'
    )


@pytest.fixture
def sample_signal_event():
    """Sample signal event for portfolio handler tests."""
    from pyeventbt.events.events import SignalEvent, SignalType, OrderType, EventType
    return SignalEvent(
        type=EventType.SIGNAL,
        symbol='EURUSD',
        time_generated=datetime(2024, 1, 15, 10, 30, 0),
        strategy_id='test_strategy',
        forecast=10.0,
        signal_type=SignalType.BUY,
        order_type=OrderType.MARKET,
        order_price=Decimal('1.1000'),
        sl=Decimal('1.0900'),
        tp=Decimal('1.1200')
    )


@pytest.fixture
def mock_broker():
    """Mock broker for trading tests."""
    from unittest.mock import MagicMock
    mock_bk = MagicMock()
    mock_bk.get_account_info.return_value = MagicMock()
    return mock_bk


# =============================================================================
# Test-specific fixtures
# =============================================================================

@pytest.fixture
def mock_portfolio_handler(mock_sizing_engine, mock_risk_engine, mock_portfolio):
    """Mock portfolio handler for trading director tests."""
    from queue import Queue
    from pyeventbt.portfolio_handler.portfolio_handler import PortfolioHandler
    return PortfolioHandler(
        events_queue=Queue(),
        sizing_engine=mock_sizing_engine,
        risk_engine=mock_risk_engine,
        portfolio=mock_portfolio,
        base_timeframe='1min'
    )