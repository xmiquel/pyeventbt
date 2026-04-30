"""Tests for PyEventBT Portfolio positions."""
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
from pyeventbt.trading_context.trading_context import TypeContext
from pyeventbt.portfolio.portfolio import Portfolio
from pyeventbt.portfolio.core.entities.open_position import OpenPosition
from pyeventbt.portfolio.core.entities.pending_order import PendingOrder
from pyeventbt.events.events import BarEvent, Bar, EventType
from datetime import datetime
from decimal import Decimal as DecAlias

class TestPortfolioPositions:  # noqa: N801
    def test_get_positions_empty(self, sample_decimal_balance):
        """get_positions should return empty tuple when no positions exist."""
        mock_ee = MagicMock()
        mock_ee._get_legal_buy_size.return_value = DecAlias('1.0')
        mock_ee._get_account_balance.return_value = sample_decimal_balance
        mock_ee._get_account_equity.return_value = sample_decimal_balance
        mock_ee._get_strategy_positions.return_value = ()
        mock_ee._get_strategy_pending_orders.return_value = ()
        
        portfolio = Portfolio(initial_balance=sample_decimal_balance, execution_engine=mock_ee, trading_context=TypeContext.BACKTEST, base_timeframe='1min')
        positions = portfolio.get_positions()
        
        assert positions == ()
    
    def test_get_positions_by_symbol_empty(self, sample_decimal_balance):
        """get_positions with symbol filter should return empty when no positions."""
        mock_ee = MagicMock()
        mock_ee._get_legal_buy_size.return_value = DecAlias('1.0')
        mock_ee._get_account_balance.return_value = sample_decimal_balance
        mock_ee._get_account_equity.return_value = sample_decimal_balance
        mock_ee._get_strategy_positions.return_value = ()
        mock_ee._get_strategy_pending_orders.return_value = ()
        
        portfolio = Portfolio(initial_balance=sample_decimal_balance, execution_engine=mock_ee, trading_context=TypeContext.BACKTEST, base_timeframe='1min')
        positions = portfolio.get_positions(symbol='NDX')
        
        assert positions == ()
    
    def test_get_pending_orders_empty(self, sample_decimal_balance):
        """get_pending_orders should return empty tuple when no orders."""
        mock_ee = MagicMock()
        mock_ee._get_legal_buy_size.return_value = DecAlias('1.0')
        mock_ee._get_account_balance.return_value = sample_decimal_balance
        mock_ee._get_account_equity.return_value = sample_decimal_balance
        mock_ee._get_strategy_positions.return_value = ()
        mock_ee._get_strategy_pending_orders.return_value = ()
        
        portfolio = Portfolio(initial_balance=sample_decimal_balance, execution_engine=mock_ee, trading_context=TypeContext.BACKTEST, base_timeframe='1min')
        orders = portfolio.get_pending_orders()
        
        assert orders == ()
    
    def test_get_pending_orders_by_symbol_empty(self, sample_decimal_balance):
        """get_pending_orders with symbol filter should return empty when no orders."""
        mock_ee = MagicMock()
        mock_ee._get_legal_buy_size.return_value = DecAlias('1.0')
        mock_ee._get_account_balance.return_value = sample_decimal_balance
        mock_ee._get_account_equity.return_value = sample_decimal_balance
        mock_ee._get_strategy_positions.return_value = ()
        mock_ee._get_strategy_pending_orders.return_value = ()
        
        portfolio = Portfolio(initial_balance=sample_decimal_balance, execution_engine=mock_ee, trading_context=TypeContext.BACKTEST, base_timeframe='1min')
        orders = portfolio.get_pending_orders(symbol='NDX')
        
        assert orders == ()
    
    def test_get_number_of_strategy_open_positions_by_symbol(self, sample_decimal_balance):
        """get_number_of_strategy_open_positions_by_symbol should return dict with LONG/SHORT/TOTAL."""
        mock_ee = MagicMock()
        mock_ee._get_legal_buy_size.return_value = DecAlias('1.0')
        mock_ee._get_account_balance.return_value = sample_decimal_balance
        mock_ee._get_account_equity.return_value = sample_decimal_balance
        mock_ee._get_strategy_positions.return_value = ()
        mock_ee._get_strategy_pending_orders.return_value = ()
        
        portfolio = Portfolio(initial_balance=sample_decimal_balance, execution_engine=mock_ee, trading_context=TypeContext.BACKTEST, base_timeframe='1min')
        counts = portfolio.get_number_of_strategy_open_positions_by_symbol('NDX')
        
        assert 'LONG' in counts
        assert 'SHORT' in counts
        assert 'TOTAL' in counts
        assert counts['LONG'] == 0
        assert counts['SHORT'] == 0
        assert counts['TOTAL'] == 0
    
    def test_get_number_of_strategy_pending_orders_by_symbol(self, sample_decimal_balance):
        """get_number_of_strategy_pending_orders_by_symbol should return dict with counts."""
        mock_ee = MagicMock()
        mock_ee._get_legal_buy_size.return_value = DecAlias('1.0')
        mock_ee._get_account_balance.return_value = sample_decimal_balance
        mock_ee._get_account_equity.return_value = sample_decimal_balance
        mock_ee._get_strategy_positions.return_value = ()
        mock_ee._get_strategy_pending_orders.return_value = ()
        
        portfolio = Portfolio(initial_balance=sample_decimal_balance, execution_engine=mock_ee, trading_context=TypeContext.BACKTEST, base_timeframe='1min')
        counts = portfolio.get_number_of_strategy_pending_orders_by_symbol('NDX')
        
        assert 'BUY_LIMIT' in counts
        assert 'SELL_LIMIT' in counts
        assert 'BUY_STOP' in counts
        assert 'SELL_STOP' in counts
        assert 'TOTAL' in counts