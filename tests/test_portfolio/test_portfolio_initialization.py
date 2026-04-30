"""Tests for PyEventBT Portfolio initialization."""
import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch
from pyeventbt.trading_context.trading_context import TypeContext
from pyeventbt.portfolio.portfolio import Portfolio
from pyeventbt.portfolio.core.entities.open_position import OpenPosition
from pyeventbt.portfolio.core.entities.pending_order import PendingOrder
from pyeventbt.events.events import BarEvent, Bar, EventType
from datetime import datetime
from decimal import Decimal

class TestPortfolioInit:  # noqa: N801
    def test_portfolio_initial_balance(self, sample_decimal_balance):
        """Portfolio should store initial balance correctly."""
        mock_ee = MagicMock()
        mock_ee._get_legal_buy_size.return_value = Decimal('1.0')
        mock_ee._get_account_balance.return_value = sample_decimal_balance
        mock_ee._get_account_equity.return_value = sample_decimal_balance
        
        portfolio = Portfolio(initial_balance=sample_decimal_balance, execution_engine=mock_ee, trading_context=TypeContext.BACKTEST, base_timeframe='1min')
        
        assert portfolio._initial_balance == sample_decimal_balance
        assert portfolio._balance == sample_decimal_balance
    
    def test_portfolio_equity_equals_initial_at_start(self, sample_decimal_balance):
        """Portfolio equity should equal initial balance at start."""
        mock_ee = MagicMock()
        mock_ee._get_legal_buy_size.return_value = Decimal('1.0')
        mock_ee._get_account_balance.return_value = sample_decimal_balance
        mock_ee._get_account_equity.return_value = sample_decimal_balance
        mock_ee._get_strategy_positions.return_value = ()
        mock_ee._get_strategy_pending_orders.return_value = ()
        
        portfolio = Portfolio(initial_balance=sample_decimal_balance, execution_engine=mock_ee, trading_context=TypeContext.BACKTEST, base_timeframe='1min')
        
        assert portfolio._equity == sample_decimal_balance
    
    def test_portfolio_strategy_positions_empty_at_start(self, sample_decimal_balance):
        """Portfolio should start with no strategy positions."""
        mock_ee = MagicMock()
        mock_ee._get_legal_buy_size.return_value = Decimal('1.0')
        mock_ee._get_account_balance.return_value = sample_decimal_balance
        mock_ee._get_account_equity.return_value = sample_decimal_balance
        mock_ee._get_strategy_positions.return_value = ()
        mock_ee._get_strategy_pending_orders.return_value = ()
        
        portfolio = Portfolio(initial_balance=sample_decimal_balance, execution_engine=mock_ee, trading_context=TypeContext.BACKTEST, base_timeframe='1min')
        
        positions = portfolio.get_positions()
        assert len(positions) == 0
    
    def test_portfolio_pending_orders_empty_at_start(self, sample_decimal_balance):
        """Portfolio should start with no pending orders."""
        mock_ee = MagicMock()
        mock_ee._get_legal_buy_size.return_value = Decimal('1.0')
        mock_ee._get_account_balance.return_value = sample_decimal_balance
        mock_ee._get_account_equity.return_value = sample_decimal_balance
        mock_ee._get_strategy_positions.return_value = ()
        mock_ee._get_strategy_pending_orders.return_value = ()
        
        portfolio = Portfolio(initial_balance=sample_decimal_balance, execution_engine=mock_ee, trading_context=TypeContext.BACKTEST, base_timeframe='1min')
        
        orders = portfolio.get_pending_orders()
        assert len(orders) == 0
    
    def test_portfolio_realised_pnl_zero_at_start(self, sample_decimal_balance):
        """Portfolio realised PnL should be zero at start."""
        mock_ee = MagicMock()
        mock_ee._get_legal_buy_size.return_value = Decimal('1.0')
        mock_ee._get_account_balance.return_value = sample_decimal_balance
        mock_ee._get_account_equity.return_value = sample_decimal_balance
        mock_ee._get_strategy_positions.return_value = ()
        mock_ee._get_strategy_pending_orders.return_value = ()
        
        portfolio = Portfolio(initial_balance=sample_decimal_balance, execution_engine=mock_ee, trading_context=TypeContext.BACKTEST, base_timeframe='1min')
        
        assert portfolio._realised_pnl == Decimal('0.0')
    
    def test_portfolio_base_timeframe_stored(self, sample_decimal_balance):
        """Portfolio base timeframe should be stored."""
        mock_ee = MagicMock()
        mock_ee._get_legal_buy_size.return_value = Decimal('1.0')
        mock_ee._get_account_balance.return_value = sample_decimal_balance
        mock_ee._get_account_equity.return_value = sample_decimal_balance
        mock_ee._get_strategy_positions.return_value = ()
        mock_ee._get_strategy_pending_orders.return_value = ()
        
        portfolio = Portfolio(initial_balance=sample_decimal_balance, execution_engine=mock_ee, trading_context=TypeContext.BACKTEST, base_timeframe='5min')
        
        assert portfolio._base_timeframe == '5min'
    
    def test_portfolio_trading_context_stored(self, sample_decimal_balance):
        """Portfolio trading context should be stored."""
        mock_ee = MagicMock()
        mock_ee._get_legal_buy_size.return_value = Decimal('1.0')
        mock_ee._get_account_balance.return_value = sample_decimal_balance
        mock_ee._get_account_equity.return_value = sample_decimal_balance
        mock_ee._get_strategy_positions.return_value = ()
        mock_ee._get_strategy_pending_orders.return_value = ()
        
        portfolio = Portfolio(initial_balance=sample_decimal_balance, execution_engine=mock_ee, trading_context=TypeContext.BACKTEST, base_timeframe='1min')
        
        assert portfolio.trading_context == TypeContext.BACKTEST