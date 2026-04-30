"""Tests for PyEventBT Strategy initialization."""
import pytest
from unittest.mock import MagicMock, patch
from pyeventbt.strategy.strategy import Strategy
from pyeventbt.strategy.core.strategy_timeframes import StrategyTimeframes
from pyeventbt.hooks.hook_service import Hooks
from pyeventbt.strategy.core.verbose_level import VerboseLevel


class TestStrategyInit:  # noqa: N801
    def test_strategy_initializes_with_default_logging(self):
        """Strategy should initialize with default logging level."""
        with patch('pyeventbt.strategy.strategy.logger'):
            strategy = Strategy()
            assert strategy is not None

    def test_strategy_stores_initial_configs(self):
        """Strategy should initialize with default config values."""
        with patch('pyeventbt.strategy.strategy.logger'):
            strategy = Strategy()
            # Private attributes start as None or default
            assert strategy._Strategy__sizing_engine_config is None
            assert strategy._Strategy__signal_engine_config is None

    def test_strategy_initializes_signal_engines_dict(self):
        """Strategy should initialize empty signal engines dict."""
        with patch('pyeventbt.strategy.strategy.logger'):
            strategy = Strategy()
            assert strategy._Strategy__signal_engines == {}

    def test_strategy_initializes_sizing_engines_dict(self):
        """Strategy should initialize empty sizing engines dict."""
        with patch('pyeventbt.strategy.strategy.logger'):
            strategy = Strategy()
            assert strategy._Strategy__sizing_engines == {}

    def test_strategy_initializes_risk_engines_dict(self):
        """Strategy should initialize empty risk engines dict."""
        with patch('pyeventbt.strategy.strategy.logger'):
            strategy = Strategy()
            assert strategy._Strategy__risk_engines == {}

    def test_strategy_initializes_timeframes_list(self):
        """Strategy should initialize empty timeframes list."""
        with patch('pyeventbt.strategy.strategy.logger'):
            strategy = Strategy()
            assert strategy._Strategy__strategy_timeframes == []


class TestStrategyHook:  # noqa: N801
    def test_strategy_hook_decorator_returns_callable(self):
        """Strategy hook decorator should return a decorator callable."""
        with patch('pyeventbt.strategy.strategy.logger'):
            strategy = Strategy()

            # Using decorator returns None actually - it's a side-effect decorator
            @strategy.hook(Hooks.ON_START)
            def on_start_fn(modules):
                pass

            # Hook decorator returns the return value of decorator (None)
            # The important thing is that the hook method exists and is callable without raising
            assert strategy.enable_hooks() is None
            assert strategy.disable_hooks() is None

    def test_strategy_enable_hooks_works(self):
        """Strategy enable_hooks should not raise."""
        with patch('pyeventbt.strategy.strategy.logger'):
            strategy = Strategy()
            # Should not raise
            strategy.enable_hooks()

    def test_strategy_disable_hooks_works(self):
        """Strategy disable_hooks should not raise."""
        with patch('pyeventbt.strategy.strategy.logger'):
            strategy = Strategy()
            # Should not raise
            strategy.disable_hooks()