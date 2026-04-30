"""
Tests for PyEventBT utils module.
Covers utility functions for order types, currency conversion, dates, and formatting.
"""
import pytest
import pandas as pd
from decimal import Decimal
from datetime import datetime
from unittest.mock import MagicMock, patch
from pyeventbt.utils import utils


class TestColorize:
    """Tests for colorize function."""

    def test_colorize_default_color(self):
        """colorize should wrap string with default color."""
        result = utils.colorize("test", utils.TerminalColors.OKBLUE)
        assert "test" in result
        assert utils.TerminalColors.OKBLUE in result
        assert utils.TerminalColors.ENDC in result

    def test_colorize_with_different_colors(self):
        """colorize should work with different terminal colors."""
        for color in [utils.TerminalColors.OKGREEN, utils.TerminalColors.WARNING, utils.TerminalColors.FAIL]:
            result = utils.colorize("message", color)
            assert "message" in result
            assert utils.TerminalColors.ENDC in result


class TestOrderTypeConversion:
    """Tests for order type conversion functions."""

    def test_order_type_str_to_int_buy(self):
        """Should convert BUY to 0."""
        result = utils.Utils.order_type_str_to_int("BUY")
        assert result == 0

    def test_order_type_str_to_int_sell(self):
        """Should convert SELL to 1."""
        result = utils.Utils.order_type_str_to_int("SELL")
        assert result == 1

    def test_order_type_str_to_int_all_types(self):
        """Should convert all order types correctly."""
        assert utils.Utils.order_type_str_to_int("BUY_LIMIT") == 2
        assert utils.Utils.order_type_str_to_int("SELL_LIMIT") == 3
        assert utils.Utils.order_type_str_to_int("BUY_STOP") == 4
        assert utils.Utils.order_type_str_to_int("SELL_STOP") == 5
        assert utils.Utils.order_type_str_to_int("BUY_STOP_LIMIT") == 6
        assert utils.Utils.order_type_str_to_int("SELL_STOP_LIMIT") == 7
        assert utils.Utils.order_type_str_to_int("CLOSE_BY") == 8

    def test_order_type_str_to_int_unknown(self):
        """Should return -1 for unknown order types."""
        result = utils.Utils.order_type_str_to_int("UNKNOWN")
        assert result == -1

    def test_order_type_int_to_str_buy(self):
        """Should convert 0 to BUY."""
        result = utils.Utils.order_type_int_to_str(0)
        assert result == "BUY"

    def test_order_type_int_to_str_sell(self):
        """Should convert 1 to SELL."""
        result = utils.Utils.order_type_int_to_str(1)
        assert result == "SELL"

    def test_order_type_int_to_str_all_types(self):
        """Should convert all integers correctly."""
        assert utils.Utils.order_type_int_to_str(2) == "BUY_LIMIT"
        assert utils.Utils.order_type_int_to_str(3) == "SELL_LIMIT"
        assert utils.Utils.order_type_int_to_str(4) == "BUY_STOP"
        assert utils.Utils.order_type_int_to_str(5) == "SELL_STOP"
        assert utils.Utils.order_type_int_to_str(6) == "BUY_STOP_LIMIT"
        assert utils.Utils.order_type_int_to_str(7) == "SELL_STOP_LIMIT"
        assert utils.Utils.order_type_int_to_str(8) == "CLOSE_BY"

    def test_order_type_int_to_str_unknown(self):
        """Should return UNKNOWN for unknown integers."""
        result = utils.Utils.order_type_int_to_str(99)
        assert result == "UNKNOWN"


class TestCheckNewM1BarCreatesNewTfBar:
    """Tests for check_new_m1_bar_creates_new_tf_bar function."""

    def test_5min_bar_at_00_04(self):
        """5min bar should be created at minute 4."""
        # At 10:04, the 5min bar for 10:00-10:05 is created
        latest_bar = pd.Timestamp('2024-01-15 10:04:00')
        result = utils.Utils.check_new_m1_bar_creates_new_tf_bar(latest_bar, '5min')
        assert result is True

    def test_5min_bar_at_00_03(self):
        """5min bar should NOT be created at minute 3."""
        # At 10:03, no new 5min bar
        latest_bar = pd.Timestamp('2024-01-15 10:03:00')
        result = utils.Utils.check_new_m1_bar_creates_new_tf_bar(latest_bar, '5min')
        assert result is False

    def test_1min_bar_every_minute(self):
        """1min bar should be created every minute."""
        latest_bar = pd.Timestamp('2024-01-15 10:30:00')
        result = utils.Utils.check_new_m1_bar_creates_new_tf_bar(latest_bar, '1min')
        assert result is True

    def test_15min_bar_at_00_14(self):
        """15min bar should be created at minute 14 (becomes 15 after +1min)."""
        # At 10:14, after +1min = 10:15, which is a multiple of 900 seconds (15min)
        latest_bar = pd.Timestamp('2024-01-15 10:14:00')
        result = utils.Utils.check_new_m1_bar_creates_new_tf_bar(latest_bar, '15min')
        assert result is True

    def test_1hour_bar_at_10_00(self):
        """1H bar should be created at hour 10:00 after +1min becomes 10:01 - wait, that's wrong."""
        # Actually: at 10:00, after +1min = 10:01, which is NOT a multiple of 3600
        # At 09:59, after +1min = 10:00, which IS a multiple of 3600
        latest_bar = pd.Timestamp('2024-01-15 09:59:00')
        result = utils.Utils.check_new_m1_bar_creates_new_tf_bar(latest_bar, '1H')
        assert result is True

    def test_invalid_timeframe_raises(self):
        """Invalid timeframe should raise ValueError."""
        latest_bar = pd.Timestamp('2024-01-15 10:00:00')
        with pytest.raises(ValueError, match="Invalid timeframe"):
            utils.Utils.check_new_m1_bar_creates_new_tf_bar(latest_bar, 'INVALID')


class TestCurrencyConversion:
    """Tests for currency conversion functions."""

    def test_convert_same_currency_returns_same(self):
        """Same currency conversion should return same amount."""
        mock_provider = MagicMock()
        result = utils.Utils.convert_currency_amount_to_another_currency(
            Decimal('100'), 'EUR', 'EUR', mock_provider
        )
        assert result == Decimal('100')

    def test_get_currency_conversion_multiplier_same_currency(self):
        """Same currency should return 1."""
        mock_provider = MagicMock()
        result = utils.Utils.get_currency_conversion_multiplier_cfd(
            'EUR', 'EUR', mock_provider
        )
        assert result == Decimal(1)

    def test_convert_currency_no_pair_raises(self):
        """Unsupported currency pair should raise ValueError."""
        mock_provider = MagicMock()
        with pytest.raises(ValueError, match="No FX pair found"):
            utils.Utils.convert_currency_amount_to_another_currency(
                Decimal('100'), 'EUR', 'XXX', mock_provider
            )

    def test_get_currency_conversion_multiplier_no_pair_raises(self):
        """Unsupported currency pair should raise ValueError."""
        mock_provider = MagicMock()
        with pytest.raises(ValueError, match="No FX pair found"):
            utils.Utils.get_currency_conversion_multiplier_cfd(
                'EUR', 'XXX', mock_provider
            )

    def test_convert_currency_calls_data_provider(self):
        """Should call data provider for conversion rate."""
        mock_provider = MagicMock()
        mock_provider.DATA_PROVIDER.get_latest_bid.return_value = Decimal('1.1000')
        
        result = utils.Utils.convert_currency_amount_to_another_currency(
            Decimal('100'), 'EUR', 'USD', mock_provider
        )
        
        mock_provider.DATA_PROVIDER.get_latest_bid.assert_called_once_with('EURUSD')
        assert result == Decimal('110')


class TestFxFuturesSuffix:
    """Tests for get_fx_futures_suffix function."""

    def test_january_returns_h_m(self):
        """January should return H (Mar) and M (Jun) contracts."""
        with patch('pyeventbt.utils.utils.datetime') as mock_dt:
            mock_dt.now.return_value.month = 1
            result = utils.Utils.get_fx_futures_suffix('EUR')
            assert result == ('EUR_H', 'EUR_M')

    def test_april_returns_m_u(self):
        """April should return M (Jun) and U (Sep) contracts."""
        with patch('pyeventbt.utils.utils.datetime') as mock_dt:
            mock_dt.now.return_value.month = 4
            result = utils.Utils.get_fx_futures_suffix('EUR')
            assert result == ('EUR_M', 'EUR_U')

    def test_july_returns_u_z(self):
        """July should return U (Sep) and Z (Dec) contracts."""
        with patch('pyeventbt.utils.utils.datetime') as mock_dt:
            mock_dt.now.return_value.month = 7
            result = utils.Utils.get_fx_futures_suffix('EUR')
            assert result == ('EUR_U', 'EUR_Z')

    def test_october_returns_z_h(self):
        """October should return Z (Dec) and H (Mar) contracts."""
        with patch('pyeventbt.utils.utils.datetime') as mock_dt:
            mock_dt.now.return_value.month = 10
            result = utils.Utils.get_fx_futures_suffix('EUR')
            assert result == ('EUR_Z', 'EUR_H')


class TestCapForecast:
    """Tests for cap_forecast function."""

    def test_cap_forecast_clamps_positive(self):
        """Should cap positive forecast at 20."""
        result = utils.Utils.cap_forecast(30.0)
        assert result == 20.0

    def test_cap_forecast_clamps_negative(self):
        """Should cap negative forecast at -20."""
        result = utils.Utils.cap_forecast(-30.0)
        assert result == -20.0

    def test_cap_forecast_unchanged_in_range(self):
        """Should return same value if in range."""
        result = utils.Utils.cap_forecast(10.0)
        assert result == 10.0

    def test_cap_forecast_zero(self):
        """Should handle zero correctly."""
        result = utils.Utils.cap_forecast(0)
        assert result == 0

    def test_cap_forecast_exact_boundary_positive(self):
        """Should handle exact positive boundary."""
        result = utils.Utils.cap_forecast(20.0)
        assert result == 20.0

    def test_cap_forecast_exact_boundary_negative(self):
        """Should handle exact negative boundary."""
        result = utils.Utils.cap_forecast(-20.0)
        assert result == -20.0


class TestDateprint:
    """Tests for dateprint function."""

    def test_dateprint_returns_string(self):
        """Should return a string."""
        result = utils.Utils.dateprint()
        assert isinstance(result, str)

    def test_dateprint_format(self):
        """Should return date in correct format."""
        result = utils.Utils.dateprint()
        # Should contain / separators and time
        assert '/' in result
        # Should have fractional seconds
        assert '.' in result


class TestCheckPlatformCompatibility:
    """Tests for check_platform_compatibility function."""

    def test_windows_returns_true(self):
        """Windows platform should return True."""
        with patch('pyeventbt.utils.utils.platform') as mock_platform:
            mock_platform.system.return_value = 'Windows'
            result = utils.check_platform_compatibility(raise_exception=False)
            assert result is True

    def test_non_windows_returns_false(self):
        """Non-Windows platform should return False."""
        with patch('pyeventbt.utils.utils.platform') as mock_platform:
            mock_platform.system.return_value = 'Linux'
            result = utils.check_platform_compatibility(raise_exception=False)
            assert result is False

    def test_non_windows_raises_exception(self):
        """Non-Windows with raise_exception=True should raise."""
        with patch('pyeventbt.utils.utils.platform') as mock_platform:
            mock_platform.system.return_value = 'Linux'
            with pytest.raises(Exception, match="PLATFORM_INCOMPATIBILTY"):
                utils.check_platform_compatibility(raise_exception=True)


class TestPrintPercentageBar:
    """Tests for print_percentage_bar function."""

    def test_zero_percentage(self):
        """Should handle 0% correctly."""
        # Just verify no exception
        utils.print_percentage_bar(0)

    def test_fifty_percentage(self):
        """Should handle 50% correctly."""
        utils.print_percentage_bar(50)

    def test_hundred_percentage(self):
        """Should handle 100% correctly."""
        utils.print_percentage_bar(100)

    def test_negative_raises(self):
        """Negative percentage should raise ValueError."""
        with pytest.raises(ValueError, match="Percentage must be between"):
            utils.print_percentage_bar(-5)

    def test_over_100_raises(self):
        """Percentage over 100 should raise ValueError."""
        with pytest.raises(ValueError, match="Percentage must be between"):
            utils.print_percentage_bar(150)


class TestAllFxSymbols:
    """Tests for ALL_FX_SYMBOLS constant."""

    def test_all_fx_symbols_is_tuple(self):
        """ALL_FX_SYMBOLS should be a tuple."""
        assert isinstance(utils.ALL_FX_SYMBOLS, tuple)

    def test_all_fx_symbols_contains_common_pairs(self):
        """Should contain common FX pairs."""
        assert 'EURUSD' in utils.ALL_FX_SYMBOLS
        assert 'GBPUSD' in utils.ALL_FX_SYMBOLS
        assert 'USDJPY' in utils.ALL_FX_SYMBOLS

    def test_all_fx_symbols_count(self):
        """Should contain expected number of pairs."""
        assert len(utils.ALL_FX_SYMBOLS) == 32


class TestTerminalColors:
    """Tests for TerminalColors class."""

    def test_terminal_colors_have_values(self):
        """All colors should have string values."""
        assert isinstance(utils.TerminalColors.HEADER, str)
        assert isinstance(utils.TerminalColors.OKBLUE, str)
        assert isinstance(utils.TerminalColors.OKGREEN, str)
        assert isinstance(utils.TerminalColors.WARNING, str)
        assert isinstance(utils.TerminalColors.FAIL, str)
        assert isinstance(utils.TerminalColors.ENDC, str)