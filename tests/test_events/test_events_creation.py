"""
Tests for PyEventBT events creation and initialization.
"""
import pytest
from datetime import datetime
from decimal import Decimal
from pydantic import ValidationError
from pyeventbt.events.events import (
    EventType, SignalType, OrderType, DealType, EventBase, BarEvent, Bar
)

class TestEventType:
    def test_event_type_values(self):
        """EventType enum should have expected values."""
        assert EventType.BAR == "BAR"
        assert EventType.SIGNAL == "SIGNAL"
        assert EventType.ORDER == "ORDER"
        assert EventType.FILL == "FILL"
        assert EventType.SCHEDULED_EVENT == "SCHEDULED_EVENT"

    def test_event_type_is_string_enum(self):
        """EventType should be usable as a string."""
        event_type = EventType.BAR
        assert event_type == "BAR"
        # Pydantic enums use format "EventType.BAR" in str()
        assert "BAR" in str(event_type)

class TestSignalType:
    def test_signal_type_values(self):
        """SignalType should have BUY and SELL values."""
        assert SignalType.BUY == "BUY"
        assert SignalType.SELL == "SELL"

class TestOrderType:
    def test_order_type_values(self):
        """OrderType should have expected values."""
        assert OrderType.MARKET == "MARKET"
        assert OrderType.LIMIT == "LIMIT"
        assert OrderType.STOP == "STOP"
        assert OrderType.CONT == "CONT"

class TestDealType:
    def test_deal_type_values(self):
        """DealType should have IN and OUT values."""
        assert DealType.IN == "IN"
        assert DealType.OUT == "OUT"


class TestBar:
    def test_bar_creation_with_integers(self):
        """Bar should accept integer values and scale correctly."""
        bar = Bar(open=100, high=105, low=98, close=103, tickvol=100, volume=50000, spread=10, digits=1)
        assert bar.open == 100
        assert bar.high == 105
        assert bar.low == 98
        assert bar.close == 103
        assert bar.digits == 1
    
    def test_bar_float_properties(self):
        """Bar float properties should return scaled values."""
        bar = Bar(open=100, high=105, low=98, close=103, tickvol=100, volume=50000, spread=10, digits=1)
        assert bar.open_f == 10.0
        assert bar.high_f == 10.5
        assert bar.low_f == 9.8
        assert bar.close_f == 10.3
    
    def test_bar_spread_f(self):
        """Bar spread_f should return scaled spread."""
        bar = Bar(open=100, high=105, low=98, close=103, tickvol=100, volume=50000, spread=10, digits=1)
        assert bar.spread_f == 1.0
    
    def test_bar_price_factor(self):
        """Bar price_factor should compute correctly."""
        bar = Bar(open=100, high=105, low=98, close=103, tickvol=100, volume=50000, spread=10, digits=1)
        assert bar.price_factor == 10.0


class TestBarEvent:
    def test_bar_event_creation(self, sample_bar, sample_datetime):
        """BarEvent should be created correctly."""
        bar = Bar(**sample_bar)
        event = BarEvent(
            symbol="NDX",
            datetime=sample_datetime,
            data=bar,
            timeframe="1min"
        )
        assert event.symbol == "NDX"
        assert event.timeframe == "1min"
        assert event.type == EventType.BAR
    
    def test_bar_event_bar_data_access(self, sample_bar, sample_datetime):
        """BarEvent should allow access to bar data."""
        bar = Bar(**sample_bar)
        event = BarEvent(
            symbol="NDX",
            datetime=sample_datetime,
            data=bar,
            timeframe="1min"
        )
        assert event.data.open_f == 10.0
        assert event.data.high_f == 10.5
    
    def test_bar_event_type_auto_set(self, sample_bar, sample_datetime):
        """BarEvent type should default to BAR."""
        bar = Bar(**sample_bar)
        event = BarEvent(
            symbol="NDX",
            datetime=sample_datetime,
            data=bar,
            timeframe="1min"
        )
        assert event.type == EventType.BAR