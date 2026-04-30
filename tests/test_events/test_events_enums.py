"""Tests for PyEventBT events enums."""
import pytest
from pyeventbt.events.events import (
    EventType, SignalType, OrderType, DealType
)

class TestEventTypeEnum:
    def test_event_type_all_values_exist(self):
        """All expected EventType values should exist."""
        assert hasattr(EventType, 'BAR')
        assert hasattr(EventType, 'SIGNAL')
        assert hasattr(EventType, 'ORDER')
        assert hasattr(EventType, 'FILL')
        assert hasattr(EventType, 'SCHEDULED_EVENT')
    
    def test_event_type_string_representation(self):
        """EventType should have proper string representation."""
        # Pydantic enums use format "EventType.BAR"
        assert "BAR" in str(EventType.BAR)
        assert "SIGNAL" in str(EventType.SIGNAL)
        assert "ORDER" in str(EventType.ORDER)
        assert "FILL" in str(EventType.FILL)
        assert "SCHEDULED_EVENT" in str(EventType.SCHEDULED_EVENT)
    
    def test_event_type_comparison(self):
        """EventType should compare correctly to strings."""
        assert EventType.BAR == "BAR"
        assert EventType.SIGNAL != "BAR"
        assert EventType.ORDER != "SIGNAL"
    
    def test_event_type_from_string(self):
        """EventType should be constructable from string."""
        assert EventType("BAR") == EventType.BAR
        assert EventType("SIGNAL") == EventType.SIGNAL

class TestSignalTypeEnum:
    def test_signal_type_values(self):
        """SignalType should have BUY and SELL."""
        # Pydantic enums use format "SignalType.BUY"
        assert "BUY" in str(SignalType.BUY)
        assert "SELL" in str(SignalType.SELL)
    
    def test_signal_type_comparison(self):
        """SignalType should compare correctly."""
        assert SignalType.BUY == "BUY"
        assert SignalType.SELL == "SELL"
        assert SignalType.BUY != SignalType.SELL

class TestOrderTypeEnum:
    def test_order_type_all_values(self):
        """OrderType should have all expected values."""
        # Pydantic enums use format "OrderType.MARKET"
        assert "MARKET" in str(OrderType.MARKET)
        assert "LIMIT" in str(OrderType.LIMIT)
        assert "STOP" in str(OrderType.STOP)
        assert "CONT" in str(OrderType.CONT)
    
    def test_order_type_count(self):
        """OrderType should have exactly 4 values."""
        values = list(OrderType)
        assert len(values) == 4

class TestDealTypeEnum:
    def test_deal_type_values(self):
        """DealType should have IN and OUT."""
        # Pydantic enums use format "DealType.IN"
        assert "IN" in str(DealType.IN)
        assert "OUT" in str(DealType.OUT)
    
    def test_deal_type_comparison(self):
        """DealType comparison."""
        assert DealType.IN == "IN"
        assert DealType.OUT == "OUT"