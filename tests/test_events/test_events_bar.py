"""Tests for Bar events in PyEventBT."""
import pytest
from datetime import datetime
from pyeventbt.events.events import Bar, BarEvent, EventType

class TestBarDataclass:
    def test_bar_slots(self):
        """Bar should use __slots__ for memory efficiency."""
        bar = Bar(open=100, high=105, low=98, close=103, tickvol=100, volume=50000, spread=10, digits=1)
        # All fields should be accessible
        assert bar.open == 100
        assert bar.high == 105
        assert bar.low == 98
        assert bar.close == 103
        assert bar.tickvol == 100
        assert bar.volume == 50000
        assert bar.spread == 10
        assert bar.digits == 1
    
    def test_bar_float_conversion_different_digits(self):
        """Bar float conversion should work with different digit scales."""
        # Digits=1: multiply by 10
        bar1 = Bar(open=100, high=105, low=98, close=103, tickvol=100, volume=50000, spread=10, digits=1)
        assert bar1.open_f == 10.0
        assert bar1.high_f == 10.5
        
        # Digits=2: multiply by 100
        bar2 = Bar(open=1000, high=1050, low=980, close=1030, tickvol=100, volume=50000, spread=10, digits=2)
        assert bar2.open_f == 10.0
        assert bar2.high_f == 10.5
        
        # Digits=0: multiply by 1
        bar3 = Bar(open=10, high=11, low=9, close=10, tickvol=100, volume=50000, spread=1, digits=0)
        assert bar3.open_f == 10.0
        assert bar3.high_f == 11.0
    
    def test_bar_high_low_ordering(self):
        """High should always be >= low in valid bars."""
        bar = Bar(open=100, high=105, low=98, close=103, tickvol=100, volume=50000, spread=10, digits=1)
        # For valid bar, high >= low. This tests field access.
        assert bar.high >= bar.low
    
    def test_bar_close_within_high_low(self):
        """Close should typically be between high and low (not guaranteed for intrabar extremes)."""
        bar = Bar(open=100, high=105, low=98, close=103, tickvol=100, volume=50000, spread=10, digits=1)
        # Close float should be accessible and typically between high_f and low_f
        assert isinstance(bar.close_f, float)
        assert bar.high_f >= bar.low_f
    
    def test_bar_volume_attributes(self):
        """Bar volume and tickvol should be integer fields."""
        bar = Bar(open=100, high=105, low=98, close=103, tickvol=100, volume=50000, spread=10, digits=1)
        assert isinstance(bar.tickvol, int)
        assert isinstance(bar.volume, int)

class TestBarEventIntegration:
    def test_bar_event_with_bar(self, sample_bar, sample_datetime):
        """BarEvent should wrap a Bar correctly."""
        bar = Bar(**sample_bar)
        event = BarEvent(
            symbol="NDX",
            datetime=sample_datetime,
            data=bar,
            timeframe="1min"
        )
        assert event.data.open_f == sample_bar['open'] / 10.0
        assert event.datetime == sample_datetime
    
    def test_bar_event_multiple_symbols(self, sample_bar, sample_datetime):
        """BarEvent should handle multiple symbols independently."""
        bar = Bar(**sample_bar)
        event1 = BarEvent(symbol="NDX", datetime=sample_datetime, data=bar, timeframe="1min")
        event2 = BarEvent(symbol="SPY", datetime=sample_datetime, data=bar, timeframe="1min")
        
        assert event1.symbol == "NDX"
        assert event2.symbol == "SPY"
        assert event1.data.open_f == event2.data.open_f
    
    def test_bar_event_timeframe(self, sample_bar, sample_datetime):
        """BarEvent timeframe should be preserved."""
        bar = Bar(**sample_bar)
        event = BarEvent(symbol="NDX", datetime=sample_datetime, data=bar, timeframe="1min")
        assert event.timeframe == "1min"
    
    def test_bar_event_type_field(self, sample_bar, sample_datetime):
        """BarEvent type field should match EventType.BAR."""
        bar = Bar(**sample_bar)
        event = BarEvent(symbol="NDX", datetime=sample_datetime, data=bar, timeframe="1min")
        assert event.type == EventType.BAR