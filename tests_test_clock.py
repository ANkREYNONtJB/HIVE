import pytest
from datetime import datetime
import pytz
from sonw.time.clock import WorldClock
import tkinter as tk
import threading
import time

@pytest.fixture
def clock():
    clock = WorldClock(['UTC', 'US/Pacific'])
    yield clock
    clock.stop()

def test_clock_initialization(clock):
    assert len(clock.timezones) == 2
    assert 'UTC' in clock.timezones
    assert 'US/Pacific' in clock.timezones
    assert isinstance(clock.root, tk.Tk)
    assert clock.running
    assert isinstance(clock.update_thread, threading.Thread)

def test_timezone_addition(clock):
    clock.add_timezone('Europe/London')
    assert 'Europe/London' in clock.timezones
    assert 'Europe/London' in clock.time_labels

def test_timezone_removal(clock):
    clock.remove_timezone('US/Pacific')
    assert 'US/Pacific' not in clock.timezones
    assert 'US/Pacific' not in clock.time_labels

def test_time_update(clock):
    # Let the clock update a few times
    time.sleep(0.3)
    
    for zone in clock.timezones:
        label_text = clock.time_labels[zone].cget("text")
        assert len(label_text) == 8  # HH:MM:SS format
        
        # Verify time format
        try:
            datetime.strptime(label_text, "%H:%M:%S")
        except ValueError:
            pytest.fail(f"Invalid time format for {zone}: {label_text}")

def test_invalid_timezone():
    with pytest.raises(Exception):
        WorldClock(['InvalidZone'])