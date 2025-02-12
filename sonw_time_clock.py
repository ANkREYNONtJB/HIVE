from datetime import datetime
import pytz
from typing import Dict, List, Optional
import tkinter as tk
from tkinter import ttk
import threading
import time

class WorldClock:
    """Digital clock widget displaying multiple timezones"""
    def __init__(self, timezones: Optional[List[str]] = None):
        self.timezones = timezones or [
            'UTC',
            'US/Pacific',
            'US/Eastern', 
            'Europe/London',
            'Asia/Tokyo'
        ]
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("World Clock")
        
        # Style configuration
        style = ttk.Style()
        style.configure(
            "Clock.TLabel",
            font=('Arial', 14),
            padding=10
        )
        
        # Create frames for each timezone
        self.time_labels: Dict[str, ttk.Label] = {}
        for zone in self.timezones:
            frame = ttk.Frame(self.root, padding="10")
            frame.pack(fill=tk.X)
            
            # Zone label
            ttk.Label(
                frame,
                text=f"{zone}:",
                style="Clock.TLabel"
            ).pack(side=tk.LEFT)
            
            # Time label
            self.time_labels[zone] = ttk.Label(
                frame,
                text="",
                style="Clock.TLabel"
            )
            self.time_labels[zone].pack(side=tk.LEFT)
            
        # Start update thread
        self.running = True
        self.update_thread = threading.Thread(
            target=self._update_time,
            daemon=True
        )
        self.update_thread.start()
        
    def _update_time(self):
        """Update time for all timezones"""
        while self.running:
            for zone in self.timezones:
                try:
                    tz = pytz.timezone(zone)
                    current = datetime.now(tz)
                    time_str = current.strftime("%H:%M:%S")
                    
                    # Update label using tkinter's thread-safe method
                    self.root.after(
                        0,
                        self.time_labels[zone].configure,
                        {"text": time_str}
                    )
                except Exception as e:
                    print(f"Error updating {zone}: {e}")
                    
            time.sleep(0.1)  # Update every 100ms
            
    def add_timezone(self, zone: str):
        """Add a new timezone to display"""
        if zone not in self.timezones:
            self.timezones.append(zone)
            
            frame = ttk.Frame(self.root, padding="10")
            frame.pack(fill=tk.X)
            
            ttk.Label(
                frame,
                text=f"{zone}:",
                style="Clock.TLabel"
            ).pack(side=tk.LEFT)
            
            self.time_labels[zone] = ttk.Label(
                frame,
                text="",
                style="Clock.TLabel"
            )
            self.time_labels[zone].pack(side=tk.LEFT)
            
    def remove_timezone(self, zone: str):
        """Remove a timezone from display"""
        if zone in self.timezones:
            self.timezones.remove(zone)
            self.time_labels[zone].master.destroy()
            del self.time_labels[zone]
            
    def start(self):
        """Start the clock"""
        self.root.mainloop()
        
    def stop(self):
        """Stop the clock and cleanup"""
        self.running = False
        self.update_thread.join(timeout=1.0)
        self.root.destroy()

def main():
    # Create clock with default timezones
    clock = WorldClock()
    
    try:
        # Add some additional timezones
        clock.add_timezone('Europe/Paris')
        clock.add_timezone('Asia/Singapore')
        
        # Start the clock
        clock.start()
    except KeyboardInterrupt:
        clock.stop()
    finally:
        print("Clock stopped")

if __name__ == "__main__":
    main()