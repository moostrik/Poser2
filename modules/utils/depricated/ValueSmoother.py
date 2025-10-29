import time
import numpy as np
from collections import deque
from enum import Enum
from typing import Union, List, Deque, Tuple, Optional


class SmoothingMethod(Enum):
    """Enumeration of available smoothing methods."""
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL = "exponential"
    ONE_EURO = "one_euro"
    NONE = "none"


class ValueSmoother:
    """
    Class for smoothing values with different input and output framerates.
    Handles jittery low-fps inputs and produces smooth high-fps outputs.
    """
    
    def __init__(
        self, 
        method: SmoothingMethod = SmoothingMethod.EXPONENTIAL,
        window_size: int = 5,
        alpha: float = 0.5,
        output_fps: float = 60.0,
        is_circular: bool = False,  # New parameter for circular values
        # One Euro filter params
        min_cutoff: float = 0.1,
        beta: float = 0.1,
        d_cutoff: float = 1.0
    ):
        """
        Initialize the smoother.
        
        Args:
            method: Smoothing method to use
            window_size: Window size for moving average
            alpha: Smoothing factor for exponential smoothing (0-1) Lower values = more smoothing
            output_fps: Target output framerate
            is_circular: Whether the value is circular (0 and 1 are the same)
            min_cutoff: Minimum cutoff frequency for One Euro filter
            beta: Speed coefficient for One Euro filter
            d_cutoff: Cutoff for derivative for One Euro filter
        """
        self.method = method
        self.window_size = window_size
        self.alpha = alpha
        self.output_fps = output_fps
        self.output_interval = 1.0 / output_fps
        self.is_circular = is_circular  # Store the circular flag
        
        # One Euro filter parameters
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        # History of values with timestamps
        self.history: Deque[Tuple[float, float]] = deque(maxlen=window_size)
        
        # Last output value and time
        self.last_output_value = None
        self.last_output_time = None
        
        # For One Euro filter
        self.prev_value = None
        self.prev_time = None
        self.prev_dx = 0.0
        
    def reset(self):
        """
        Reset the smoother to its initial state, clearing history and cached values.
        Call this when you want to start fresh or when there's a discontinuity in the data.
        """
        # Clear history
        self.history.clear()
        
        # Reset output tracking
        self.last_output_value = None
        self.last_output_time = None
        
        # Reset One Euro filter state
        self.prev_value = None
        self.prev_time = None
        self.prev_dx = 0.0        
        
    def add_value(self, value: float, timestamp: Optional[float] = None):
        """
        Add a new value to be smoothed.
        
        Args:
            value: The new value (between 0 and 1)
            timestamp: Optional timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Ensure value is between 0 and 1
        value = max(0.0, min(1.0, value))
            
        self.history.append((timestamp, value))
        
        # Initialize output if not done yet
        if self.last_output_value is None:
            self.last_output_value = value
            self.last_output_time = timestamp
            
        # Initialize One Euro filter state
        if self.prev_value is None:
            self.prev_value = value
            self.prev_time = timestamp
    
    def get_smoothed_value(self, current_time: Optional[float] = None) -> float | None:
        """
        Get the smoothed value for the current time.
        
        Args:
            current_time: Time to get value for (uses current time if None)
            
        Returns:
            Smoothed value at the specified time
        """
        if not self.history:
            return None
            
        if current_time is None:
            current_time = time.time()
            
        # If we need to output multiple frames since last output
        if self.last_output_time is not None:
            time_diff = current_time - self.last_output_time
            steps = int(time_diff / self.output_interval)
            
            # Generate intermediate values at output framerate
            for i in range(1, steps + 1):
                t = self.last_output_time + i * self.output_interval
                if t <= current_time:
                    self.last_output_value = self._smooth_at_time(t)
                    self.last_output_time = t
        else:
            self.last_output_value = self._smooth_at_time(current_time)
            self.last_output_time = current_time
                
        return self.last_output_value
    
    def _smooth_at_time(self, time_point: float) -> float:
        """
        Apply smoothing at a specific time point.
        
        Args:
            time_point: Time to get value for
            
        Returns:
            Smoothed value
        """
        # Get the interpolated raw value at this time
        raw_value = self._interpolate_value(time_point)
        
        if raw_value is None:
            return self.last_output_value if self.last_output_value is not None else 0.0
            
        # Apply selected smoothing method
        if self.method == SmoothingMethod.MOVING_AVERAGE:
            return self._moving_average(raw_value)
        elif self.method == SmoothingMethod.EXPONENTIAL:
            return self._exponential_smoothing(raw_value)
        elif self.method == SmoothingMethod.ONE_EURO:
            return self._one_euro_filter(raw_value, time_point)
        else:  # NONE
            return raw_value
    
    def _interpolate_value(self, time_point: float) -> Optional[float]:
        """
        Interpolate value at a specific time point from the history.
        
        Args:
            time_point: Time to interpolate at
            
        Returns:
            Interpolated value or None if no data
        """
        if not self.history:
            return None
            
        # Find the surrounding data points
        before = None
        after = None
        
        for ts, val in self.history:
            if ts <= time_point:
                before = (ts, val)
            if ts >= time_point and after is None:
                after = (ts, val)
                
        # Handle edge cases
        if before is None:
            return self.history[0][1]  # Return earliest value
        if after is None:
            return self.history[-1][1]  # Return latest value
            
        # Linear interpolation between the two points
        if before[0] == after[0]:  # Same timestamp
            return before[1]
            
        # Interpolation factor (0 to 1)
        t = (time_point - before[0]) / (after[0] - before[0])
        
        if self.is_circular:
            # Handle circular interpolation
            return self._circular_interpolate(before[1], after[1], t)
        else:
            # Standard linear interpolation
            return before[1] + t * (after[1] - before[1])
    
    def _circular_interpolate(self, a: float, b: float, t: float) -> float:
        """
        Interpolate between two circular values.
        
        Args:
            a: Starting value (0-1)
            b: Ending value (0-1)
            t: Interpolation factor (0-1)
            
        Returns:
            Interpolated circular value
        """
        # Find the shortest path between a and b on a circle
        delta = (b - a) % 1.0
        if delta > 0.5:
            delta -= 1.0  # Choose the shorter direction around the circle
            
        # Apply interpolation and wrap back to 0-1 range
        result = (a + t * delta) % 1.0
        return result
    
    def _moving_average(self, new_value: float) -> float:
        """Apply moving average smoothing."""
        if not self.is_circular:
            # Standard moving average
            values = [v for _, v in self.history]
            values.append(new_value)
            return sum(values) / len(values)
        else:
            # Circular moving average - convert to angular representation
            values = [v for _, v in self.history]
            values.append(new_value)
            
            # Convert to radians (0-1 maps to 0-2π)
            angles = [v * 2 * np.pi for v in values]
            
            # Average the sin and cos components to handle the circular nature
            sin_avg = np.mean([np.sin(angle) for angle in angles])
            cos_avg = np.mean([np.cos(angle) for angle in angles])
            
            # Convert back to 0-1 range
            result = np.arctan2(sin_avg, cos_avg) / (2 * np.pi)
            if result < 0:
                result += 1.0
                
            return result
    
    def _exponential_smoothing(self, new_value: float) -> float:
        """Apply exponential smoothing."""
        if self.last_output_value is None:
            return new_value
            
        if not self.is_circular:
            # Standard exponential smoothing
            return self.alpha * new_value + (1 - self.alpha) * self.last_output_value
        else:
            # Handle circular values
            delta = (new_value - self.last_output_value) % 1.0
            if delta > 0.5:
                delta -= 1.0  # Choose shorter path
                
            # Apply smoothing to the delta and add to last value
            smoothed_delta = self.alpha * delta
            result = (self.last_output_value + smoothed_delta) % 1.0
            return result
    
    def _one_euro_filter(self, new_value: float, timestamp: float) -> float:
        """
        Apply One Euro filter - reduces jitter while preserving quick movements.
        Based on: https://gery.casiez.net/1euro/
        """
        if self.prev_value is None:
            self.prev_value = new_value
            self.prev_time = timestamp
            return new_value
            
        dt = timestamp - self.prev_time
        if dt == 0:
            return self.last_output_value if self.last_output_value is not None else new_value
            
        # Compute velocity, handling circular values if needed
        if self.is_circular:
            # Calculate shortest delta for circular values
            delta = (new_value - self.prev_value) % 1.0
            if delta > 0.5:
                delta -= 1.0
            dx = delta / dt
        else:
            dx = (new_value - self.prev_value) / dt
        
        # Filter velocity
        dxf = self._exponential_smoothing_factor(
            dx, self.d_cutoff, dt
        )
        
        # Compute cutoff frequency
        cutoff = self.min_cutoff + self.beta * abs(dxf)
        
        # Filter position - this is what was wrong
        if self.is_circular:
            # Handle circular filtering
            te = 1.0 / (2.0 * np.pi * cutoff)
            alpha = 1.0 / (1.0 + te / dt)
            
            # Calculate shortest delta for circular values
            delta = (new_value - self.prev_value) % 1.0
            if delta > 0.5:
                delta -= 1.0
                
            # Apply smoothing to the delta and add to previous value
            smoothed_delta = alpha * delta
            result = (self.prev_value + smoothed_delta) % 1.0
        else:
            # Standard filtering for position component
            te = 1.0 / (2.0 * np.pi * cutoff)
            alpha = 1.0 / (1.0 + te / dt)
            result = alpha * new_value + (1.0 - alpha) * self.prev_value
    
        # Update state
        self.prev_value = result
        self.prev_time = timestamp
        self.prev_dx = dxf
        
        return result
        
    def _exponential_smoothing_factor(self, x: float, cutoff: float, dt: float) -> float:
        """Helper function for One Euro filter."""
        te = 1.0 / (2.0 * np.pi * cutoff)
        alpha = 1.0 / (1.0 + te / dt)
        return alpha * x + (1.0 - alpha) * (self.prev_dx if hasattr(self, 'prev_dx') else 0)