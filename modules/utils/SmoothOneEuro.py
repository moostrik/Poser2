from OneEuroFilter import OneEuroFilter
import time
import math
START_TIME: float = time.time()

class SmoothOneEuro:
    def __init__(self, freq=60.0, responsiveness=0.5, smoothness=0.5):
        self.filter = OneEuroFilter(freq=freq)
        self.set_responsiveness(responsiveness)
        self.set_smoothness(smoothness)
        self.input_value: float | None = None
        self.output_value: float | None = None

    def add_value(self, value: float) -> None:
        self.input_value = value
        
    def update(self) -> None:
        time_elapsed = time.time() - START_TIME
        if self.input_value is not None:
            self.output_value = self.filter(self.input_value, time_elapsed)
        
    def get_smoothed_value(self) -> float | None:
        return self.output_value
    
    def set_smoothness(self, smoothness: float) -> None:
        """
        Set the smoothness (0.0-1.0), higher is smoother.
        This adjusts how aggressively the filter smooths out noise.
        Higher values remove more noise but increase lag.
        """
        smoothness = min(max(smoothness, 0.0), 1.0)
        mincutoff = 1.0 - (smoothness * 0.9)
        self.filter.setMinCutoff(mincutoff)
    
    def set_responsiveness(self, responsiveness: float) -> None:
        """
        Set the responsiveness (0.0-1.0), higher is more responsive.        
        This adjusts how quickly the filter responds to changes.
        Higher values make the filter track input more closely but may
        let through more noise.
        """
        responsiveness = min(max(responsiveness, 0.0), 1.0)
        beta = responsiveness * 0.5
        self.filter.setBeta(beta)
    
    def reset(self) -> None:
        self.filter.reset()
        self.input_value = None
        self.output_value = None
        
        
class SmoothOneEuroCircular:
    def __init__(self, freq=60.0, responsiveness=0.5, smoothness=0.5):
        self.fx = OneEuroFilter(freq=freq)
        self.fy = OneEuroFilter(freq=freq)
        self.set_responsiveness(responsiveness)
        self.set_smoothness(smoothness)
        self.x: float | None = None
        self.y: float | None = None
        self.output_value: float | None = None
        
    def add_value(self, value: float) -> None:
        """
        value_norm: angle normalized in [0,1], where 0=-pi, 1=pi
        """
        # Convert normalized [0,1] → angle in [-pi, pi]
        # angle = (value * 2 * math.pi) - math.pi
        angle = value
        
        self.x = math.cos(angle)
        self.y = math.sin(angle) 
        
    def update(self) -> None:
        time_elapsed = time.time() - START_TIME
        if self.x is not None and self.y is not None:
            x_smooth = self.fx(self.x, time_elapsed)
            y_smooth = self.fy(self.y, time_elapsed)
            
            value = math.atan2(y_smooth, x_smooth)
            normalized_value = (value + math.pi) / (2 * math.pi)

            self.output_value = normalized_value

    def get_smoothed_value(self) -> float | None:
        return self.output_value
    
    def set_smoothness(self, smoothness: float) -> None:
        smoothness = min(max(smoothness, 0.0), 1.0)
        mincutoff = 1.0 - (smoothness * 0.9)
        self.fx.setMinCutoff(mincutoff)
        self.fy.setMinCutoff(mincutoff)

    def set_responsiveness(self, responsiveness: float) -> None:
        responsiveness = min(max(responsiveness, 0.0), 1.0)
        beta = responsiveness * 0.5
        self.fx.setBeta(beta)
        self.fy.setBeta(beta)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()
        self.x = None
        self.y = None
        self.output_value = None