from OneEuroFilter import OneEuroFilter as _OneEuroFilter

import math

class OneEuroFilter(_OneEuroFilter):

    def __init__(self, freq: float, mincutoff: float = 1, beta: float = 0, dcutoff: float = 1) -> None:
        super().__init__(freq, mincutoff, beta, dcutoff)

    @property
    def value(self) -> float:
        """Current filtered value"""
        return self.__x.lastFilteredValue()

    @property
    def target(self) -> float:
        """Current target value"""
        return self.__x.lastValue()

class OneEuroFilterAngular():

    def __init__(self, freq:float, mincutoff:float=1.0, beta:float=0.0, dcutoff:float=1.0) -> None:
        self.__sin_interp = OneEuroFilter(freq, mincutoff, beta, dcutoff)
        self.__cos_interp = OneEuroFilter(freq, mincutoff, beta, dcutoff)
        self.__value: float = 0.0
        self.__target: float = 0.0

    def __call__(self, x:float, timestamp: float | None = None) -> float:
        self.__target = x
        sin_value: float = math.sin(x)
        cos_value: float = math.cos(x)
        filtered_sin: float = self.__sin_interp(sin_value, timestamp) # type: ignore
        filtered_cos: float = self.__cos_interp(cos_value, timestamp) # type: ignore
        self.__value = math.atan2(filtered_sin, filtered_cos)
        return self.__value

    @property
    def value(self) -> float:
        return self.__value

    @property
    def target(self) -> float:
        return self.__target

    def filter(self, x:float, timestamp: float | None = None) -> float:
        return self.__call__(x, timestamp)

    def setFrequency(self, freq:float) -> None:
        self.__sin_interp.setFrequency(freq)
        self.__cos_interp.setFrequency(freq)

    def setMinCutoff(self, mincutoff:float) -> None:
        self.__sin_interp.setMinCutoff(mincutoff)
        self.__cos_interp.setMinCutoff(mincutoff)

    def setBeta(self, beta:float) -> None:
        self.__sin_interp.setBeta(beta)
        self.__cos_interp.setBeta(beta)

    def setDerivateCutoff(self, dcutoff:float) -> None:
        self.__sin_interp.setDerivateCutoff(dcutoff)
        self.__cos_interp.setDerivateCutoff(dcutoff)

    def setParameters(self, freq: float,  mincutoff: float=1.0, beta:float=0.0, dcutoff=1.0) -> None:
        self.__sin_interp.setParameters(freq, mincutoff, beta, dcutoff)
        self.__cos_interp.setParameters(freq, mincutoff, beta, dcutoff)

    def reset(self) -> None:
        self.__sin_interp.reset()
        self.__cos_interp.reset()

class SpringFilter:
    """Simple spring-damper system for smooth value interpolation.

    :freq: Update frequency in Hz (for dt calculation if timestamp not provided)
    :frequency: Natural frequency in Hz (higher = faster response, typical: 1-5)
    :damping: Damping ratio (0.7-1.0 = critical damping, no overshoot)
    :initial_value: Starting value
    """

    def __init__(self, freq: float, responsiveness: float = 2.0, damping: float = 0.8, initial_value: float = 0.0) -> None:
        if freq <= 0:
            raise ValueError("freq should be >0")
        if responsiveness <= 0:
            raise ValueError("frequency should be >0")
        if damping <= 0:
            raise ValueError("damping should be >0")

        self.__freq = float(freq)
        self.__responsiveness = float(responsiveness)
        self.__damping = float(damping)
        self.__target = float(initial_value)
        self.__value = float(initial_value)
        self.__velocity = 0.0
        self.__lasttime = None

    @property
    def value(self) -> float:
        """Current filtered value"""
        return self.__value

    @property
    def target(self) -> float:
        """Current target value"""
        return self.__target

    def __call__(self, x: float, timestamp: float | None = None) -> float:
        """Update the spring simulation with a new target value.

        :param x: New target value
        :param timestamp: Timestamp in seconds (optional)
        :returns: Current smoothed value
        """
        # Calculate dt based on timestamps or use fixed frequency
        if self.__lasttime and timestamp and timestamp > self.__lasttime:
            dt: float = timestamp - self.__lasttime
        else:
            dt = 1.0 / self.__freq
        self.__lasttime = timestamp
        # Update target
        self.__target = float(x)
        # Spring-damper physics
        omega = 2.0 * math.pi * self.__responsiveness

        # Spring force and damping
        spring_force = omega * omega * (self.__target - self.__value)
        damping_force = 2.0 * self.__damping * omega * self.__velocity

        # Update velocity and position
        self.__velocity += (spring_force - damping_force) * dt
        self.__value += self.__velocity * dt

        return self.__value

    def filter(self, x: float, timestamp: float | None = None) -> float:
        """Update the spring simulation with a new target value."""
        return self.__call__(x, timestamp)

    def setFrequency(self, freq: float) -> None:
        """ Sets the frequency of the input signal.

        :param freq: An estimate of the frequency in Hz of the signal (> 0), if timestamps are not available.
        :raises ValueError: If one of the frequency is not >0
        """
        if freq <= 0:
            raise ValueError("freq should be >0")
        else:
            self.__freq = float(freq)

    def setResponsiveness(self, responsiveness: float) -> None:
        """ Sets the natural responsiveness of the spring.

        :param responsiveness: Natural responsiveness (> 0). Higher = faster response.
        :raises ValueError: If responsiveness is not >0
        """
        if responsiveness <= 0:
            raise ValueError("responsiveness should be >0")
        else:
            self.__responsiveness = float(responsiveness)

    def setDamping(self, damping: float) -> None:
        """ Sets the damping ratio.

        :param damping: Damping ratio (> 0)
                        0.7-1.0 = critical damping (no overshoot)
                        < 0.7 = underdamped (bouncy)
                        > 1.0 = overdamped (sluggish)
        :raises ValueError: If damping is not >0
        """
        if damping <= 0:
            raise ValueError("damping should be >0")
        else:
            self.__damping = float(damping)

    def setParameters(self, freq: float, responsiveness: float = 2.0, damping: float = 0.8) -> None:
        """Sets all parameters of the spring-damper.

        :param freq: Update frequency in Hz (> 0)
        :param responsiveness: Natural responsiveness (> 0)
        :param damping: Damping ratio (> 0)
        :raises ValueError: If any parameter is not >0
        """
        self.setFrequency(freq)
        self.setResponsiveness(responsiveness)
        self.setDamping(damping)

    def reset(self, initial_value: float = 0.0) -> None:
        """Resets the internal state of the spring-damper.

        :param initial_value: Value to reset to (default: 0.0)
        """
        self.__value = float(initial_value)
        self.__velocity = 0.0
        self.__target = float(initial_value)
        self.__lasttime = None

class SpringFilterAngular(SpringFilter):
    """Angular version of SpringDamper, handles angle wrapping.

    :freq: Update frequency in Hz (for dt calculation if timestamp not provided)
    :frequency: Natural frequency in Hz (higher = faster response, typical: 1-5)
    :damping: Damping ratio (0.7-1.0 = critical damping, no overshoot)
    :initial_value: Starting angle in radians
    """

    def __init__(self, freq: float, responsiveness: float = 2.0, damping: float = 0.8, initial_value: float = 0.0) -> None:
        super().__init__(freq, responsiveness, damping, initial_value)
        self.__last_input = initial_value  # Track last wrapped input

    @property
    def value(self) -> float:
        """Current filtered angle in radians, wrapped to [-π, π]"""
        raw_value = super().value
        return (raw_value + math.pi) % (2 * math.pi) - math.pi

    @property
    def target(self) -> float:
        """Current target angle in radians (last input), wrapped to [-π, π]"""
        return self.__last_input

    def __call__(self, x: float, timestamp: float | None = None) -> float:
        """Update the spring simulation with a new target angle, handling wrap-around.

        :param x: New target angle in radians
        :param timestamp: Timestamp in seconds (optional)
        :returns: Current smoothed angle in radians, wrapped to [-π, π]
        """
        # Wrap input to [-π, π]
        x_wrapped = (x + math.pi) % (2 * math.pi) - math.pi
        self.__last_input = x_wrapped

        # Calculate shortest angular difference from last input
        # Use the parent's unwrapped value for continuous motion
        current_unwrapped = super().value
        delta = (x_wrapped - ((current_unwrapped + math.pi) % (2 * math.pi) - math.pi) + math.pi) % (2 * math.pi) - math.pi
        new_target = current_unwrapped + delta

        # Update spring with unwrapped continuous target
        result = super().__call__(new_target, timestamp)

        # Return wrapped result
        return (result + math.pi) % (2 * math.pi) - math.pi


    def reset(self, initial_value: float = 0.0) -> None:
        """Resets the internal state.

        :param initial_value: Angle to reset to in radians
        """
        wrapped = (initial_value + math.pi) % (2 * math.pi) - math.pi
        super().reset(wrapped)
        self.__last_input = wrapped

class EMAFilter:
    """Simple exponential smoothing (EMA/LERP).

    :freq: Update frequency in Hz (for timestamp support)
    :alpha: Smoothing factor (0-1). Higher = more responsive.
            Typical: 0.1-0.3 for smooth, 0.5-0.8 for responsive
    :initial_value: Starting value
    """

    def __init__(self, freq: float, alpha: float = 0.2, initial_value: float = 0.0) -> None:
        if freq <= 0:
            raise ValueError("freq should be >0")
        if alpha <= 0 or alpha > 1.0:
            raise ValueError("alpha should be in (0.0, 1.0]")

        self.__freq = float(freq)
        self.__alpha = float(alpha)
        self.__value = float(initial_value)
        self.__lastvalue = None
        self.__lasttime = None

    @property
    def value(self) -> float:
        """Current filtered value"""
        return self.__value

    @property
    def target(self) -> float | None:
        """Last input value"""
        return self.__lastvalue

    def __call__(self, x: float, timestamp: float | None = None) -> float:
        """Update the smoothed value.

        :param x: New input value
        :param timestamp: Timestamp in seconds (optional, for compatibility)
        :returns: Current smoothed value
        """
        self.__lasttime = timestamp
        self.__lastvalue = x
        self.__value += (x - self.__value) * self.__alpha
        return self.__value

    def filter(self, x: float, timestamp: float | None = None) -> float:
        """Filters a noisy value.

        :param x: Noisy value to filter
        :param timestamp: Timestamp in seconds (optional)
        :returns: The filtered value
        """
        return self.__call__(x, timestamp)

    def setFrequency(self, freq: float) -> None:
        """Sets the frequency (for compatibility with OneEuroFilter interface).

        :param freq: Frequency in Hz (> 0)
        :raises ValueError: If frequency is not >0
        """
        if freq <= 0:
            raise ValueError("freq should be >0")
        else:
            self.__freq = float(freq)

    def setAlpha(self, alpha: float) -> None:
        """Sets the smoothing factor.

        :param alpha: Smoothing factor (0-1). Higher = more responsive.
        :raises ValueError: If alpha is not in (0, 1]
        """
        if alpha <= 0 or alpha > 1.0:
            raise ValueError("alpha should be in (0.0, 1.0]")
        else:
            self.__alpha = float(alpha)

    def setParameters(self, freq: float, alpha: float = 0.2) -> None:
        """Sets all parameters.

        :param freq: Update frequency in Hz (> 0)
        :param alpha: Smoothing factor (0-1)
        :raises ValueError: If any parameter is invalid
        """
        self.setFrequency(freq)
        self.setAlpha(alpha)

    def reset(self, value: float = 0.0) -> None:
        """Resets the internal state.

        :param value: Value to reset to (default: 0.0)
        """
        self.__value = float(value)
        self.__lastvalue = None
        self.__lasttime = None

class EMAFilterAngular(EMAFilter):
    """Angular version of ExponentialSmoothing, handles angle wrapping.

    :freq: Update frequency in Hz (for timestamp support)
    :alpha: Smoothing factor (0-1). Higher = more responsive.
            Typical: 0.1-0.3 for smooth, 0.5-0.8 for responsive
    :initial_value: Starting angle in radians
    """

    def __init__(self, freq: float, alpha: float = 0.2, initial_value: float = 0.0) -> None:
        # Wrap initial value
        wrapped_initial = (initial_value + math.pi) % (2 * math.pi) - math.pi
        super().__init__(freq, alpha, wrapped_initial)
        self.__last_input = wrapped_initial  # Track last wrapped input

    @property
    def value(self) -> float:
        """Current filtered angle in radians, wrapped to [-π, π]"""
        raw_value = super().value
        return (raw_value + math.pi) % (2 * math.pi) - math.pi

    @property
    def target(self) -> float | None:
        """Last input angle in radians, wrapped to [-π, π]"""
        return self.__last_input

    def __call__(self, x: float, timestamp: float | None = None) -> float:
        """Update the smoothed angle value, handling wrap-around.

        :param x: New input angle in radians
        :param timestamp: Timestamp in seconds (optional, for compatibility)
        :returns: Current smoothed angle in radians, wrapped to [-π, π]
        """
        # Wrap input to [-π, π]
        x_wrapped = (x + math.pi) % (2 * math.pi) - math.pi
        self.__last_input = x_wrapped

        # Get current unwrapped value
        current_unwrapped = super().value

        # Calculate shortest angular difference
        delta = (x_wrapped - ((current_unwrapped + math.pi) % (2 * math.pi) - math.pi) + math.pi) % (2 * math.pi) - math.pi

        # Apply to unwrapped continuous target
        new_target = current_unwrapped + delta

        # Update with unwrapped value
        result = super().__call__(new_target, timestamp)

        # Return wrapped result
        return (result + math.pi) % (2 * math.pi) - math.pi

    def reset(self, value: float = 0.0) -> None:
        """Resets the internal state.

        :param value: Angle to reset to in radians
        """
        wrapped = (value + math.pi) % (2 * math.pi) - math.pi
        super().reset(wrapped)
        self.__last_input = wrapped