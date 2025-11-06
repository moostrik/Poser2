# Standard library imports
from collections import deque
import math
from time import time
import warnings

# Third-party imports
import numpy as np

# Local application imports
from modules.utils.HotReloadMethods import HotReloadMethods

MAX_EXTRAPOLATION = 1.1  # 10% overshoot for timing jitter


class ScalarPredictiveHermite:
    """Numeric value interpolator using cubic Hermite interpolation with velocity smoothing.

    Interpolates between input samples using cubic Hermite splines, with velocity smoothing
    and one-step prediction for minimal latency and smooth transitions.

    Attributes:
        interpolated_value: Current interpolated value. NaN until first sample arrives.

    Note:
        Small discontinuities may occur when new samples arrive if the actual value
        differs from the prediction. The smoothness depends on input signal characteristics
        and the alpha_v smoothing parameter.

    Note:
        Velocity is initialized to zero, resulting in a ramp-up effect on the first
        sample pair: v_curr = (1 - alpha_v) * v_inst. This provides smooth startup
        behavior. The same ramp-up occurs when recovering from NaN input.
    """

    def __init__(self, input_rate: float, alpha_v: float = 0.45) -> None:
        """Initialize the interpolator.

        Args:
            input_rate: Sampling rate of the input data (e.g., 23 Hz).
            alpha_v: Smoothing factor for velocity (0.0 to 1.0, default: 0.45).
                Higher values = more smoothing but more lag.
                Lower values = less smoothing but faster response.

        Note:
            On first valid sample pair (or when recovering from NaN), the instantaneous
            velocity is used directly. Subsequent samples use exponential smoothing.
        """
        self.input_rate: float = input_rate
        self.alpha_v: float = alpha_v
        self.interval: float = 1.0 / input_rate

        # State variables
        self.buffer: deque[float] = deque(maxlen=2)
        self.v_curr: float = 0.0  # ✅ Change from nan to 0.0
        self.p_pred: float = math.nan
        self.last_sample_time: float = 0.0
        self.interpolated_value: float = math.nan
        self._initialized = False

    def add_sample(self, value: float, sample_time: float | None = None) -> None:
        """Add a new sample to the interpolator.

        Args:
            value: New input sample.
            sample_time: Timestamp in seconds when the sample was captured.
                If None, uses time() (not recommended for best accuracy).

        Note:
            For accurate interpolation, always provide the actual capture timestamp,
            not the arrival time. This accounts for processing delays.
        """
        if sample_time is None:
            sample_time = time()

        self.buffer.append(value)
        self.last_sample_time: float = sample_time

        # Initialize on first sample
        if not self._initialized:
            self.interpolated_value = value
            self._initialized = True
            return

        # Calculate velocity and prediction (buffer has 2 samples after first call)
        self.v_curr, self.p_pred = self._calculate_velocity_and_prediction(
            self.buffer[-2], self.buffer[-1], self.v_curr, self.alpha_v, self.interval
        )

    def update(self, current_time: float | None = None) -> None:
        """Update the interpolated value for the current time.

        Updates self.interpolated_value based on elapsed time since the last sample.

        Args:
            current_time: Optional timestamp in seconds. If None, uses time().
        """
        if not self._initialized:
            return  # No data yet

        if len(self.buffer) < 2:
            # Only one sample available, interpolated_value already set to that sample
            return

        if current_time is None:
            current_time = time()

        # Validate time ordering and clamp to prevent negative elapsed time
        if current_time < self.last_sample_time:
            warnings.warn("Interpolator received out-of-order time update.", RuntimeWarning)
            current_time = self.last_sample_time

        elapsed_time: float = current_time - self.last_sample_time
        s: float = elapsed_time / self.interval  # Normalize: 0 = at last sample, 1 = one input period later

        # Allow slight overshoot for timing jitter (up to 10% beyond expected sample time)
        # This handles PC scheduling delays gracefully while preventing unbounded extrapolation
        s = max(0.0, min(s, MAX_EXTRAPOLATION))

        # Generate the current interpolated value
        self.interpolated_value = self._generate_interpolated_value(self.buffer[-1], self.p_pred, self.v_curr, self.interval, s)

    def reset(self) -> None:
        """Reset the interpolator to its initial state."""
        self.buffer.clear()
        self.v_curr = 0.0  # ✅ Change from nan to 0.0
        self.p_pred = math.nan
        self.last_sample_time = 0.0
        self.interpolated_value = math.nan
        self._initialized = False

    @staticmethod
    def _calculate_velocity_and_prediction(
        p_prev: float, p_curr: float, v_prev: float, alpha_v: float, interval: float
    ) -> tuple[float, float]:
        """Calculate smoothed velocity and predict the next position.

        Args:
            p_prev: Previous sample value.
            p_curr: Current sample value.
            v_prev: Previous smoothed velocity.
            alpha_v: Smoothing factor for velocity.
            interval: Input sampling interval.

        Returns:
            Tuple containing (smoothed velocity, predicted next position).

        Note:
            NaN values in velocity calculations are treated as zero (stationary).
            On first valid sample pair, uses instantaneous velocity directly.
        """
        # Instantaneous velocity
        v_inst: float = (p_curr - p_prev) / interval

        # Replace NaN with zero
        if math.isnan(v_inst):
            v_inst = 0.0

        # Smoothed velocity - no special case needed!
        v_curr: float = alpha_v * v_prev + (1 - alpha_v) * v_inst  # ✅ Simple!

        # Predicted next position
        p_pred: float = p_curr + v_curr * interval

        return v_curr, p_pred

    @staticmethod
    def _generate_interpolated_value(p_curr: float, p_pred: float, v_curr: float, interval: float, s: float) -> float:
        """Generate the interpolated value for the current normalized time.

        Uses cubic Hermite interpolation between the current sample and predicted
        next position, with tangents derived from the smoothed velocity.

        Args:
            p_curr: Current sample value (at s=0).
            p_pred: Predicted next position (at s=1).
            v_curr: Current smoothed velocity.
            interval: Input sampling interval.
            s: Normalized time (0 <= s <= MAX_EXTRAPOLATION), where 0 = p_curr, 1 = p_pred.

        Returns:
            Interpolated value at the normalized time s.
        """
        m0: float = v_curr * interval  # Tangent at p_curr
        m1: float = v_curr * interval  # Tangent at p_pred (assumes constant velocity)

        # Hermite basis functions
        h00: float = 2 * s**3 - 3 * s**2 + 1
        h10: float = s**3 - 2 * s**2 + s
        h01: float = -2 * s**3 + 3 * s**2
        h11: float = s**3 - s**2

        # Interpolated value using cubic Hermite polynomial
        return h00 * p_curr + h10 * m0 + h01 * p_pred + h11 * m1


class VectorPredictiveHermite:
    """Vectorized numeric value interpolator using cubic Hermite interpolation.

    Interpolates between input samples (NumPy arrays) using cubic Hermite splines,
    with velocity smoothing and one-step prediction for minimal latency.

    Attributes:
        interpolated_value: Current interpolated array. Contains NaN values until first sample arrives.

    Note:
        Small discontinuities may occur when new samples arrive if the actual value
        differs from the prediction. The smoothness depends on input signal characteristics
        and the alpha_v smoothing parameter.

        NaN values in input samples will propagate through interpolation element-wise.
        Elements that are NaN in input will remain NaN in output. For dense data with
        sporadic NaN values, consider pre-filtering NaN values before feeding to the interpolator.
    """

    def __init__(self, input_rate: float, vector_size: int, alpha_v: float = 0.45) -> None:
        """Initialize the vectorized interpolator.

        Args:
            input_rate: Sampling rate of the input data (e.g., 23 Hz).
            vector_size: Size of the input arrays (e.g., number of pose keypoints).
            alpha_v: Smoothing factor for velocity (0.0 to 1.0, default: 0.45).
                Higher values = more smoothing but more lag.
                Lower values = less smoothing but faster response.

       Note:
            Velocity is initialized to zero per element, resulting in a ramp-up effect on
            the first valid sample pair: v_curr = (1 - alpha_v) * v_inst. This provides
            smooth startup behavior and allows independent recovery when elements transition
            from NaN to valid data.

        """
        self.input_rate: float = input_rate
        self.alpha_v: float = alpha_v
        self.interval: float = 1.0 / input_rate
        self.vector_size: int = vector_size

        # State variables
        self.buffer: deque[np.ndarray] = deque(maxlen=2)
        self.v_curr: np.ndarray = np.zeros(vector_size)  # ✅ Change from nan to zeros
        self.p_pred: np.ndarray = np.full(vector_size, np.nan)
        self.last_sample_time = 0.0
        self.interpolated_value: np.ndarray = np.full(vector_size, np.nan)
        self._initialized = False

    def add_sample(self, value: np.ndarray, sample_time: float | None = None) -> None:
        """Add a new sample to the interpolator.

        Args:
            value: New input sample (NumPy array).
            sample_time: Timestamp in seconds when the sample was captured.
                If None, uses time() (not recommended for best accuracy).

        Note:
            For accurate interpolation, always provide the actual capture timestamp,
            not the arrival time. This accounts for processing delays.

        Raises:
            ValueError: If value shape doesn't match vector_size.
        """
        if value.shape[0] != self.vector_size:
            raise ValueError(f"Expected array of size {self.vector_size}, got {value.shape[0]}")

        if sample_time is None:
            sample_time = time()

        self.buffer.append(value)
        self.last_sample_time = sample_time

        # Initialize on first sample
        if not self._initialized:
            self.interpolated_value = value
            self._initialized = True
            return

        # Calculate velocity and prediction
        self.v_curr, self.p_pred = self._calculate_velocity_and_prediction(
            self.buffer[-2], self.buffer[-1], self.v_curr, self.alpha_v, self.interval
        )

    def update(self, current_time: float | None = None) -> None:
        """Update the interpolated value for the current time.

        Updates self.interpolated_value based on elapsed time since the last sample.

        Args:
            current_time: Optional timestamp in seconds. If None, uses time().
        """
        if not self._initialized:
            return  # No data yet

        if len(self.buffer) < 2:
            # Only one sample available, interpolated_value already set to that sample
            return

        if current_time is None:
            current_time = time()

        # Validate time ordering and clamp to prevent negative elapsed time
        if current_time < self.last_sample_time:
            warnings.warn("Interpolator received out-of-order time update.", RuntimeWarning)
            current_time = self.last_sample_time

        elapsed_time: float = current_time - self.last_sample_time
        s: float = elapsed_time / self.interval

        # Allow slight overshoot for timing jitter (up to 10% beyond expected sample time)
        s = max(0.0, min(s, MAX_EXTRAPOLATION))

        # Generate the current interpolated value
        self.interpolated_value = self._generate_interpolated_value(
            self.buffer[-1], self.p_pred, self.v_curr, self.interval, s
        )


        self.ud()
    def ud(self) -> None:
        pass

    def reset(self) -> None:
        """Reset the interpolator to its initial state."""
        self.buffer.clear()
        self.v_curr = np.zeros(self.vector_size)  # ✅ Change from nan to zeros
        self.p_pred = np.full(self.vector_size, np.nan)
        self.last_sample_time = 0.0
        self.interpolated_value = np.full(self.vector_size, np.nan)
        self._initialized = False

    @staticmethod
    def _calculate_velocity_and_prediction(
        p_prev: np.ndarray, p_curr: np.ndarray, v_prev: np.ndarray, alpha_v: float, interval: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate smoothed velocity and predict the next position.

        Args:
            p_prev: Previous sample array.
            p_curr: Current sample array.
            v_prev: Previous smoothed velocity array.
            alpha_v: Smoothing factor for velocity.
            interval: Input sampling interval.

        Returns:
            Tuple containing (smoothed velocity array, predicted next position array).

        Note:
            NaN values in velocity calculations are treated as zero (stationary).
            On first valid sample pair (per element), uses instantaneous velocity directly.
        """
        # Instantaneous velocity
        v_inst: np.ndarray = (p_curr - p_prev) / interval

        # Replace NaN with zero
        v_inst = np.nan_to_num(v_inst, nan=0.0)

        # Smoothed velocity - no special case needed!
        v_curr: np.ndarray = alpha_v * v_prev + (1 - alpha_v) * v_inst  # ✅ Simple!

        # Predicted next position
        p_pred: np.ndarray = p_curr + v_curr * interval

        return v_curr, p_pred

    @staticmethod
    def _generate_interpolated_value(
        p_curr: np.ndarray, p_pred: np.ndarray, v_curr: np.ndarray, interval: float, s: float
    ) -> np.ndarray:
        """Generate the interpolated value for the current normalized time.

        Uses cubic Hermite interpolation between the current sample and predicted
        next position, with tangents derived from the smoothed velocity.

        Args:
            p_curr: Current sample array (at s=0).
            p_pred: Predicted next position array (at s=1).
            v_curr: Current smoothed velocity array.
            interval: Input sampling interval.
            s: Normalized time (0 <= s <= MAX_EXTRAPOLATION), where 0 = p_curr, 1 = p_pred.

        Returns:
            Interpolated array at the normalized time s.
        """
        m0: np.ndarray = v_curr * interval  # Tangent at p_curr
        m1: np.ndarray = v_curr * interval  # Tangent at p_pred (assumes constant velocity)

        # Hermite basis functions (scalars)
        h00: float = 2 * s**3 - 3 * s**2 + 1
        h10: float = s**3 - 2 * s**2 + s
        h01: float = -2 * s**3 + 3 * s**2
        h11: float = s**3 - s**2

        # Interpolated value using cubic Hermite polynomial (vectorized)
        return h00 * p_curr + h10 * m0 + h01 * p_pred + h11 * m1


class ScalarPredictiveAngleHermite(ScalarPredictiveHermite):
    """Angle interpolator using cubic Hermite interpolation with proper angular wrapping.

    Handles angles in radians [-π, π] with proper wrapping around the discontinuity.
    Uses the shortest angular distance for velocity calculations and interpolation.

    Attributes:
        interpolated_value: Current interpolated angle in radians [-π, π]. NaN until first sample.

    Note:
        Velocity is initialized to zero, resulting in a ramp-up effect on the first
        sample pair. This provides smooth startup behavior and occurs when recovering
        from NaN input.
    """

    @staticmethod
    def _calculate_velocity_and_prediction(p_prev: float, p_curr: float, v_prev: float, alpha_v: float, interval: float) -> tuple[float, float]:
        """Calculate smoothed angular velocity and predict the next angle.

        Args:
            p_prev: Previous angle in radians [-π, π].
            p_curr: Current angle in radians [-π, π].
            v_prev: Previous smoothed angular velocity in rad/s.
            alpha_v: Smoothing factor for velocity.
            interval: Input sampling interval in seconds.

        Returns:
            Tuple containing (smoothed angular velocity, predicted next angle in [-π, π]).

        Note:
            Uses shortest angular distance for velocity calculation. NaN values are
            treated as zero velocity (stationary).
        """
        # Calculate angular difference using shortest path
        if math.isnan(p_prev) or math.isnan(p_curr):
            delta: float = math.nan
        else:
            delta = p_curr - p_prev
            # Wrap to [-π, π]
            delta = math.atan2(math.sin(delta), math.cos(delta))

        # Instantaneous angular velocity
        v_inst: float = delta / interval

        # Replace NaN with zero
        if math.isnan(v_inst):
            v_inst = 0.0

        # Smoothed angular velocity
        v_curr: float = alpha_v * v_prev + (1 - alpha_v) * v_inst

        # Predicted next angle
        if math.isnan(p_curr):
            p_pred: float = math.nan
        else:
            p_pred_raw: float = p_curr + v_curr * interval
            # Wrap to [-π, π]
            p_pred = math.atan2(math.sin(p_pred_raw), math.cos(p_pred_raw))

        return v_curr, p_pred

    @staticmethod
    def _generate_interpolated_value(p_curr: float, p_pred: float, v_curr: float, interval: float, s: float) -> float:
        """Generate interpolated angle with proper wrapping.

        Args:
            p_curr: Current angle in radians [-π, π] (at s=0).
            p_pred: Predicted next angle in radians [-π, π] (at s=1).
            v_curr: Current smoothed angular velocity in rad/s.
            interval: Input sampling interval.
            s: Normalized time (0 <= s <= MAX_EXTRAPOLATION).

        Returns:
            Interpolated angle in radians [-π, π] at normalized time s.

        Note:
            Uses angular distance for interpolation to handle wrapping correctly.
        """
        # Handle NaN
        if math.isnan(p_curr) or math.isnan(p_pred):
            return math.nan

        # Calculate shortest angular distance from p_curr to p_pred
        delta: float = p_pred - p_curr
        delta = math.atan2(math.sin(delta), math.cos(delta))

        # Tangents for Hermite interpolation (angular)
        m0: float = v_curr * interval
        m1: float = v_curr * interval

        # Hermite basis functions
        h00: float = 2 * s**3 - 3 * s**2 + 1
        h10: float = s**3 - 2 * s**2 + s
        h01: float = -2 * s**3 + 3 * s**2
        h11: float = s**3 - s**2

        # Hermite interpolation in angular space:
        # We interpolate the angular distance (delta) from p_curr, not absolute positions
        # Standard Hermite: h00*p0 + h10*m0 + h01*p1 + h11*m1
        # Angular version: h00*0 + h10*m0 + h01*delta + h11*m1
        # (h00 term is zero because we start at p_curr and add interpolated delta)
        interpolated_raw: float = p_curr + h10 * m0 + h01 * delta + h11 * m1

        # Wrap result to [-π, π]
        return math.atan2(math.sin(interpolated_raw), math.cos(interpolated_raw))



class VectorPredictiveAngleHermite:
    """Vectorized angle interpolator with velocity steering and dead reckoning.

    Uses dead reckoning with gentle velocity steering toward predicted positions.
    Never resets position on new samples, instead smoothly adjusts velocity to
    steer toward targets, eliminating discontinuities entirely.

    Attributes:
        interpolated_value: Current interpolated angles in radians [-π, π]. Contains NaN
            values until first sample arrives.

    Note:
        Velocity steering provides perfectly smooth motion at the cost of some latency.
        The alpha_steer parameter controls how aggressively the interpolator corrects
        course toward new predictions.
    """

    def __init__(self, input_rate: float, vector_size: int, alpha_v: float = 0.45, alpha_steer: float = 0.3, history_size: int = 4, v_nominal: float = 1.0) -> None:
        """Initialize the vectorized angle interpolator with velocity steering.

        Args:
            input_rate: Sampling rate of the input data (e.g., 23 Hz).
            vector_size: Size of the input arrays (e.g., number of angles).
            alpha_v: Smoothing factor for velocity calculation from samples (0.0 to 1.0, default: 0.45).
                Higher values = more smoothing of measured velocity.
            alpha_steer: Steering strength for velocity correction (0.0 to 1.0, default: 0.3).
                Higher values = faster correction toward target but more responsive to noise.
                Lower values = smoother but more latency.
            history_size: Number of recent samples to use for polynomial prediction (default: 4).
                More samples = better acceleration detection, but more latency and noise sensitivity.
            v_nominal: Nominal velocity threshold for full steering strength (rad/s, default: 1.0).
                Velocities below this get reduced steering, velocities at or above get full steering.
        """
        self.input_rate: float = input_rate
        self.alpha_v: float = alpha_v
        self.alpha_steer: float = alpha_steer
        self.interval: float = 1.0 / input_rate
        self.vector_size: int = vector_size
        self.history_size: int = history_size
        self.v_nominal: float = v_nominal  # ✅ NEW: Nominal velocity threshold

        # State variables
        self.history: deque[np.ndarray] = deque(maxlen=history_size)
        self.history_times: deque[float] = deque(maxlen=history_size)
        self.p_prev: np.ndarray = np.full(vector_size, np.nan)
        self.p_curr: np.ndarray = np.full(vector_size, np.nan)
        self.v_curr: np.ndarray = np.zeros(vector_size)
        self.p_pred: np.ndarray = np.full(vector_size, np.nan)
        self.target_velocity: np.ndarray = np.zeros(vector_size)
        self.velocity_scale: np.ndarray = np.ones(vector_size)  # ✅ NEW: Per-element steering scale
        self.last_sample_time: float = 0.0
        self.last_update_time: float = 0.0
        self.interpolated_value: np.ndarray = np.full(vector_size, np.nan)
        self._initialized = False

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    def add_sample(self, value: np.ndarray, sample_time: float | None = None) -> None:
        """Add a new sample to the interpolator.

        Args:
            value: New input sample (NumPy array of angles in radians).
            sample_time: Timestamp in seconds when the sample was captured.
                If None, uses time() (not recommended for best accuracy).

        Note:
            Does NOT reset interpolated position. Calculates target velocity
            that update() will gradually steer toward.

        Raises:
            ValueError: If value shape doesn't match vector_size.
        """

        if value.shape[0] != self.vector_size:
            raise ValueError(f"Expected array of size {self.vector_size}, got {value.shape[0]}")

        if sample_time is None:
            sample_time = time()
            # print(sample_time)

        # Add to history
        self.history.append(value)
        self.history_times.append(sample_time)

        # Shift samples
        self.p_prev = self.p_curr
        self.p_curr = value
        self.last_sample_time = sample_time

        # Initialize on first sample
        if not self._initialized:
            self.interpolated_value = value.copy()
            self.last_update_time = sample_time
            self._initialized = True
            return

        # Calculate measured velocity from samples (pure, no steering influence)
        v_measured = self._calculate_velocity(self.p_prev, self.p_curr, self.interval)

        # ✅ NEW: Calculate velocity-based steering scale per element
        # Low velocity → low scale (weak steering), high velocity → high scale (full steering)
        v_magnitude = np.abs(v_measured)
        self.velocity_scale = np.clip(v_magnitude / self.v_nominal, 0.0, 1.0)

        # Predict using polynomial fitting if we have enough history
        if len(self.history) >= 3:
            self.p_pred = self._predict_polynomial(self.history, self.history_times, self.interval)
        else:
            # Fallback to linear prediction for first few samples
            self.p_pred = self._predict_next_angle(self.p_curr, v_measured, self.interval)

        # Smooth the measured velocity
        # self.v_curr = self.alpha_v * self.v_curr + (1 - self.alpha_v) * v_measured
        self.v_curr = np.nan_to_num(self.alpha_v * self.v_curr + (1 - self.alpha_v) * v_measured, nan=0.0)

        # Calculate and STORE target velocity (don't apply correction yet)
        self.target_velocity = self._calculate_steering_velocity(
            self.interpolated_value, self.p_pred, self.interval
        )

        # print(self.target_velocity)


    def update(self, current_time: float | None = None) -> None:
        """Update the interpolated value for the current time.

        Uses dead reckoning: integrates current velocity from last update time.
        Gradually steers velocity toward target calculated from last sample.

        Args:
            current_time: Optional timestamp in seconds. If None, uses time().
        """
        if not self._initialized:
            return

        if current_time is None:
            current_time = time()

        # Validate time ordering
        if current_time < self.last_update_time:
            warnings.warn("Interpolator received out-of-order time update.", RuntimeWarning)
            current_time = self.last_update_time

        # Calculate time delta since last update
        dt: float = current_time - self.last_update_time
        self.last_update_time = current_time

        # ✅ Apply adaptive steering: weak when slow, strong when fast
        velocity_correction = self.target_velocity - self.v_curr
        effective_alpha_steer = self.alpha_steer * self.velocity_scale  # Per-element modulation
        self.v_curr = self.v_curr #+ effective_alpha_steer * velocity_correction

        # Dead reckoning: integrate velocity
        delta_position = self.v_curr * dt
        self.interpolated_value = self.interpolated_value + delta_position

        # Wrap to [-π, π]
        self.interpolated_value = np.arctan2(np.sin(self.interpolated_value), np.cos(self.interpolated_value))



        self.alpha_v: float = 0.25
        self.alpha_steer: float = 0.05
        self.v_nominal: float = 3.01

    def reset(self) -> None:
        """Reset the interpolator to its initial state."""
        self.history.clear()
        self.history_times.clear()
        self.p_prev = np.full(self.vector_size, np.nan)
        self.p_curr = np.full(self.vector_size, np.nan)
        self.v_curr = np.zeros(self.vector_size)
        self.p_pred = np.full(self.vector_size, np.nan)
        self.target_velocity = np.zeros(self.vector_size)
        self.velocity_scale = np.ones(self.vector_size)  # ✅ NEW: Reset velocity scale
        self.last_sample_time = 0.0
        self.last_update_time = 0.0
        self.interpolated_value = np.full(self.vector_size, np.nan)
        self._initialized = False

    @staticmethod
    def _calculate_velocity(p_prev: np.ndarray, p_curr: np.ndarray, interval: float) -> np.ndarray:
        """Calculate instantaneous angular velocity from two samples.

        Args:
            p_prev: Previous angles in radians [-π, π].
            p_curr: Current angles in radians [-π, π].
            interval: Time interval between samples in seconds.

        Returns:
            Angular velocities in rad/s.

        Note:
            Uses shortest angular distance per element. NaN values result in zero velocity.
        """
        # Calculate angular differences using shortest path
        delta = p_curr - p_prev
        delta = np.arctan2(np.sin(delta), np.cos(delta))

        # Instantaneous angular velocities
        v_inst = delta / interval

        # Replace NaN with zero
        return np.nan_to_num(v_inst, nan=0.0)

    @staticmethod
    def _predict_next_angle(p_curr: np.ndarray, v_curr: np.ndarray, interval: float) -> np.ndarray:
        """Predict next angle position based on current velocity.

        Args:
            p_curr: Current angles in radians [-π, π].
            v_curr: Current angular velocities in rad/s.
            interval: Prediction time interval in seconds.

        Returns:
            Predicted angles in radians [-π, π].
        """
        p_pred_raw = p_curr + v_curr * interval
        return np.arctan2(np.sin(p_pred_raw), np.cos(p_pred_raw))

    @staticmethod
    def _calculate_steering_velocity(p_current: np.ndarray, p_target: np.ndarray, interval: float) -> np.ndarray:
        """Calculate required velocity to reach target from current position.

        Args:
            p_current: Current interpolated angles in radians [-π, π].
            p_target: Target angles (predicted position) in radians [-π, π].
            interval: Time interval to reach target in seconds.

        Returns:
            Required angular velocities in rad/s to reach target.

        Note:
            Uses shortest angular distance. Returns zero for NaN targets.
        """
        # Calculate angular distance to target (shortest path)
        delta = p_target - p_current
        delta = np.arctan2(np.sin(delta), np.cos(delta))

        # Required velocity to reach target in one interval
        v_required = delta / interval

        # Handle NaN targets
        return np.where(np.isnan(v_required), 0.0, v_required)

    @staticmethod
    def _predict_polynomial(history: deque[np.ndarray], history_times: deque[float], interval: float) -> np.ndarray:
        """Predict next angle using polynomial fitting through recent history.

        Args:
            history: Recent angle samples in radians [-π, π].
            history_times: Timestamps for each history sample.
            interval: Prediction time interval (one sample period ahead).

        Returns:
            Predicted angles in radians [-π, π].

        Note:
            Fits a quadratic polynomial through the last 3-4 samples and extrapolates.
            Handles angular wrapping by unwrapping before fitting, then wrapping result.
        """
        n_samples = len(history)
        if n_samples < 3:
            # Not enough history, return NaN
            return np.full(history[0].shape[0], np.nan)

        # Convert to numpy array: shape (n_samples, vector_size)
        angles = np.array(list(history))

        # Normalize times to [0, 1, 2, ...] for numerical stability
        times = np.array(list(history_times))
        times = times - times[0]  # Start from zero

        # Prediction time (one interval after last sample)
        t_pred = times[-1] + interval

        # Unwrap angles per element to avoid discontinuities during fitting
        # This converts [-π, π] to continuous values
        angles_unwrapped = np.unwrap(angles, axis=0)

        # Fit polynomial per element (vectorized)
        # Use quadratic (degree 2) for balance between smoothness and acceleration capture
        predictions = np.zeros(angles.shape[1])

        for i in range(angles.shape[1]):
            # Handle NaN values: skip elements with any NaN in history
            if np.any(np.isnan(angles_unwrapped[:, i])):
                predictions[i] = np.nan
                continue

            # Fit polynomial: p(t) = a*t^2 + b*t + c
            # Use fewer points if available (minimum 3 for quadratic)
            coeffs = np.polyfit(times, angles_unwrapped[:, i], deg=min(2, n_samples - 1))

            # Extrapolate to prediction time
            predictions[i] = np.polyval(coeffs, t_pred)

        # Wrap predictions back to [-π, π]
        return np.arctan2(np.sin(predictions), np.cos(predictions))