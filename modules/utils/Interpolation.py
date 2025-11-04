import math
from collections import deque
from time import time
import numpy as np
import warnings

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


class VectorPredictiveAngleHermite(VectorPredictiveHermite):
    """Vectorized angle interpolator with proper angular wrapping.

    Handles arrays of angles in radians [-π, π] with proper wrapping around discontinuities.
    Uses the shortest angular distance for velocity calculations and interpolation.

    Attributes:
        interpolated_value: Current interpolated angles in radians [-π, π]. Contains NaN
            values until first sample arrives.

    Note:
        Velocity is initialized to zero per element, resulting in a ramp-up effect on
        the first valid sample pair. This allows independent recovery when elements
        transition from NaN to valid data.
    """

    @staticmethod
    def _calculate_velocity_and_prediction(p_prev: np.ndarray, p_curr: np.ndarray, v_prev: np.ndarray, alpha_v: float, interval: float) -> tuple[np.ndarray, np.ndarray]:
        """Calculate smoothed angular velocities and predict next angles.

        Args:
            p_prev: Previous angles in radians [-π, π].
            p_curr: Current angles in radians [-π, π].
            v_prev: Previous smoothed angular velocities in rad/s.
            alpha_v: Smoothing factor for velocity.
            interval: Input sampling interval in seconds.

        Returns:
            Tuple containing (smoothed angular velocities, predicted next angles in [-π, π]).

        Note:
            Uses shortest angular distance per element. NaN values are treated as zero
            velocity (stationary) element-wise.
        """
        # Calculate angular differences using shortest path
        delta = p_curr - p_prev
        # Wrap to [-π, π]
        delta: np.ndarray = np.arctan2(np.sin(delta), np.cos(delta))

        # Instantaneous angular velocities
        v_inst: np.ndarray = delta / interval

        # Replace NaN with zero
        v_inst = np.nan_to_num(v_inst, nan=0.0)

        # Smoothed angular velocities
        v_curr: np.ndarray = alpha_v * v_prev + (1 - alpha_v) * v_inst

        # Predicted next angles
        p_pred_raw = p_curr + v_curr * interval
        # Wrap to [-π, π]
        p_pred: np.ndarray = np.arctan2(np.sin(p_pred_raw), np.cos(p_pred_raw))

        return v_curr, p_pred

    @staticmethod
    def _generate_interpolated_value(p_curr: np.ndarray, p_pred: np.ndarray, v_curr: np.ndarray, interval: float, s: float) -> np.ndarray:
        """Generate interpolated angles with proper wrapping.

        Args:
            p_curr: Current angles in radians [-π, π] (at s=0).
            p_pred: Predicted next angles in radians [-π, π] (at s=1).
            v_curr: Current smoothed angular velocities in rad/s.
            interval: Input sampling interval.
            s: Normalized time (0 <= s <= MAX_EXTRAPOLATION).

        Returns:
            Interpolated angles in radians [-π, π] at normalized time s.

        Note:
            Uses angular distance per element for interpolation to handle wrapping correctly.
        """
        # Calculate shortest angular distance from p_curr to p_pred
        delta = p_pred - p_curr
        delta: np.ndarray = np.arctan2(np.sin(delta), np.cos(delta))

        # Tangents for Hermite interpolation (angular)
        m0: np.ndarray = v_curr * interval
        m1: np.ndarray = v_curr * interval

        # Hermite basis functions (scalars)
        h00: float = 2 * s**3 - 3 * s**2 + 1
        h10: float = s**3 - 2 * s**2 + s
        h01: float = -2 * s**3 + 3 * s**2
        h11: float = s**3 - s**2

        # Hermite interpolation in angular space:
        # We interpolate the angular distance (delta) from p_curr, not absolute positions
        # Standard Hermite: h00*p0 + h10*m0 + h01*p1 + h11*m1
        # Angular version: h00*0 + h10*m0 + h01*delta + h11*m1
        # (h00 term is zero because we start at p_curr and add interpolated delta)
        interpolated_raw = p_curr + h10 * m0 + h01 * delta + h11 * m1

        # Wrap result to [-π, π]
        return np.arctan2(np.sin(interpolated_raw), np.cos(interpolated_raw))