"""Writes the panel's camera settings (exposure, gain, white balance, colour) to the features of a
Vimba camera when they change. The panel is the source of truth: everything is sent once after the
camera opens, over what its user set loaded. A feature the camera model does not have is a warning,
once. A write the camera refuses (a value it does not take, a control channel that timed out) is a
warning too, and is tried again: it does not switch the feature off. A failed read is the same kind
of warning. Both are logged once per reason, not at every retry. Exposure and gain are the panel's
value with auto off and the camera's with auto on; the values in use are read back for the panel.
"Set WB" (white balance) lets the camera measure once and takes the ratios it found back into the panel."""

import logging
from threading import Event, Lock
from time import monotonic
from typing import Any, Callable

from apps.smile.cam.settings import CameraSettings

logger = logging.getLogger(__name__)

RATIO_FEATURES: tuple[str, ...] = ('BalanceRatioAbs', 'BalanceRatio')   # GigE models, USB models
MEASURE_TIMEOUT: float = 5.0    # seconds the camera gets to find its white balance
READ_INTERVAL: float = 0.5      # seconds between reads of the actual exposure and gain
RATE_SMOOTHING: float = 0.2     # share of the newest half second in the frame rate readout


class FeatureWriter:
    def __init__(self, settings: CameraSettings) -> None:
        self.settings: CameraSettings = settings
        self._applied: dict[str, Any] = {}      # values last written to the camera, per setting
        self._missing: set[str] = set()         # features this camera does not have
        self._retry: dict[str, float] = {}      # per setting whose write failed: when it is tried again
        self._problems: dict[str, str] = {}     # per feature: its trouble (a refused write, a failed read) as logged
        self._ratio_feature: str | None = None
        self._measure: Event = Event()
        self._measure_end: float | None = None  # set while the camera measures its white balance
        self._read_next: float = 0.0            # time of the next read of the actual values
        self._frames: int = 0                   # complete frames received since the last read
        self._dropped: int = 0                  # incomplete frames skipped since the start
        self._frames_lock: Lock = Lock()
        self._frames_time: float | None = None  # time of the last read
        self._frame_rate: float | None = None   # smoothed frames per second

        self.settings.bind(CameraSettings.set_WB, self._on_wb_set)

    def close(self) -> None:
        self.settings.unbind(CameraSettings.set_WB, self._on_wb_set)

    def reset(self) -> None:
        """The camera was opened and its user set loaded (at the start, or again after its frames
        stopped): nothing counts as written, everything is sent again. Call on the camera thread,
        or before it starts."""
        self._applied.clear()
        self._missing.clear()
        self._retry.clear()
        self._problems.clear()
        self._ratio_feature = None
        self._measure_end = None

    def count_frame(self, complete: bool = True) -> None:
        """A frame arrived from the camera (any thread): counted for the frame rate readout, or as
        dropped when it is incomplete (network packets lost) and skipped."""
        with self._frames_lock:
            if complete:
                self._frames += 1
            else:
                self._dropped += 1

    def update(self, camera: Any) -> None:
        """Send what changed. Call on the camera thread, with the camera open."""
        s: CameraSettings = self.settings
        self._sync('frame_rate', float(s.target_frame_rate), lambda v: self._set(camera, 'AcquisitionFrameRateAbs', v))
        self._sync('auto_target', int(s.target_brightness), lambda v: all([self._set(camera, 'ExposureAutoTarget', v), self._set(camera, 'GainAutoTarget', v)]))
        self._sync('exposure_max', float(s.exposure_max), lambda v: self._set(camera, 'ExposureAutoMax', v))
        self._sync('gain_max', float(s.gain_max), lambda v: self._set(camera, 'GainAutoMax', v))
        self._auto_or_manual(camera, 'exposure', s.auto_exposure, float(s.manual_exposure), 'ExposureAuto', 'ExposureTimeAbs')
        self._auto_or_manual(camera, 'gain', s.auto_gain, float(s.manual_gain), 'GainAuto', 'Gain')
        self._read_actual(camera)
        self._sync('saturation', float(s.saturation), lambda v: self._set(camera, 'Saturation', v))
        self._sync('hue', float(s.hue), lambda v: self._set(camera, 'Hue', v))
        self._sync('gamma', float(s.gamma), lambda v: self._set(camera, 'Gamma', v))
        self._white_balance(camera)

    def _auto_or_manual(self, camera: Any, key: str, auto: bool, value: float, auto_feature: str, feature: str) -> None:
        self._sync(key + '_auto', auto, lambda v: self._set(camera, auto_feature, 'Continuous' if v else 'Off'))
        if auto:
            # the camera owns the value: the panel's is sent again when auto goes off
            self._applied.pop(key, None)
        else:
            self._sync(key, value, lambda v: self._set(camera, feature, v))

    def _read_actual(self, camera: Any) -> None:
        """The exposure, gain and white balance the camera uses now, for the panel."""
        now: float = monotonic()
        if now < self._read_next:
            return
        self._read_next = now + READ_INTERVAL
        s: CameraSettings = self.settings

        with self._frames_lock:
            frames: int = self._frames
            self._frames = 0
            s.dropped_frames = self._dropped
        if self._frames_time is not None and now > self._frames_time:
            rate: float = frames / (now - self._frames_time)
            self._frame_rate = rate if self._frame_rate is None else self._frame_rate + RATE_SMOOTHING * (rate - self._frame_rate)
            s.frame_rate = round(self._frame_rate, 1)
        self._frames_time = now

        try:
            s.exposure = round(float(camera.get_feature_by_name('ExposureTimeAbs').get()))
            s.gain = round(float(camera.get_feature_by_name('Gain').get()), 1)
        except Exception:
            pass
        red: float | None = self._get_ratio(camera, 'Red')
        blue: float | None = self._get_ratio(camera, 'Blue')
        if red is not None:
            s.WB_red = round(red, 2)
        if blue is not None:
            s.WB_blue = round(blue, 2)

    def _white_balance(self, camera: Any) -> None:
        s: CameraSettings = self.settings
        wb_auto: bool = s.auto_WB
        self._sync('wb_auto', wb_auto, lambda v: self._set(camera, 'BalanceWhiteAuto', 'Continuous' if v else 'Off'))
        if wb_auto:
            # the camera owns the ratios: the panel's are sent again when auto goes off
            self._measure.clear()
            self._measure_end = None
            self._applied.pop('wb_red', None)
            self._applied.pop('wb_blue', None)
            return

        if self._measure.is_set():
            self._measure.clear()
            if self._set(camera, 'BalanceWhiteAuto', 'Once'):
                self._measure_end = monotonic() + MEASURE_TIMEOUT
        if self._measure_end is not None:
            self._measuring(camera)
            return

        self._sync('wb_red', float(s.manual_WB_red), lambda v: self._set_ratio(camera, 'Red', v))
        self._sync('wb_blue', float(s.manual_WB_blue), lambda v: self._set_ratio(camera, 'Blue', v))

    def _measuring(self, camera: Any) -> None:
        """Wait for the camera to finish its measurement, then take its ratios into the panel."""
        try:
            done: bool = str(camera.get_feature_by_name('BalanceWhiteAuto').get()) == 'Off'
            self._problem('BalanceWhiteAuto', None)
        except Exception as e:
            self._problem('BalanceWhiteAuto', f'not readable: {e}')
            done = True
        if not done:
            if self._measure_end is None or monotonic() < self._measure_end:
                return
            logger.warning('Camera white balance did not settle in %.0f s', MEASURE_TIMEOUT)
            self._set(camera, 'BalanceWhiteAuto', 'Off')
        self._measure_end = None

        s: CameraSettings = self.settings
        red: float | None = self._get_ratio(camera, 'Red')
        blue: float | None = self._get_ratio(camera, 'Blue')
        if red is not None:
            s.manual_WB_red = round(red, 2)
            self._applied['wb_red'] = float(s.manual_WB_red)
        if blue is not None:
            s.manual_WB_blue = round(blue, 2)
            self._applied['wb_blue'] = float(s.manual_WB_blue)

    def _sync(self, key: str, value: Any, write: Callable[[Any], Any]) -> None:
        """Write the value when it is not the one the camera has. A write that did not get through is
        not counted as applied: it is tried again, at the pace of the reads."""
        if self._applied.get(key) == value:
            return
        now: float = monotonic()
        if now < self._retry.get(key, 0.0):
            return
        if write(value):
            self._applied[key] = value
            self._retry.pop(key, None)
        else:
            self._applied.pop(key, None)
            self._retry[key] = now + READ_INTERVAL

    def _set(self, camera: Any, name: str, value: Any) -> bool:
        if name in self._missing:
            return False
        try:
            feature = camera.get_feature_by_name(name)
        except Exception as e:
            # the camera model does not have it: not asked again
            self._missing.add(name)
            logger.warning('Camera feature %s not available: %s', name, e)
            return False
        try:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                low, high = feature.get_range()
                value = type(low)(min(max(value, low), high))     # in the feature's range and number type
            feature.set(value)
        except Exception as e:
            # refused this time: the feature stays, the caller tries again
            self._problem(name, f'not set to {value}: {e}')
            return False
        self._problem(name, None)
        return True

    def _problem(self, name: str, problem: str | None) -> None:
        """A feature's trouble, logged when it is new, not at every retry (a camera that is away
        asks every half second); None when the feature worked, so the next trouble is logged again."""
        if problem is None:
            self._problems.pop(name, None)
        elif self._problems.get(name) != problem:
            self._problems[name] = problem
            logger.warning('Camera feature %s %s', name, problem)

    def _ratio(self, camera: Any) -> str | None:
        """The name of the camera's balance ratio feature, or None when it has none."""
        if self._ratio_feature is None and 'BalanceRatio' not in self._missing:
            for name in RATIO_FEATURES:
                try:
                    camera.get_feature_by_name(name)
                except Exception:
                    continue
                self._ratio_feature = name
                break
            else:
                self._missing.add('BalanceRatio')
                logger.warning('Camera feature BalanceRatio not available')
        return self._ratio_feature

    def _set_ratio(self, camera: Any, channel: str, value: float) -> bool:
        name: str | None = self._ratio(camera)
        if name is None:
            return False
        return self._set(camera, 'BalanceRatioSelector', channel) and self._set(camera, name, value)

    def _get_ratio(self, camera: Any, channel: str) -> float | None:
        name: str | None = self._ratio(camera)
        if name is None or not self._set(camera, 'BalanceRatioSelector', channel):
            return None
        try:
            value: float = float(camera.get_feature_by_name(name).get())
        except Exception as e:
            self._problem(name, f'not readable: {e}')
            return None
        self._problem(name, None)
        return value

    def _on_wb_set(self, _: Any = None) -> None:
        self._measure.set()
