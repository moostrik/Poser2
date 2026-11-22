"""What an OAK device reports through depthai, on depthai 2 and 3 alike: the code both OAK backends share. Only the
backends import it; the contract never loads depthai."""

from .capture_clock import capture_time, smoothed_latency
from .calibration import  read_lens_calibration, imu_rotation_to_camera
from .sensors import      log_connected_sensors
from .tracklets import    TRACKING_STATUS
