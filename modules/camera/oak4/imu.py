"""The OAK4's IMU: its turn on the board against the camera."""

# The turn the board mounts the IMU at about the optical axis: a half turn on the OAK4, measured (an upright OAK4
# Pro W read a roll of -178.6 with this at 0); the OAK-D's is a quarter turn. A board constant, not a unit's
# `roll_offset`. Since depthai 3.6.1 `ACCELEROMETER_RAW` is the IMU's own frame, the one
# `getImuToCameraExtrinsics` describes.
IMU_BOARD_ROLL: float = 180.0
