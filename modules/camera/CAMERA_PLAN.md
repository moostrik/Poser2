# The camera's plan

The camera package held up to the rules (2026-11-14, the apps' plan F; again 2026-11-21). Who uses what, from the
code: the contract (`CameraSource`, `CameraFrame`, `CameraTracklet(s)`, the settings, `CameraHealth`, the capture
match) is read by every app and by inference, recording, tracker and render. The backends are each one kind of
camera, used now by the apps that have it: `oak` by white_space and hd_trio, `oak4` by deep_flow, low_flow and
smile, `vimba` and `webcam` by smile. That is general code with one user, which stays.

The steps below are what is left of the scan of 2026-11-21. **(seen)** is read in the code,
**(deduction)** inferred.

## F. With an OAK-D at the PC

- F1. The live check of the OAK-D's capture stamp, which is built (`capture_clock.py`, shared with the OAK4): an
  OAK-D, white_space live, its frames' and tracklets' latency readings against oak4's. Waits for an OAK-D at the PC: none
  was attached on 2026-11-22 **(seen: depthai 2 found no device)**. RECORDING_PLAN.md B's OAK-D capture time is this
  step.

## Open

- The OAK4 is to run standalone, as an OAK App **(site fact)**. Depth alignment then needs a home: `ImageAlign` is
  the only alignment on the camera and fattens near objects ([Depth](oak4/OAK4.md#depth)), and its firmware cannot be
  changed. The options are to keep sending raw depth with its `DepthAlignment` to a PC that runs `DepthAlign`, or to
  run the same reprojection as a host node on the camera's CPU, whose speed for 144,000 points a frame has not been
  measured. A Script node is too slow for it **(deduction)**.
- Flow Motion's A/B of the depth's confidence needs a recording with it and a player moving. No recording with a
  confidence is kept: the two probes of 2026-11-21 were made to measure the encoder and are deleted.

## Rules it touches

- Stamp a frame once, with its moment of capture where the device reports one: F1
