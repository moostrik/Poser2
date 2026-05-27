"""
depthai v3 multi-camera test — mirrors the app's exact camera setup.

4 camera IDs from apps/white_space/data/settings/studio.json.
Two are not connected; two are real. DeviceInfo is resolved in the main
thread before any camera thread starts (same fix applied to camera.py).

Usage:
    python scripts/test_multicam.py [--camb]

Flags:
    --camb    use CAM_B + GRAY8 (production config) instead of CAM_A + BGR888p
"""

import argparse
import os
import time

# Disable depthai crash dump collection (DEPTHAI_CRASHDUMP=0).
# Without this, closing any device triggers collectAndLogCrashDump →
# archiveFilesCompressed, which crashes in depthai 3.6.1 on Windows.
# The env var is read at call time by isCrashDumpCollectionEnabled(),
# so setting it here (before any dai.Device is created) is sufficient.
os.environ["DEPTHAI_CRASHDUMP"] = "0"
from threading import Thread, Barrier, BrokenBarrierError, Lock
from typing import Optional
import cv2
import depthai as dai


# ---------------------------------------------------------------------------
# Camera IDs — from apps/white_space/data/settings/studio.json
# ---------------------------------------------------------------------------

CAMERA_IDS = [
    '14442C101136D1D200',
    '14442C10F124D9D600',
    '14442C10110AD3D200',
    '14442C1031DDD2D200',
]

W, H = 640, 400
FPS_TARGET = 30
QUEUE_MAX = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def lookup_devices(ids: list[str]) -> dict[str, Optional[dai.DeviceInfo]]:
    """Resolve DeviceInfo for all IDs once in the main thread."""
    available = {info.getDeviceId(): info for info in dai.Device.getAllAvailableDevices()}
    result = {}
    for device_id in ids:
        info = available.get(device_id)
        result[device_id] = info
        status = "FOUND" if info else "NOT AVAILABLE"
        print(f"  {device_id}: {status}")
    return result


def build_pipeline(device: dai.Device, use_camb: bool) -> tuple[dai.Pipeline, dai.MessageQueue]:
    pipeline = dai.Pipeline(device)
    cam = pipeline.create(dai.node.Camera)
    if use_camb:
        cam.build(dai.CameraBoardSocket.CAM_B)
        out = cam.requestOutput((W, H), type=dai.ImgFrame.Type.GRAY8, fps=FPS_TARGET)
    else:
        cam.build(dai.CameraBoardSocket.CAM_A)
        out = cam.requestOutput((W, H), type=dai.ImgFrame.Type.BGR888p, fps=FPS_TARGET)
    q = out.createOutputQueue(maxSize=QUEUE_MAX, blocking=False)
    return pipeline, q


# ---------------------------------------------------------------------------
# Strategy 3: Boot-all-then-start-all (mirrors camera.py)
#   DeviceInfo resolved in main thread before threads start.
#   Phase A (concurrent, no lock): boot + build + queues.
#   Barrier: wait for ALL cameras (including missing ones) before starting.
#   Phase B (sequential under lock): pipeline.start().
# ---------------------------------------------------------------------------

def run(device_infos: dict[str, Optional[dai.DeviceInfo]], use_camb: bool) -> None:
    n = len(device_infos)
    barrier = Barrier(n)
    start_lock = Lock()
    results: dict[str, Optional[tuple]] = {d: None for d in device_infos}

    def open_camera(device_id: str, info: Optional[dai.DeviceInfo]) -> None:
        if info is None:
            print(f"  {device_id}: NOT AVAILABLE")
            try:
                barrier.wait(timeout=30.0)
            except BrokenBarrierError:
                pass
            return

        print(f"  {device_id}: booting...")
        try:
            device = dai.Device(info)
        except RuntimeError as e:
            print(f"  {device_id}: boot FAILED: {e}")
            try:
                barrier.wait(timeout=30.0)
            except BrokenBarrierError:
                pass
            return

        pipeline, q = build_pipeline(device, use_camb)
        print(f"  {device_id}: built, waiting at barrier...")
        try:
            barrier.wait(timeout=30.0)
        except BrokenBarrierError:
            print(f"  {device_id}: barrier timeout")
            return

        # Brief settle delay after the barrier: gives the USB host and device
        # firmware time to stabilise after all concurrent Phase A boots complete.
        time.sleep(0.5)

        with start_lock:
            try:
                pipeline.start()
                print(f"  {device_id}: OPEN and streaming")
                results[device_id] = (device_id, pipeline, q)
            except Exception as e:
                print(f"  {device_id}: pipeline.start() FAILED: {e}")

    threads = [
        Thread(target=open_camera, args=(did, info), daemon=True)
        for did, info in device_infos.items()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    setups = [r for r in results.values() if r is not None]
    if not setups:
        print("  No cameras streaming.")
        return

    display_loop(setups, use_camb)

    print("Stopping all pipelines...")
    stop_threads = [
        Thread(target=lambda did=device_id, p=pipeline: (p.stop(), print(f"  [{did}]: closed")), daemon=True)
        for device_id, pipeline, _ in setups
    ]
    for t in stop_threads:
        t.start()
    for t in stop_threads:
        t.join(timeout=10.0)
    print("Done.")

    # Wait for all stopped cameras to re-enumerate on the USB bus.
    stopped_ids = {device_id for device_id, _, _ in setups}
    max_tries = 6
    for attempt in range(1, max_tries + 1):
        available_ids = {info.getDeviceId() for info in dai.Device.getAllAvailableDevices()}
        missing = stopped_ids - available_ids
        if not missing:
            print(f"All cameras back on USB bus — safe to restart (attempt {attempt}).")
            break
        print(f"  Cameras still resetting on USB bus (attempt {attempt}/{max_tries}): {missing}")
        time.sleep(0.5)
    else:
        print(f"  USB reset timed out — cameras may not be available on next run: {missing}")


# ---------------------------------------------------------------------------
# Display loop
# ---------------------------------------------------------------------------

def display_loop(setups: list[tuple], use_camb: bool) -> None:
    print(f"\nDisplaying {len(setups)} camera(s). Press 'q' to quit.\n")
    frame_counts = [0] * len(setups)
    dead = [False] * len(setups)
    t_fps = time.time()

    while True:
        for i, (mxid, pipeline, q) in enumerate(setups):
            if dead[i]:
                continue
            try:
                if not pipeline.isRunning():
                    print(f"  [{mxid[:8]}] pipeline stopped")
                    dead[i] = True
                    continue
                msg = q.tryGet()
                if msg is not None:
                    frame = msg.getCvFrame()
                    if use_camb:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    cv2.imshow(f"cam{i} [{mxid[:8]}]", frame)
                    frame_counts[i] += 1
            except Exception as e:
                if not dead[i]:
                    print(f"  [{mxid[:8]}] ERROR: {e}")
                    dead[i] = True

        now = time.time()
        if now - t_fps >= 2.0:
            elapsed = now - t_fps
            for i, (mxid, _, _) in enumerate(setups):
                status = "DEAD" if dead[i] else f"{frame_counts[i] / elapsed:.1f} fps"
                print(f"  cam{i} [{mxid[:8]}]: {status}")
            frame_counts = [0] * len(setups)
            t_fps = now

        if all(dead):
            print("  All cameras dead.")
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camb", action="store_true",
                        help="Use CAM_B + GRAY8 instead of CAM_A + BGR888p")
    args = parser.parse_args()

    n_available = len(dai.Device.getAllAvailableDevices())
    print(f"Available cameras: {n_available}")
    print(f"\nConfigured camera IDs ({len(CAMERA_IDS)}):")
    device_infos = lookup_devices(CAMERA_IDS)

    cam_type = "CAM_B + GRAY8" if args.camb else "CAM_A + BGR888p"
    print(f"\nRunning with {cam_type}\n")

    run(device_infos, args.camb)


if __name__ == "__main__":
    main()
