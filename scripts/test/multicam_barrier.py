"""
THERE IS NO OFFICIAL DEPTHAI V3 MULTICAM SUPPORT FOR OAK-D Pro W (according to the luxonis ai)
depthai v3 multi-camera test — mirrors the app's exact camera setup.

4 camera IDs from apps/white_space/data/settings/studio.json.
Sometimes not all cameras are connected. DeviceInfo is resolved in the main
thread before any camera thread starts (same fix applied to camera.py).

Usage:
    python scripts/test_multicam.py [--camb]

Flags:
    --camb    use CAM_B + GRAY8 (production config) instead of CAM_A + BGR888p
"""

import argparse
import logging
import os
import time

# Disable depthai crash dump collection (DEPTHAI_CRASHDUMP=0).
# Without this, closing any device triggers collectAndLogCrashDump →
# archiveFilesCompressed, which crashes in depthai 3.6.1 on Windows.
# The env var is read at call time by isCrashDumpCollectionEnabled(),
# so setting it here (before any dai.Device is created) is sufficient.
os.environ["DEPTHAI_CRASHDUMP"] = "0"
os.environ["DEPTHAI_RECONNECT_TIMEOUT"] = "0"


from threading import Thread, Barrier, BrokenBarrierError, Event, Lock
from typing import Optional
import cv2
import depthai as dai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
_t0 = time.perf_counter()


def _t() -> str:
    """Elapsed seconds since script start."""
    return f"+{time.perf_counter() - _t0:6.2f}s"


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
        log.info("[%s] %s  %s", device_id[-8:], _t(), status)
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

    def open_camera(device_id: str, info: Optional[dai.DeviceInfo], index: int) -> None:
        short = device_id[-8:]
        if info is None:
            log.info("[%s] %s  NOT AVAILABLE", short, _t())
            try:
                barrier.wait(timeout=30.0)
            except BrokenBarrierError:
                pass
            return

        time.sleep(index * 0.5)  # force spread: cam0=0s, cam1=1.5s, cam2=3s, cam3=4.5s
        log.info("[%s] %s  booting...", short, _t())
        t_boot = time.perf_counter()
        try:
            device = dai.Device(info)
        except RuntimeError as e:
            log.info("[%s] %s  boot FAILED: %s", short, _t(), e)
            try:
                barrier.wait(timeout=30.0)
            except BrokenBarrierError:
                pass
            return
        log.info("[%s] %s  booted  (%.2fs)", short, _t(), time.perf_counter() - t_boot)

        pipeline, q = build_pipeline(device, use_camb)
        log.info("[%s] %s  built, waiting at barrier...", short, _t())
        t_barrier = time.perf_counter()

        # Keepalive thread: poll the device every 100 ms while this thread
        # blocks at the barrier.  Without active USB transfers, Windows USB
        # Selective Suspend can drop the idle device's XLink connection.
        keepalive_stop = Event()
        def _keepalive():
            while not keepalive_stop.wait(timeout=0.1):
                try:
                    device.getConnectedCameras()
                except Exception:
                    break
        Thread(target=_keepalive, daemon=True).start()

        try:
            barrier.wait(timeout=30.0)
        except BrokenBarrierError:
            log.info("[%s] %s  barrier timeout", short, _t())
            keepalive_stop.set()
            return
        finally:
            keepalive_stop.set()

        log.info("[%s] %s  barrier passed  (waited %.2fs)", short, _t(), time.perf_counter() - t_barrier)

        # Brief settle delay after the barrier: gives the USB host and device
        # firmware time to stabilise after all concurrent Phase A boots complete.
        time.sleep(1.0)

        log.info("[%s] %s  waiting for start_lock...", short, _t())
        with start_lock:
            log.info("[%s] %s  acquired start_lock", short, _t())
            t_ps = time.perf_counter()
            try:
                pipeline.start()
                log.info("[%s] %s  OPEN and streaming  (start took %.2fs)", short, _t(), time.perf_counter() - t_ps)
                results[device_id] = (device_id, pipeline, q)
            except Exception as e:
                log.info("[%s] %s  pipeline.start() FAILED after %.2fs: %s", short, _t(), time.perf_counter() - t_ps, e)

    threads = [
        Thread(target=open_camera, args=(did, info, i), daemon=True)
        for i, (did, info) in enumerate(device_infos.items())
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    setups = [r for r in results.values() if r is not None]
    if not setups:
        log.info("%s  No cameras streaming.", _t())
        return

    display_loop(setups, use_camb)

    log.info("%s  Stopping all pipelines...", _t())
    stop_threads = [
        Thread(target=lambda did=device_id, p=pipeline: (p.stop(), log.info("[%s] %s  closed", did[-8:], _t())), daemon=True)
        for device_id, pipeline, _ in setups
    ]
    for t in stop_threads:
        t.start()
    for t in stop_threads:
        t.join(timeout=10.0)
    log.info("%s  Done.", _t())

    # Wait for all stopped cameras to re-enumerate on the USB bus.
    stopped_ids = {device_id for device_id, _, _ in setups}
    max_tries = 6
    for attempt in range(1, max_tries + 1):
        available_ids = {info.getDeviceId() for info in dai.Device.getAllAvailableDevices()}
        missing = stopped_ids - available_ids
        if not missing:
            log.info("%s  All cameras back on USB bus — safe to restart (attempt %d).", _t(), attempt)
            break
        log.info("%s  Cameras still resetting on USB bus (attempt %d/%d): %s", _t(), attempt, max_tries, missing)
        time.sleep(0.5)
    else:
        log.info("%s  USB reset timed out — cameras may not be available on next run: %s", _t(), missing)


# ---------------------------------------------------------------------------
# Display loop
# ---------------------------------------------------------------------------

def display_loop(setups: list[tuple], use_camb: bool) -> None:
    log.info("%s  Displaying %d camera(s). Press 'q' to quit.", _t(), len(setups))
    frame_counts = [0] * len(setups)
    dead = [False] * len(setups)
    t_fps = time.time()

    while True:
        for i, (mxid, pipeline, q) in enumerate(setups):
            if dead[i]:
                continue
            try:
                if not pipeline.isRunning():
                    log.info("[%s] %s  pipeline stopped", mxid[:8], _t())
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
                    log.info("[%s] %s  ERROR: %s", mxid[:8], _t(), e)
                    dead[i] = True

        now = time.time()
        if now - t_fps >= 2.0:
            elapsed = now - t_fps
            for i, (mxid, _, _) in enumerate(setups):
                status = "DEAD" if dead[i] else f"{frame_counts[i] / elapsed:.1f} fps"
                log.info("cam%d [%s]: %s", i, mxid[:8], status)
            frame_counts = [0] * len(setups)
            t_fps = now

        if all(dead):
            log.info("%s  All cameras dead.", _t())
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
    log.info("%s  Available cameras: %d", _t(), n_available)
    log.info("%s  Configured camera IDs (%d):", _t(), len(CAMERA_IDS))
    device_infos = lookup_devices(CAMERA_IDS)

    cam_type = "CAM_B + GRAY8" if args.camb else "CAM_A + BGR888p"
    log.info("%s  Running with %s", _t(), cam_type)

    run(device_infos, args.camb)


if __name__ == "__main__":
    main()
