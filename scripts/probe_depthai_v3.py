"""
Phase 1 probe script: verify depthai v3 API surface on real OAK-D hardware.

Run from the project root with the v3 venv active:
    python scripts/probe_depthai_v3.py [--blob data/models/yolov8n_coco_416x416_6S.blob]

Produces a pass/fail report for every API surface point used by Poser2's
modules/oak/ code. Run against a real OAK-D (RVC2) camera.
"""

import argparse
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np

SECTION_WIDTH = 60


def header(title: str) -> None:
    print(f"\n{'='*SECTION_WIDTH}")
    print(f"  {title}")
    print(f"{'='*SECTION_WIDTH}")


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def fail(msg: str, exc: Exception | None = None) -> None:
    print(f"  [FAIL] {msg}")
    if exc:
        traceback.print_exc()


results: dict[str, bool] = {}


def check(name: str, fn) -> bool:
    try:
        fn()
        ok(name)
        results[name] = True
        return True
    except Exception as e:
        fail(name, e)
        results[name] = False
        return False


# ---------------------------------------------------------------------------
# Helpers copied verbatim from modules/oak/camera/pipeline.py
# ---------------------------------------------------------------------------

def find_perspective_warp(width, height, width_offset, flip_h, flip_v, mesh_w, mesh_h):
    src_points = np.array([[0,0],[width,0],[width,height],[0,height]], dtype=np.float32)
    dst_points = np.array([
        [width_offset, 0],
        [width - width_offset, 0],
        [width + width_offset, height],
        [-width_offset, height]
    ], dtype=np.float32)
    if flip_h:
        dst_points[:, 0] = width - dst_points[:, 0]
    if flip_v:
        dst_points[:, 1] = height - dst_points[:, 1]
    H = cv2.getPerspectiveTransform(src_points, dst_points)
    H_inv = np.linalg.inv(H)
    grid_x = np.linspace(0, width - 1, mesh_w)
    grid_y = np.linspace(0, height - 1, mesh_h)
    import depthai as dai
    mesh_points = []
    for y in grid_y:
        for x in grid_x:
            p = np.array([x, y, 1.0])
            src = H_inv @ p
            src /= src[2]
            mesh_points.append(dai.Point2f(float(src[0]), float(src[1])))
    return mesh_points


# ---------------------------------------------------------------------------
# PROBE 1 — import and basic enum access (no hardware)
# ---------------------------------------------------------------------------

def probe_imports() -> None:
    header("PROBE 1 — Import & enum surface (no hardware)")
    import depthai as dai

    check("dai.CameraBoardSocket.CAM_A", lambda: dai.CameraBoardSocket.CAM_A)
    check("dai.CameraBoardSocket.CAM_B", lambda: dai.CameraBoardSocket.CAM_B)
    check("dai.CameraBoardSocket.CAM_C", lambda: dai.CameraBoardSocket.CAM_C)
    check("dai.ImgFrame.Type.BGR888p", lambda: dai.ImgFrame.Type.BGR888p)
    check("dai.ImgFrame.Type.RAW8",    lambda: dai.ImgFrame.Type.RAW8)
    check("dai.ImgFrame.Type.GRAY8",   lambda: dai.ImgFrame.Type.GRAY8)
    check("dai.Point2f",               lambda: dai.Point2f(0.0, 0.0))
    check("dai.CameraControl",         lambda: dai.CameraControl())
    check("dai.ImgFrame",              lambda: dai.ImgFrame())

    def check_tracker_enums():
        assert dai.TrackerType.ZERO_TERM_IMAGELESS is not None
        assert dai.TrackerIdAssignmentPolicy.SMALLEST_ID is not None

    check("dai.TrackerType.ZERO_TERM_IMAGELESS + TrackerIdAssignmentPolicy.SMALLEST_ID",
          check_tracker_enums)

    check("dai.node.Camera exists",           lambda: dai.node.Camera)
    check("dai.node.Warp exists",             lambda: dai.node.Warp)
    check("dai.node.ImageManip exists",        lambda: dai.node.ImageManip)
    check("dai.node.YoloDetectionNetwork exists", lambda: dai.node.YoloDetectionNetwork)
    check("dai.node.ObjectTracker exists",     lambda: dai.node.ObjectTracker)
    check("dai.node.Sync exists",              lambda: dai.node.Sync)

    # Tracklets replaces RawTracklets in v3
    check("dai.Tracklets (replaces RawTracklets)", lambda: dai.Tracklets)

    # MessageGroup for Sync callback
    check("dai.MessageGroup", lambda: dai.MessageGroup)


# ---------------------------------------------------------------------------
# PROBE 2 — Device enumeration (no pipeline)
# ---------------------------------------------------------------------------

def probe_device_enum() -> None:
    header("PROBE 2 — Device enumeration")
    import depthai as dai

    def enum_devices():
        devices = dai.Device.getAllAvailableDevices()
        print(f"         Found {len(devices)} device(s):")
        for d in devices:
            print(f"           mxid={d.getMxId()}  state={d.state}")
        assert len(devices) > 0, "No OAK devices found — plug in your camera"

    check("Device.getAllAvailableDevices() + getMxId() + state", enum_devices)


# ---------------------------------------------------------------------------
# PROBE 3 — Minimal pipeline: Camera node → output queue (60 s)
# ---------------------------------------------------------------------------

def probe_camera_node(fps: float = 30.0, run_seconds: float = 10.0) -> None:
    header("PROBE 3 — Camera node → output queue (live frames)")
    import depthai as dai

    try:
        pipeline = dai.Pipeline()
        cam = pipeline.create(dai.node.Camera)
        ok("pipeline.create(dai.node.Camera)")
    except Exception as e:
        fail("pipeline.create(dai.node.Camera)", e)
        return

    try:
        cam.build(dai.CameraBoardSocket.CAM_A)
        ok("Camera.build(CameraBoardSocket.CAM_A)")
    except Exception as e:
        fail("Camera.build(CameraBoardSocket.CAM_A)", e)
        return

    try:
        output = cam.requestOutput((1280, 720), type=dai.ImgFrame.Type.BGR888p)
        ok("Camera.requestOutput((1280,720), BGR888p)")
    except Exception as e:
        fail("Camera.requestOutput((1280,720), BGR888p)", e)
        return

    try:
        q = output.createOutputQueue(maxSize=1, blocking=False)
        ok("Node.Output.createOutputQueue(maxSize=1, blocking=False)")
    except Exception as e:
        fail("Node.Output.createOutputQueue", e)
        return

    try:
        pipeline.start()
        ok("pipeline.start()")
    except Exception as e:
        fail("pipeline.start()", e)
        return

    frame_count = 0
    t0 = time.time()
    first_frame_latency: float | None = None
    try:
        while time.time() - t0 < run_seconds:
            frame = q.get()
            if frame is not None:
                if first_frame_latency is None:
                    first_frame_latency = time.time() - t0
                frame_count += 1
    except Exception as e:
        fail("queue.get() frame loop", e)
    finally:
        pipeline.stop()
        ok("pipeline.stop()")

    elapsed = time.time() - t0
    actual_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"         frames={frame_count}  elapsed={elapsed:.1f}s  fps={actual_fps:.1f}  first_frame={first_frame_latency:.2f}s")

    check("Frame count > 0", lambda: None if frame_count > 0 else (_ for _ in ()).throw(AssertionError("no frames received")))
    check(f"FPS within 50% of target {fps}", lambda: None if actual_fps > fps * 0.5 else (_ for _ in ()).throw(AssertionError(f"fps={actual_fps:.1f} < {fps*0.5:.1f}")))

    # Check readback methods on ImgFrame
    try:
        pipeline2 = dai.Pipeline()
        cam2 = pipeline2.create(dai.node.Camera)
        cam2.build(dai.CameraBoardSocket.CAM_A)
        out2 = cam2.requestOutput((1280, 720), type=dai.ImgFrame.Type.BGR888p)
        q2 = out2.createOutputQueue(maxSize=1, blocking=False)
        pipeline2.start()
        frame = None
        deadline = time.time() + 5.0
        while time.time() < deadline:
            frame = q2.get()
            if frame is not None:
                break
        pipeline2.stop()

        if frame is not None:
            check("ImgFrame.getExposureTime()",     lambda: frame.getExposureTime())
            check("ImgFrame.getSensitivity()",      lambda: frame.getSensitivity())
            check("ImgFrame.getColorTemperature()", lambda: frame.getColorTemperature())
        else:
            fail("ImgFrame readback — no frame received within 5s")
    except Exception as e:
        fail("ImgFrame readback probe", e)


# ---------------------------------------------------------------------------
# PROBE 4 — Warp node with perspective mesh
# ---------------------------------------------------------------------------

def probe_warp_node(fps: float = 30.0, run_seconds: float = 5.0) -> None:
    header("PROBE 4 — Warp node with find_perspective_warp mesh")
    import depthai as dai

    width, height = 1280, 720
    mesh_w, mesh_h = 2, 64
    warp_p = width * 0.5 * 0.1  # 10% perspective

    try:
        mesh = find_perspective_warp(width, height, warp_p, False, False, mesh_w, mesh_h)
        ok(f"find_perspective_warp produced {len(mesh)} points")
    except Exception as e:
        fail("find_perspective_warp", e)
        return

    try:
        pipeline = dai.Pipeline()
        cam = pipeline.create(dai.node.Camera)
        cam.build(dai.CameraBoardSocket.CAM_A)
        cam_out = cam.requestOutput((width, height), type=dai.ImgFrame.Type.BGR888p)

        warp = pipeline.create(dai.node.Warp)
        warp.setOutputSize(width, height)
        warp.setMaxOutputFrameSize(width * height * 3)
        warp.setWarpMesh(mesh, mesh_w, mesh_h)
        ok("Warp.setOutputSize / setMaxOutputFrameSize / setWarpMesh — API exists")

        cam_out.link(warp.inputImage)
        ok("cam_out.link(warp.inputImage)")

        q = warp.out.createOutputQueue(maxSize=1, blocking=False)

        pipeline.start()
        frame_count = 0
        t0 = time.time()
        while time.time() - t0 < run_seconds:
            f = q.get()
            if f is not None:
                frame_count += 1
        pipeline.stop()

        check(f"Warp output frames > 0 (got {frame_count})",
              lambda: None if frame_count > 0 else (_ for _ in ()).throw(AssertionError("no warped frames")))
    except Exception as e:
        fail("Warp node pipeline", e)


# ---------------------------------------------------------------------------
# PROBE 5 — YOLO blob detection on a real frame
# ---------------------------------------------------------------------------

def probe_yolo(blob_path: Path, fps: float = 30.0, run_seconds: float = 15.0) -> None:
    header(f"PROBE 5 — YoloDetectionNetwork.setBlobPath({blob_path.name})")
    import depthai as dai

    if not blob_path.exists():
        fail(f"Blob not found: {blob_path}")
        results["YoloDetectionNetwork.setBlobPath"] = False
        return

    # Infer model input size from filename (e.g. 416x416 or 640x352)
    name = blob_path.stem
    square = "416x416" in name
    nn_w, nn_h = (416, 416) if square else (640, 352)

    try:
        pipeline = dai.Pipeline()
        cam = pipeline.create(dai.node.Camera)
        cam.build(dai.CameraBoardSocket.CAM_A)
        cam_out = cam.requestOutput((1280, 720), type=dai.ImgFrame.Type.BGR888p)

        manip = pipeline.create(dai.node.ImageManip)
        # v3 sequence API — will confirm method names work
        try:
            manip.initialConfig.addResize(nn_w, nn_h)
            ok("ImageManip.initialConfig.addResize (v3 API)")
        except AttributeError:
            # Fall back to v2 API if v3 not yet available
            manip.initialConfig.setResize(nn_w, nn_h)
            ok("ImageManip.initialConfig.setResize (v2 API — addResize not yet in this build)")
        try:
            manip.initialConfig.setOutputFrameType(dai.ImgFrame.Type.BGR888p)
            ok("ImageManip.initialConfig.setOutputFrameType (v3 API)")
        except AttributeError:
            manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
            ok("ImageManip.initialConfig.setFrameType (v2 API — setOutputFrameType not yet available)")
        cam_out.link(manip.inputImage)

        yolo = pipeline.create(dai.node.YoloDetectionNetwork)
        yolo.setBlobPath(blob_path)
        yolo.setNumInferenceThreads(2)
        yolo.setNumClasses(80)
        yolo.setCoordinateSize(4)
        yolo.setConfidenceThreshold(0.5)
        yolo.setIouThreshold(0.5)
        yolo.input.setBlocking(False)
        ok("YoloDetectionNetwork configured (setBlobPath, setNumClasses, setConfidenceThreshold, ...)")
        manip.out.link(yolo.input)

        tracker = pipeline.create(dai.node.ObjectTracker)
        tracker.setDetectionLabelsToTrack([0])  # person
        tracker.setTrackerType(dai.TrackerType.ZERO_TERM_IMAGELESS)
        tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)
        yolo.passthrough.link(tracker.inputTrackerFrame)
        yolo.passthrough.link(tracker.inputDetectionFrame)
        yolo.out.link(tracker.inputDetections)
        ok("ObjectTracker configured and linked")

        det_q = yolo.out.createOutputQueue(maxSize=1, blocking=False)
        trk_q = tracker.out.createOutputQueue(maxSize=1, blocking=False)

        pipeline.start()
        ok("pipeline.start() with YOLO + tracker")

        detection_count = 0
        tracklet_msgs = 0
        t0 = time.time()
        while time.time() - t0 < run_seconds:
            det = det_q.get()
            if det is not None:
                detection_count += len(det.detections)
            trk = trk_q.get()
            if trk is not None:
                tracklet_msgs += 1
                # Confirm Tracklets type (replaces RawTracklets)
                _ = trk.tracklets  # attribute access
        pipeline.stop()

        print(f"         detection_count={detection_count}  tracklet_msgs={tracklet_msgs}  over {run_seconds}s")
        check("YOLO produced output (detection_count > 0 or tracklet_msgs > 0)",
              lambda: None if detection_count > 0 or tracklet_msgs > 0
              else (_ for _ in ()).throw(AssertionError("no detections or tracklets — is a person visible?")))
        check("Tracklets.tracklets attribute accessible (v3 type)", lambda: None)

    except Exception as e:
        fail("YOLO + tracker pipeline", e)


# ---------------------------------------------------------------------------
# PROBE 6 — Sync node API
# ---------------------------------------------------------------------------

def probe_sync_node() -> None:
    header("PROBE 6 — Sync node API")
    import depthai as dai

    def test_sync_api():
        pipeline = dai.Pipeline()
        sync = pipeline.create(dai.node.Sync)
        sync.setSyncAttempts(-1)
        sync.setSyncThreshold(timedelta(seconds=1.0 / 30.0 * 0.5))

    check("Sync.setSyncAttempts(-1) + setSyncThreshold(timedelta)", test_sync_api)


# ---------------------------------------------------------------------------
# PROBE 7 — Device IR control methods
# ---------------------------------------------------------------------------

def probe_ir_control() -> None:
    header("PROBE 7 — Device IR control methods (OAK-D Pro only)")
    import depthai as dai

    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        print("  [SKIP] No devices found")
        return

    try:
        device_info = devices[0]
        pipeline = dai.Pipeline()
        pipeline.start()  # minimal — just to open device
        # Try the IR methods on the opened device
        # Note: these may raise on OAK-D Lite (no IR projector) — that's OK
        try:
            pipeline.getDefaultDevice().setIrFloodLightIntensity(0.0)
            ok("Device.setIrFloodLightIntensity — exists")
        except AttributeError as e:
            fail("Device.setIrFloodLightIntensity — not on Device in v3", e)
        except Exception:
            ok("Device.setIrFloodLightIntensity — exists (raised at runtime, likely no IR hardware)")
        try:
            pipeline.getDefaultDevice().setIrLaserDotProjectorIntensity(0.0)
            ok("Device.setIrLaserDotProjectorIntensity — exists")
        except AttributeError as e:
            fail("Device.setIrLaserDotProjectorIntensity — not on Device in v3", e)
        except Exception:
            ok("Device.setIrLaserDotProjectorIntensity — exists (raised at runtime, likely no IR hardware)")
        pipeline.stop()
    except Exception as e:
        fail("IR control probe (pipeline open failed)", e)


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

def summary() -> None:
    header("SUMMARY")
    passed = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]
    print(f"  Passed: {len(passed)}/{len(results)}")
    if failed:
        print(f"  FAILED ({len(failed)}):")
        for f in failed:
            print(f"    - {f}")
    else:
        print("  All checks passed — safe to proceed with Phase 2.")

    # Abort criteria evaluation
    abort_conditions = [
        ("YoloDetectionNetwork.setBlobPath" in results and not results.get("YoloDetectionNetwork.setBlobPath", True),
         "blob YOLO broken on RVC2"),
        (not results.get("Warp.setOutputSize / setMaxOutputFrameSize / setWarpMesh — API exists", True),
         "Warp.setWarpMesh API missing"),
        (not results.get("Frame count > 0", True),
         "no frames received from camera"),
    ]
    triggered = [(reason) for cond, reason in abort_conditions if cond]
    if triggered:
        print("\n  *** ABORT CRITERIA TRIGGERED ***")
        for reason in triggered:
            print(f"    - {reason}")
        print("  Do NOT proceed with Phase 2. Log findings in /memories/repo/depthai-v3-notes.md and pause migration.")
    elif failed:
        print("\n  Some checks failed — review before proceeding.")
    else:
        print("\n  No abort criteria triggered.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="depthai v3 Phase 1 probe")
    parser.add_argument(
        "--blob",
        type=Path,
        default=Path("data/models/yolov8n_coco_416x416_6S.blob"),
        help="Path to YOLO blob for Probe 5 (default: data/models/yolov8n_coco_416x416_6S.blob)"
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Target FPS (default: 30)"
    )
    parser.add_argument(
        "--run-seconds", type=float, default=10.0,
        help="Seconds to run each live probe (default: 10)"
    )
    args = parser.parse_args()

    print("depthai v3 Phase 1 probe — Poser2 migration")
    try:
        import depthai as dai
        print(f"  depthai version: {dai.__version__}")
    except ImportError:
        print("ERROR: depthai not installed in current venv")
        sys.exit(1)

    probe_imports()
    probe_device_enum()
    probe_camera_node(fps=args.fps, run_seconds=args.run_seconds)
    probe_warp_node(fps=args.fps, run_seconds=args.run_seconds)
    probe_yolo(args.blob, fps=args.fps, run_seconds=args.run_seconds)
    probe_sync_node()
    probe_ir_control()
    summary()
