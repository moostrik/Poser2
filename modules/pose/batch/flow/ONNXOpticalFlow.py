# Standard library imports
from dataclasses import dataclass, field
from queue import Queue, Empty
from threading import Thread, Lock, Event
import time
import traceback
from typing import Callable

# Third-party imports
import numpy as np
import torch
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.pose.Settings import Settings


@dataclass
class OpticalFlowInput:
    """Batch of consecutive frame pairs for optical flow computation."""
    batch_id: int
    frame_pairs: list[tuple[np.ndarray, np.ndarray]]  # List of (prev_frame, curr_frame) pairs
    tracklet_ids: list[int] = field(default_factory=list)


@dataclass
class OpticalFlowOutput:
    """Results from optical flow computation. processed=False indicates batch was dropped."""
    batch_id: int
    flow_tensor: torch.Tensor | None = None  # GPU tensor (B, 2, H, W) FP32, flow field (x, y)
    tracklet_ids: list[int] = field(default_factory=list)
    processed: bool = True
    inference_time_ms: float = 0.0


OpticalFlowOutputCallback = Callable[[OpticalFlowOutput], None]


class ONNXOpticalFlow(Thread):
    """Asynchronous GPU optical flow computation using RAFT with ONNX Runtime.

    RAFT (Recurrent All-Pairs Field Transforms) computes dense optical flow between
    consecutive video frames using iterative refinement.

    Uses a single-slot queue: only the most recent submitted batch waits to be processed.
    Older pending batches are dropped. Batches already processing on GPU cannot be cancelled.

    All results (success and dropped) are delivered via callbacks in notification order.
    """

    def __init__(self, settings: 'Settings') -> None:
        super().__init__()

        self.enabled: bool = settings.flow_enabled if hasattr(settings, 'flow_enabled') else False
        if not self.enabled:
            print('Optical Flow WARNING: Optical flow is disabled')

        self.model_path: str = settings.model_path
        self.model_name: str = settings.flow_model
        self.model_file: str = f"{self.model_path}/{self.model_name}"
        self.model_width: int = settings.flow_width
        self.model_height: int = settings.flow_height
        self.resolution_name: str = settings.flow_resolution.name
        self.verbose: bool = settings.verbose

        # Thread coordination
        self._shutdown_event: Event = Event()
        self._notify_update_event: Event = Event()
        self._model_ready: Event = Event()

        # Input queue
        self._input_lock: Lock = Lock()
        self._pending_batch: OpticalFlowInput | None = None
        self._input_timestamp: float = time.time()
        self._last_dropped_batch_id: int = 0

        # Callbacks
        self._callback_lock: Lock = Lock()
        self._callbacks: set[OpticalFlowOutputCallback] = set()
        self._callback_queue: Queue[OpticalFlowOutput | None] = Queue(maxsize=2)
        self._callback_thread: Thread = Thread(target=self._callback_worker_loop, daemon=True)

        # ONNX session (initialized in run thread)
        self._session: ort.InferenceSession | None = None

        # Thread pool for parallel inference
        self._executor: ThreadPoolExecutor | None = None
        self._max_workers: int = min(getattr(settings, 'max_poses', 3), 4)

    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and system is ready to process optical flow"""
        return self._model_ready.is_set() and not self._shutdown_event.is_set() and self.is_alive()

    def start(self) -> None:
        if not self.enabled:
            return
        self._callback_thread.start()
        super().start()

    def stop(self) -> None:
        if not self.enabled:
            return
        """Stop both inference and callback threads gracefully"""
        self._shutdown_event.set()

        # Shutdown thread pool
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)

        # Wake up inference thread
        self._notify_update_event.set()
        self.join(timeout=2.0)

        if self.is_alive():
            print("Warning: RAFT Optical Flow inference thread did not stop cleanly")

        # Wake up callback thread with sentinel
        try:
            self._callback_queue.put_nowait(None)
        except:
            pass

        self._callback_thread.join(timeout=2.0)
        if self._callback_thread.is_alive():
            print("Warning: RAFT Optical Flow callback thread did not stop cleanly")

    def run(self) -> None:
        """Main inference thread loop. Initializes ONNX session and processes batches."""
        try:
            # Initialize ONNX Runtime session with CUDA provider
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider',
            ]

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 1
            sess_options.log_severity_level = 3

            self._session = ort.InferenceSession(
                self.model_file,
                sess_options=sess_options,
                providers=providers
            )

            # Verify CUDA provider is active
            providers_used = self._session.get_providers()
            if 'CUDAExecutionProvider' not in providers_used:
                print("RAFT Optical Flow WARNING: CUDA provider not available, using CPU")

            # Warmup: initialize CUDA kernels with dummy inference
            self._model_warmup(self._session)

            # Create thread pool for parallel inference
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="RAFT-Worker")

            self._model_ready.set()
            print(f"RAFT Optical Flow: {self.resolution_name} model loaded ({self.model_width}x{self.model_height}, {providers_used[0]}) with {self._max_workers} workers")

        except Exception as e:
            print(f"RAFT Optical Flow Error: Failed to load model - {str(e)}")
            traceback.print_exc()
            return

        while not self._shutdown_event.is_set():
            self._notify_update_event.wait()

            if self._shutdown_event.is_set():
                break

            self._notify_update_event.clear()

            try:
                self._process_pending_batch()

            except Exception as e:
                print(f"RAFT Optical Flow Error: {str(e)}")
                traceback.print_exc()

    def submit_batch(self, input_batch: OpticalFlowInput) -> None:
        """Submit batch for processing. Replaces any pending (not yet started) batch."""
        if self._shutdown_event.is_set():
            return

        if not self._model_ready.is_set():
            return

        dropped_batch: OpticalFlowInput | None = None

        with self._input_lock:
            if self._pending_batch is not None:
                dropped_batch = self._pending_batch
                lag = int((time.time() - self._input_timestamp) * 1000)
                # if self.verbose:
                # print(f"RAFT Optical Flow: Dropped batch {dropped_batch.batch_id} with lag {lag} ms")
                self._last_dropped_batch_id = dropped_batch.batch_id

            self._pending_batch = input_batch
            self._input_timestamp = time.time()

        # Notify about dropped batch
        if dropped_batch is not None:
            dropped_output = OpticalFlowOutput(
                batch_id=dropped_batch.batch_id,
                tracklet_ids=dropped_batch.tracklet_ids,
                processed=False
            )
            try:
                self._callback_queue.put_nowait(dropped_output)
            except:
                pass

        self._notify_update_event.set()

    def _retrieve_pending_batch(self) -> OpticalFlowInput | None:
        """Atomically get and clear pending batch."""
        with self._input_lock:
            batch = self._pending_batch
            self._pending_batch = None
            return batch

    def _process_pending_batch(self) -> None:
        """Process the pending batch using RAFT ONNX inference."""
        batch: OpticalFlowInput | None = self._retrieve_pending_batch()

        if batch is None:
            return

        output = OpticalFlowOutput(batch_id=batch.batch_id, tracklet_ids=batch.tracklet_ids, processed=True)

        # Run inference
        if batch.frame_pairs and self._session is not None and self._executor is not None:
            batch_start = time.perf_counter()

            # Process frame pairs in parallel using thread pool
            futures = []
            for (prev_frame, curr_frame), tracklet_id in zip(batch.frame_pairs, batch.tracklet_ids):
                future = self._executor.submit(self._infer_optical_flow, prev_frame, curr_frame)
                futures.append((future, tracklet_id))

            # Collect results in original order
            flow_list: list[torch.Tensor] = []
            for future, tracklet_id in futures:
                try:
                    flow = future.result()
                    flow_list.append(flow)
                except Exception as e:
                    print(f"RAFT Optical Flow Error: Inference failed for tracklet {tracklet_id}: {str(e)}")
                    # Create zero flow on error
                    h, w = batch.frame_pairs[0][0].shape[:2]
                    flow_list.append(torch.zeros((2, h, w), dtype=torch.float32, device='cuda'))

            # Stack into batch tensor
            if flow_list:
                flow_tensor = torch.stack(flow_list)  # (B, 2, H, W) FP32 on CUDA
            else:
                flow_tensor = torch.empty(0, dtype=torch.float32, device='cuda')

            inference_time_ms: float = (time.perf_counter() - batch_start) * 1000.0

            output = OpticalFlowOutput(
                batch_id=batch.batch_id,
                flow_tensor=flow_tensor,
                tracklet_ids=batch.tracklet_ids,
                processed=True,
                inference_time_ms=inference_time_ms
            )

        # Queue for callbacks
        try:
            self._callback_queue.put_nowait(output)
        except Exception:
            print("RAFT Optical Flow Warning: Callback queue full, dropping results")

    def _infer_optical_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> torch.Tensor:
        """Run RAFT inference on frame pair.

        Args:
            prev_frame: Previous frame (H, W, 3) BGR uint8
            curr_frame: Current frame (H, W, 3) BGR uint8

        Returns:
            Flow tensor (2, H, W) FP32 on CUDA, where [0] is x-flow, [1] is y-flow
        """
        if self._session is None:
            h, w = prev_frame.shape[:2]
            return torch.zeros((2, h, w), dtype=torch.float32, device='cuda')

        # Preprocess: BGR -> RGB, HWC -> NCHW, keep in [0, 255] range (RAFT expects this)
        def preprocess(img: np.ndarray) -> np.ndarray:
            img_rgb = img[:, :, ::-1].astype(np.float32)  # BGR to RGB, FP32
            img_chw = np.transpose(img_rgb, (2, 0, 1))  # (H, W, 3) -> (3, H, W)
            img_nchw = np.expand_dims(img_chw, axis=0)  # (1, 3, H, W)
            return img_nchw

        image1 = preprocess(prev_frame)
        image2 = preprocess(curr_frame)

        # Run ONNX inference
        # RAFT model expects inputs: image1, image2
        # Output: flow (1, 2, H, W)
        input_names = [inp.name for inp in self._session.get_inputs()]
        onnx_inputs = {
            input_names[0]: image1,
            input_names[1]: image2
        }

        outputs = self._session.run(None, onnx_inputs)
        flow_np = outputs[0]  # (1, 2, H, W) FP32

        # Convert to PyTorch CUDA tensor
        flow_tensor = torch.from_numpy(flow_np).cuda()  # (1, 2, H, W)
        flow_tensor = flow_tensor.squeeze(0)  # (2, H, W)

        return flow_tensor

    def _callback_worker_loop(self) -> None:
        """Dispatch queued results to registered callbacks."""
        while not self._shutdown_event.is_set():
            try:
                output: OpticalFlowOutput | None = self._callback_queue.get(timeout=0.5)

                if output is None:
                    break

                with self._callback_lock:
                    callbacks = list(self._callbacks)

                for callback in callbacks:
                    try:
                        callback(output)
                    except Exception as e:
                        print(f"RAFT Optical Flow Callback Error: {str(e)}")
                        traceback.print_exc()

                self._callback_queue.task_done()
            except Empty:
                continue

    def register_callback(self, callback: OpticalFlowOutputCallback) -> None:
        """Register callback to receive optical flow results."""
        with self._callback_lock:
            self._callbacks.add(callback)

    def _model_warmup(self, session: ort.InferenceSession) -> None:
        """Initialize CUDA kernels with dummy inference."""
        try:
            dummy_frame = np.zeros((self.model_height, self.model_width, 3), dtype=np.uint8)

            for i in range(2):
                _ = self._infer_optical_flow(dummy_frame, dummy_frame)

            print("RAFT Optical Flow: Warmup complete")
        except Exception as e:
            print(f"RAFT Optical Flow: Warmup failed (non-critical) - {str(e)}")
