# Standard library imports
from collections import OrderedDict
from dataclasses import dataclass, field
from queue import Queue, Empty
from threading import Thread, Lock, Event
import time
import traceback
from typing import Callable

# Third-party imports
import numpy as np
import torch

# Ensure numpy functions can be safely used in torch serialization
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.Float32DType, np.dtypes.UInt8DType]) # pyright: ignore

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.pose.Settings import Settings


@dataclass
class SegmentationInput:
    """Batch of images for segmentation. Images should be (H, W, 3) BGR uint8."""
    batch_id: int
    images: list[np.ndarray]
    tracklet_ids: list[int] = field(default_factory=list)  # Corresponding tracklet IDs for each image


@dataclass
class SegmentationOutput:
    """Results from segmentation. processed=False indicates batch was dropped."""
    batch_id: int
    mask_tensor: torch.Tensor | None = None    # GPU tensor (B, H, W) FP16, alpha matte [0, 1]
    tracklet_ids: list[int] = field(default_factory=list)  # Corresponding tracklet IDs
    processed: bool = True          # False if batch was dropped before processing
    inference_time_ms: float = 0.0  # For monitoring


SegmentationOutputCallback = Callable[[SegmentationOutput], None]


class MODNetSegmentation(Thread):
    """Asynchronous GPU person segmentation using MODNet.

    Uses a single-slot queue: only the most recent submitted batch waits to be processed.
    Older pending batches are dropped. Batches already processing on GPU cannot be cancelled.

    All results (success and dropped) are delivered via callbacks in notification order,
    which may differ from batch_id order due to async processing.
    """

    def __init__(self, settings: 'Settings') -> None:
        super().__init__()

        self.enabled: bool = settings.segmentation_enabled
        if not self.enabled:
            print('Segmentation WARNING: Segmentation is disabled')

        self.model_path: str = settings.model_path
        self.model_name: str = settings.segmentation_model_name
        self.model_checkpoint: str = f"{self.model_path}/{self.model_name}"
        self.verbose: bool = settings.verbose

        # Thread coordination
        self._shutdown_event: Event = Event()
        self._notify_update_event: Event = Event()
        self._model_ready: Event = Event()

        # Input queue
        self._input_lock: Lock = Lock()
        self._pending_batch: SegmentationInput | None = None
        self._input_timestamp: float = time.time()
        self._last_dropped_batch_id: int = 0

        # Callbacks
        self._callback_lock: Lock = Lock()
        self._callbacks: set[SegmentationOutputCallback] = set()
        self._callback_queue: Queue[SegmentationOutput | None] = Queue(maxsize=2)
        self._callback_thread: Thread = Thread(target=self._callback_worker_loop, daemon=True)

    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and system is ready to process segmentation"""
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

        # Wake up inference thread
        self._notify_update_event.set()
        self.join(timeout=2.0)

        if self.is_alive():
            print("Warning: Segmentation inference thread did not stop cleanly")

        # Wake up callback thread with sentinel
        try:
            self._callback_queue.put_nowait(None)
        except:
            pass

        self._callback_thread.join(timeout=2.0)
        if self._callback_thread.is_alive():
            print("Warning: Segmentation callback thread did not stop cleanly")

    def run(self) -> None:
        torch.cuda.set_device(0)
        stream = torch.cuda.Stream(device=0, priority=-1)
        torch.cuda.set_stream(stream)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Load MODNet model
        from modules.pose.segmentation.modnet import MODNet

        model = MODNet(backbone_pretrained=False)
        checkpoint = torch.load(self.model_checkpoint)

        # Strip 'module.' prefix from DataParallel checkpoint
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.half().cuda().eval()

        self._model_ready.set()  # Signal model is ready
        print(f"Segmentation: MODNet model ready")

        while not self._shutdown_event.is_set():
            self._notify_update_event.wait()

            if self._shutdown_event.is_set():
                break

            self._notify_update_event.clear()

            try:
                self._process_pending_batch(model, stream)

            except Exception as e:
                print(f"Segmentation Error: {str(e)}")
                traceback.print_exc()

    def submit_batch(self, input_batch: SegmentationInput) -> None:
        """Submit batch for processing. Replaces any pending (not yet started) batch.

        Dropped batches trigger callbacks with processed=False.
        """
        if self._shutdown_event.is_set():
            return

        dropped_batch: SegmentationInput | None = None

        with self._input_lock:
            if self._pending_batch is not None:
                dropped_batch = self._pending_batch
                if self.verbose:
                    lag = int((time.time() - self._input_timestamp) * 1000)
                    print(f"Segmentation: Dropped batch {dropped_batch.batch_id} with lag {lag} ms after {dropped_batch.batch_id - self._last_dropped_batch_id} batches")
                self._last_dropped_batch_id = dropped_batch.batch_id

            self._pending_batch = input_batch
            self._input_timestamp = time.time()

        # Notify about dropped batch
        if dropped_batch is not None:
            dropped_output = SegmentationOutput(
                batch_id=dropped_batch.batch_id,
                tracklet_ids=dropped_batch.tracklet_ids,
                processed=False  # Mark as not processed
            )
            try:
                self._callback_queue.put_nowait(dropped_output)
            except:
                if self.verbose:
                    print("Segmentation Warning: Callback queue full, not critical for dropped notifications")
                pass  # Queue full, not critical for dropped notifications

        self._notify_update_event.set()

    def _retrieve_pending_batch(self) -> SegmentationInput | None:
        """Atomically get and clear pending batch. Once retrieved, batch cannot be cancelled.

        Returns:
            The pending batch if one exists, None otherwise
        """
        with self._input_lock:
            batch = self._pending_batch
            self._pending_batch = None  # Clear slot - batch is now committed
            return batch

    def _process_pending_batch(self, model: torch.nn.Module, stream: torch.cuda.Stream) -> None:
        # Get pending input
        batch: SegmentationInput | None = self._retrieve_pending_batch()

        if batch is None:
            if self.verbose:
                print("Segmentation Warning: No pending batch to process, this should not happen")
            return

        output = SegmentationOutput(batch_id=batch.batch_id, tracklet_ids=batch.tracklet_ids, processed=True)

        # Run inference
        if batch.images:
            batch_start = time.perf_counter()

            with torch.cuda.stream(stream):
                mask_tensor: torch.Tensor = MODNetSegmentation._infer_segmentation(model, batch.images)
                stream.synchronize()

            inference_time_ms: float = (time.perf_counter() - batch_start) * 1000.0

            # Create output (processed=True by default) - keep tensor on GPU
            output = SegmentationOutput(
                batch_id=batch.batch_id,
                mask_tensor=mask_tensor,
                tracklet_ids=batch.tracklet_ids,
                processed=True,
                inference_time_ms=inference_time_ms
            )

        # Queue for callbacks
        try:
            self._callback_queue.put_nowait(output)
        except Exception:
            print("Segmentation Warning: Callback queue full, dropping inference results")

    # CALLBACK
    def _callback_worker_loop(self) -> None:
        """Dispatch queued results to registered callbacks. Runs on separate thread."""
        while not self._shutdown_event.is_set():
            try:
                if self._callback_queue.qsize() > 1:
                    print("Segmentation Warning: Callback queue size > 1, consumers may be falling behind")

                output: SegmentationOutput | None = self._callback_queue.get(timeout=0.5)

                if output is None:
                    break

                with self._callback_lock:
                    callbacks = list(self._callbacks)

                for callback in callbacks:
                    try:
                        callback(output)
                    except Exception as e:
                        print(f"Segmentation Callback Error: {str(e)}")
                        traceback.print_exc()

                self._callback_queue.task_done()
            except Empty:
                continue

    def register_callback(self, callback: SegmentationOutputCallback) -> None:
        """Register callback to receive segmentation results (both success and dropped batches)."""
        with self._callback_lock:
            self._callbacks.add(callback)

    # STATIC METHODS
    @staticmethod
    def _infer_segmentation(model: torch.nn.Module, imgs: list[np.ndarray]) -> torch.Tensor:
        """Run MODNet segmentation on batch of images. Returns GPU tensor (B, H, W) [0-1].

        Masks stay on GPU for efficient conversion to OpenGL textures.
        """
        if not imgs:
            return torch.empty(0, dtype=torch.float16, device='cuda')

        # Convert batch to tensor efficiently
        # imgs are (H, W, 3) uint8 BGR - convert to RGB and normalize to [-1, 1]
        batch_list = []
        for img in imgs:
            # Convert BGR to RGB and to float [0, 255]
            img_rgb = img[:, :, ::-1].astype(np.float32)
            # Normalize to [-1, 1]: (x / 255 - 0.5) / 0.5 = x / 127.5 - 1
            img_normalized = (img_rgb / 127.5) - 1.0
            # Convert to tensor (H, W, 3) -> (3, H, W)
            img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)
            batch_list.append(img_tensor)

        # Stack and upload to GPU as FP16
        batch = torch.stack(batch_list).cuda().half()  # (B, 3, H, W) FP16

        with torch.inference_mode():
            # MODNet forward returns (pred_semantic, pred_detail, pred_matte) when inference=False
            # or just pred_matte when inference=True, but original code may return tuple
            result = model(batch, True)
            # Handle both cases: tuple or single tensor
            if isinstance(result, tuple):
                matte = result[2]  # pred_matte is the third element
            else:
                matte = result  # Already the matte tensor

        # Return GPU tensor (B, H, W) - keep on GPU for OpenGL interop
        return matte.squeeze(1)  # (B, H, W) FP16 on CUDA
