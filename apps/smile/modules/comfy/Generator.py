"""Image generator: sends each capture to ComfyUI and turns its output into GenMessages.

Per capture: depth of the capture (CAM_DEPTH, starts the flash out) -> sampler previews and the
finished portrait (DIFFUSION) -> the grin frames, streamed while ComfyUI makes them and played at
a fixed fps (DIFF_LOOP) -> depth of the last frame (IDLE, the bloom mask) -> IDLE.
Any failure or timeout ends with IDLE so the sequence can't hang; a reset (interrupt) ends
silently because the sequence has already moved on.
"""

import logging
import os
import time
from collections import deque
from threading import Event, Lock, Thread

import numpy as np
import websocket
from requests import RequestException

from modules.comfy.client import ComfyClient, Event as ComfyEvent
from modules.comfy.messages import GenMessage, GenState, MessageCallback
from modules.comfy.settings import ComfySettings
from modules.comfy.styles import Styles
from modules.comfy.workflow import Workflow

logger = logging.getLogger(__name__)

WARMUP_IMAGE: str = 'data/comfy/warmup.jpg'
RECONNECT_INTERVAL: float = 2.0


class Interrupted(Exception):
    pass


class Generator(Thread):
    def __init__(self, settings: ComfySettings) -> None:
        super().__init__(name='generator')
        self.settings: ComfySettings = settings
        self.workflow: Workflow = Workflow(settings.workflow)
        self.styles: Styles = Styles(settings.styles)
        self.client: ComfyClient = ComfyClient(settings.host, settings.port)

        self._mutex: Lock = Lock()
        self._callbacks: list[MessageCallback] = []
        self._image: np.ndarray | None = None
        self._stop_event: Event = Event()
        self._interrupt_event: Event = Event()
        self._silent: bool = False
        self._warmed_up: bool = False
        self.is_diffusing: bool = False

    # PUBLIC
    def stop(self) -> None:
        self._stop_event.set()
        self.interrupt()
        if self.is_alive():
            self.join()
        self.client.close()

    def set_image(self, image: np.ndarray) -> None:
        with self._mutex:
            self._image = image

    def interrupt(self) -> None:
        if self.is_diffusing:
            self._interrupt_event.set()

    def next_style(self) -> str:
        """Call once per visitor (at the flash); returns the style id, which also picks the music."""
        style_id: str = self.styles.advance(self.settings.style_repeat)
        self.settings.style = style_id
        return style_id

    def display_crop(self, image: np.ndarray) -> np.ndarray:
        """The part of a portrait-framed image that the grin frames show, so everything lines up."""
        h, w = image.shape[:2]
        x, y, cw, ch = self.workflow.grin_crop
        return image[int(y * h):int((y + ch) * h), int(x * w):int((x + cw) * w)]

    def add_message_callback(self, callback: MessageCallback) -> None:
        with self._mutex:
            self._callbacks.append(callback)

    def clear_message_callbacks(self) -> None:
        with self._mutex:
            self._callbacks = []

    # THREAD
    def run(self) -> None:
        while not self._stop_event.is_set():
            if not self.client.connected:
                if not self._connect():
                    self._stop_event.wait(RECONNECT_INTERVAL)
                    continue
            try:
                if not self._warmed_up:
                    self._warm_up()
                image: np.ndarray | None = self._take_image()
                if image is None:
                    self.client.receive(0.05)       # keep the websocket drained while idle
                    continue
                self._generate(image)
            except (websocket.WebSocketException, RequestException, ConnectionError, OSError):
                logger.warning('Lost the connection to ComfyUI', exc_info=True)
                self.client.close()
                self.settings.connected = False

    def _connect(self) -> bool:
        try:
            self.client.connect()
        except Exception:
            self.settings.connected = False
            self.settings.status = f'waiting for ComfyUI at {self.settings.host}:{self.settings.port}'
            return False
        self.settings.connected = True
        self.settings.status = 'ready'
        logger.info('Connected to ComfyUI at %s:%s', self.settings.host, self.settings.port)
        return True

    def _take_image(self) -> np.ndarray | None:
        with self._mutex:
            image, self._image = self._image, None
            return image

    def _warm_up(self) -> None:
        # the first run loads all models (~80 s); do it before the first visitor if an image is there
        self._warmed_up = True
        if not os.path.exists(WARMUP_IMAGE):
            return
        import cv2
        image: np.ndarray | None = cv2.imread(WARMUP_IMAGE)
        if image is None:
            return
        self.settings.status = 'warming up'
        self._silent = True
        try:
            self._generate(image)
        finally:
            self._silent = False
        self.settings.status = 'ready'

    # GENERATION
    def _emit(self, message: GenMessage) -> None:
        if self._silent or self._interrupt_event.is_set():
            return
        with self._mutex:
            callbacks = list(self._callbacks)
        for c in callbacks:
            c(message)

    def _generate(self, image: np.ndarray) -> None:
        start: float = time.time()
        self._interrupt_event.clear()
        self.is_diffusing = True
        try:
            self._run_job(image, start)
        except Interrupted:
            logger.info('Generation interrupted')
        except (websocket.WebSocketException, RequestException, ConnectionError, OSError):
            self._emit(GenMessage(GenState.IDLE))
            raise
        except Exception:
            logger.exception('Generation failed')
            self._emit(GenMessage(GenState.IDLE))
        finally:
            self.is_diffusing = False
            self._interrupt_event.clear()
            self.settings.last_duration = round(time.time() - start, 1)
            if self.settings.connected:
                self.settings.status = 'ready'

    def _check(self, deadline: float) -> None:
        if self._stop_event.is_set() or self._interrupt_event.is_set():
            self._interrupt_comfy()
            raise Interrupted()
        if time.time() > deadline:
            self._interrupt_comfy()
            raise TimeoutError('generation took longer than job_timeout')

    def _interrupt_comfy(self) -> None:
        try:
            self.client.interrupt()
        except Exception:
            logger.warning('Could not interrupt ComfyUI', exc_info=True)

    def _run_job(self, image: np.ndarray, start: float) -> None:
        s: ComfySettings = self.settings
        deadline: float = start + s.job_timeout
        total: int = s.frames

        s.status = 'generating'
        wf: Workflow = self.workflow
        name: str = self.client.upload(image, 'smile_capture.png')
        main_id: str = self.client.queue(wf.build(name, self.styles.current_text(), total))

        frames: deque[np.ndarray] = deque()
        playing: bool = False
        next_frame: float = 0.0
        played: int = 0
        last_depth: np.ndarray | None = None
        sampler_progress: float = 0.0
        done: bool = False

        while True:
            self._check(deadline)
            now: float = time.time()

            # play the grin at a fixed fps once enough frames are buffered
            if frames and (playing or done or len(frames) >= s.prebuffer):
                if not playing:
                    playing = True
                    next_frame = now
                if now >= next_frame:
                    frame: np.ndarray = frames.popleft()
                    played += 1
                    self._emit(GenMessage(GenState.DIFF_LOOP, min(played / total, 1.0), frame))
                    interval: float = 1.0 / max(s.fps, 1.0)
                    # after running out of frames, continue from now instead of catching up in a burst
                    next_frame = next_frame + interval if now - next_frame < interval else now + interval
            if done and not frames:
                break

            wait: float = 0.01
            if playing and frames:
                wait = max(0.001, min(0.01, next_frame - time.time()))
            event: ComfyEvent | None = self.client.receive(wait)
            if event is None:
                continue
            prompt_id: str | None = event.data.get('prompt_id')

            if prompt_id != main_id:
                continue

            if event.type == 'preview' and event.image is not None:
                node: str | None = event.data.get('node_id')
                if node == wf.sampler:
                    self._emit(GenMessage(GenState.DIFFUSION, sampler_progress, self.display_crop(event.image)))
                elif node == wf.grin:
                    frames.append(event.image)

            elif event.type == 'progress':
                if event.data.get('node') == wf.sampler:
                    sampler_progress = event.data['value'] / max(event.data['max'], 1)

            elif event.type == 'executed':
                node = event.data.get('node')
                output: dict = event.data.get('output') or {}
                if node == wf.capture_depth and output.get('images'):
                    depth: np.ndarray | None = self.client.view(output['images'][0])
                    if depth is not None:
                        self._emit(GenMessage(GenState.CAM_DEPTH, 0.0, self.display_crop(depth)))
                elif node == wf.portrait and output.get('images'):
                    portrait: np.ndarray | None = self.client.view(output['images'][0])
                    if portrait is not None:
                        self._emit(GenMessage(GenState.DIFFUSION, 1.0, self.display_crop(portrait), final=True))
                elif node == wf.last_depth and output.get('images'):
                    last_depth = self.client.view(output['images'][0])
                elif node == wf.description and output.get('text'):
                    s.description = str(output['text'][0])

            elif event.type == 'execution_success':
                done = True

            elif event.type in ('execution_error', 'execution_interrupted'):
                if self._interrupt_event.is_set():
                    raise Interrupted()
                raise RuntimeError(f"ComfyUI {event.type}: {event.data.get('exception_message', '')}")

        # the depth of the last grin frame is the mask for the bloom back to the camera
        self._emit(GenMessage(GenState.IDLE, 1.0, last_depth))
        self._emit(GenMessage(GenState.IDLE))
