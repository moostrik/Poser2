"""Image generator: sends each capture to ComfyUI and turns its output into GenMessages.

Per capture: depth of the capture (CAM_DEPTH, starts the flash out) -> sampler previews and the
finished portrait (DIFFUSION) -> the grin frames as ComfyUI makes them (DIFF_LOOP, for saving; the
GrinPlayer gets them too and the render shows them from there on its own clock) -> once the last
frame has been shown: depth of the last frame (IDLE, the bloom mask) -> IDLE.
With grin.loop or grin.still on, the grin (or its peak pose as one frame) is made again (DIFF_LOOP)
with the panel's current grin values until both are off; the portrait comes from ComfyUI's cache.
Any failure or timeout ends with IDLE so the sequence can't hang; a reset (interrupt) ends
silently because the sequence has already moved on.
"""

import logging
import os
import time
from threading import Event, Lock, Thread

import numpy as np
import websocket
from requests import RequestException

from modules.comfy.client import ComfyClient, Event as ComfyEvent
from modules.render.GrinPlayer import GrinPlayer
from modules.comfy.messages import GenMessage, GenState, MessageCallback
from modules.comfy.settings import PipelineSettings
from modules.comfy.styles import Styles
from modules.comfy.workflow import Workflow

logger = logging.getLogger(__name__)

WARMUP_IMAGE: str = 'data/comfy/warmup.jpg'
RECONNECT_INTERVAL: float = 2.0


class Interrupted(Exception):
    pass


class Generator(Thread):
    def __init__(self, settings: PipelineSettings, player: GrinPlayer) -> None:
        super().__init__(name='generator')
        self.settings: PipelineSettings = settings
        self.player: GrinPlayer = player
        self.workflow: Workflow = Workflow(settings.connection.workflow)
        self.styles: Styles = Styles(settings.style)
        self.client: ComfyClient = ComfyClient(settings.connection.host, settings.connection.port)

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
            self.player.clear()

    def next_style(self) -> str:
        """Call once per visitor (when its sequence starts); returns the style id, which also picks the music."""
        return self.styles.advance()

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
                self.settings.connection.connected = False

    def _connect(self) -> bool:
        c = self.settings.connection
        try:
            self.client.connect()
        except Exception:
            c.connected = False
            c.status = f'waiting for ComfyUI at {c.host}:{c.port}'
            return False
        c.connected = True
        c.status = 'ready'
        logger.info('Connected to ComfyUI at %s:%s', c.host, c.port)
        self._load_options()
        return True

    def _load_options(self) -> None:
        """The choices the panel offers, as this ComfyUI server has them."""
        try:
            sampler: dict = self.client.object_info('KSampler')['input']['required']
        except Exception:
            logger.warning('Could not read the option lists from ComfyUI', exc_info=True)
            return
        self.settings.portrait.sampler_options = list(sampler['sampler_name'][0])
        self.settings.portrait.scheduler_options = list(sampler['scheduler'][0])

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
        if self._is_warm():
            logger.info('ComfyUI is warm, no warm-up')
            return
        self.settings.connection.status = 'warming up'
        self._silent = True
        try:
            self._generate(image)
        finally:
            self._silent = False
        self.settings.connection.status = 'ready'

    def _is_warm(self) -> bool:
        """ComfyUI's last run was this pipeline: its models are still loaded."""
        try:
            return self.workflow.same_pipeline(self.client.last_prompt())
        except Exception:
            logger.warning('Could not read the last run from ComfyUI', exc_info=True)
            return False

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
        except TimeoutError as e:       # our own job_timeout (an OSError, but not a lost connection)
            logger.warning('Generation timed out: %s', e)
            self._emit(GenMessage(GenState.IDLE))
        except (websocket.WebSocketException, RequestException, ConnectionError, OSError):
            self._emit(GenMessage(GenState.IDLE))
            raise
        except Exception:
            logger.exception('Generation failed')
            self._emit(GenMessage(GenState.IDLE))
        finally:
            self.is_diffusing = False
            self._interrupt_event.clear()
            self.settings.connection.last_duration = round(time.time() - start, 1)
            if self.settings.connection.connected:
                self.settings.connection.status = 'ready'

    def _check(self, deadline: float, last_event: str = '') -> None:
        if self._stop_event.is_set() or self._interrupt_event.is_set():
            self._interrupt_comfy()
            raise Interrupted()
        if time.time() > deadline:
            self._interrupt_comfy()
            raise TimeoutError(f'longer than job_timeout{last_event}')

    def _step(self, step: str) -> None:
        """The step of the generation in the panel; the warm-up says so in front."""
        status: str = f'warming up: {step}' if self._silent else step
        if self.settings.connection.status != status:
            self.settings.connection.status = status

    def _interrupt_comfy(self) -> None:
        try:
            self.client.interrupt()
        except Exception:
            logger.warning('Could not interrupt ComfyUI', exc_info=True)

    def _idle(self, seconds: float) -> None:
        """Wait with the websocket drained; stop and interrupt still end the job."""
        end: float = time.time() + seconds
        while time.time() < end:
            self._check(float('inf'))
            self.client.receive(0.05)

    def _run_job(self, image: np.ndarray, start: float) -> None:
        s: PipelineSettings = self.settings
        wf: Workflow = self.workflow

        if self.client.busy():
            self._idle(1.0)     # our previous job leaves the queue just after its last event
        if self.client.busy():
            # a job that timed out is still running (ComfyUI only stops between steps): don't queue behind it
            self._interrupt_comfy()
            raise RuntimeError('ComfyUI is still busy with an earlier job')
        self._step('generating')
        name: str = self.client.upload(image, 'smile_capture.png')
        workflow: dict = wf.build(name, self.styles.current_text(), self.styles.current_negative(), s)
        last_depth, _ = self._run_pass(workflow, start + s.connection.job_timeout)

        # grin loop or still (testing): the grin again, with the values the panel has now; not for the warm-up
        while (s.grin.loop or s.grin.still) and not self._silent:
            s.connection.status = 'grin still' if s.grin.still else 'grin loop'
            replay: dict = wf.build_replay(workflow, s)
            if replay == workflow:
                # nothing changed (a still, or fixed seed): ComfyUI would answer from its cache, without frames
                self._idle(0.1)
                continue
            workflow = replay
            depth, made = self._run_pass(workflow, time.time() + s.connection.job_timeout, replay=True)
            if depth is not None:
                last_depth = depth
            if made == 0:
                self._idle(0.5)

        # the depth of the last grin frame is the mask for the bloom back to the camera
        self._emit(GenMessage(GenState.IDLE, 1.0, last_depth))
        self._emit(GenMessage(GenState.IDLE))

    def _run_pass(self, workflow: dict, deadline: float, replay: bool = False) -> tuple[np.ndarray | None, int]:
        """Queue the workflow and hand its grin to the player; returns, once the last frame has been shown,
        the depth of that frame and the frames made.
        A replay only makes the grin: ComfyUI reports the cached capture depth and portrait again."""
        try:
            return self._receive_pass(workflow, deadline, replay)
        except BaseException:
            self.player.clear()     # after a failure or a reset nothing of the grin is left to play
            raise

    def _receive_pass(self, workflow: dict, deadline: float, replay: bool) -> tuple[np.ndarray | None, int]:
        s: PipelineSettings = self.settings
        wf: Workflow = self.workflow
        total: int = wf.frame_count(workflow)
        if not self._silent:
            # the first grin after a portrait holds on its first frame (the blend); the warm-up opens no clip
            self.player.begin(total, hold=not replay)
        main_id: str = self.client.queue(workflow)

        made: int = 0
        last_depth: np.ndarray | None = None
        sampler_progress: float = 0.0
        last_event: str = 'queued'      # for the timeout message: where ComfyUI went silent
        last_time: float = time.time()
        step: str = 'queued'            # the same for the panel: the step that is running

        def show(text: str) -> None:
            nonlocal step
            step = text
            if not replay:              # a replay keeps 'grin loop' / 'grin still'
                self._step(text)

        while True:
            self._check(deadline, f", no event for {time.time() - last_time:.0f} s after '{last_event}' during '{step}'")
            event: ComfyEvent | None = self.client.receive(0.05)
            if event is None:
                continue
            prompt_id: str | None = event.data.get('prompt_id')

            if prompt_id != main_id:
                continue
            last_event = f"{event.type} {event.data.get('node_id') or event.data.get('node') or ''}".strip()
            last_time = time.time()

            if event.type == 'preview' and event.image is not None:
                node: str | None = event.data.get('node_id')
                if node == wf.grin:
                    made += 1
                    show(f'grin {made}/{total}')
                    self.player.add(event.image)
                    self._emit(GenMessage(GenState.DIFF_LOOP, min(made / total, 1.0), event.image))
                elif node == wf.sampler and not replay:
                    self._emit(GenMessage(GenState.DIFFUSION, sampler_progress, event.image))

            elif event.type == 'executing':
                name: str | None = wf.step(event.data.get('node'))
                if name is not None:
                    show(name)

            elif event.type == 'progress':
                if event.data.get('node') == wf.sampler:
                    sampler_progress = event.data['value'] / max(event.data['max'], 1)
                    show(f"portrait {event.data['value']}/{event.data['max']}")

            elif event.type == 'executed':
                node = event.data.get('node')
                output: dict = event.data.get('output') or {}
                if node == wf.last_depth and output.get('images'):
                    last_depth = self.client.view(output['images'][0])
                elif replay:
                    pass        # the cached capture depth, portrait and description, reported again
                elif node == wf.capture_depth and output.get('images'):
                    depth: np.ndarray | None = self.client.view(output['images'][0])
                    if depth is not None:
                        self._emit(GenMessage(GenState.CAM_DEPTH, 0.0, depth))
                elif node == wf.portrait and output.get('images'):
                    portrait: np.ndarray | None = self.client.view(output['images'][0])
                    if portrait is not None:
                        self._emit(GenMessage(GenState.DIFFUSION, 1.0, portrait, final=True))
                        self.player.hold_from_now()
                elif node == wf.description and output.get('text'):
                    s.describe.description = str(output['text'][0])

            elif event.type == 'execution_success':
                break

            elif event.type in ('execution_error', 'execution_interrupted'):
                if self._interrupt_event.is_set():
                    raise Interrupted()
                raise RuntimeError(f"ComfyUI {event.type}: {event.data.get('exception_message', '')}")

        # the render shows the frames on its own clock: IDLE and the next pass wait for the last one
        self.player.end()
        if self.player.busy:
            show('playing grin')
        while self.player.busy:
            self._check(deadline)
            self.client.receive(0.02)
        self._check(deadline)       # a reset empties the player too
        return last_depth, made
