"""Image generator: sends each capture to ComfyUI and turns its output into GenMessages.

Per capture: depth of the capture (CAM_DEPTH, starts the flash out) -> sampler previews and the
finished portrait (DIFFUSION) -> the grin frames as ComfyUI makes them (DIFF_LOOP, for saving; the
GrinPlayer gets them too and the render shows them from there on its own clock) -> once the last
frame has been shown: depth of the last frame (IDLE, the bloom mask) -> IDLE.
With grin.loop or grin.still on, the grin (or its peak pose as one frame) is made again (DIFF_LOOP)
with the panel's current grin values until both are off; the portrait comes from ComfyUI's cache.
With skip_generation (debug) only the description and the capture's depth are made:
CAM_DEPTH -> IDLE; with skip_animation the portrait too: CAM_DEPTH -> DIFFUSION -> IDLE.
Both with the capture's depth as the bloom mask.
A workflow that ComfyUI ran without sending one event is NoEvents: the generator is not ready until
a warm-up gets its events again, and a warm-up that fails is tried again instead of counting as ready.
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
from modules.comfy.settings import PipelineSettings, StyleSettings
from modules.comfy.styles import Styles
from modules.comfy.workflow import NO_LORA, Workflow

logger = logging.getLogger(__name__)

WARMUP_IMAGE: str = 'data/comfy/warmup.jpg'
WARMUP_TIMEOUT: float = 300.0   # s; the warm-up loads every model, a visitor has connection.job_timeout
WARMUP_RETRY: float = 5.0       # s before a failed warm-up is tried again
RECONNECT_INTERVAL: float = 2.0
SILENT_AFTER: float = 5.0       # s without any event of a queued workflow before asking whether it still runs
SILENT_RECHECK: float = 2.0
NO_EVENTS: str = 'no events from ComfyUI: close or reload its browser tab'
# the grin settings in a visitor's record; of the ranges it has what was drawn from them
RECORD_GRIN_FIELDS: tuple[str, ...] = ('frames', 'face_region', 'stitching', 'ease', 'midpoint', 'midpoint_deviation')


class Interrupted(Exception):
    pass


class NoEvents(RuntimeError):
    """ComfyUI ran the workflow without sending us one event. It sends the events of all clients from
    one queue: a client that stopped reading (a ComfyUI browser tab put to sleep) blocks it for us too."""


class Generator(Thread):
    def __init__(self, settings: PipelineSettings, player: GrinPlayer) -> None:
        super().__init__(name='generator')
        self.settings: PipelineSettings = settings
        self.player: GrinPlayer = player
        self.workflow: Workflow = Workflow(settings.connection.workflow)
        self.styles: Styles = Styles(settings)
        self.client: ComfyClient = ComfyClient(settings.connection.host, settings.connection.port)

        self._mutex: Lock = Lock()
        self._callbacks: list[MessageCallback] = []
        self._image: np.ndarray | None = None
        self._capture_depth: np.ndarray | None = None   # of the job that is running
        self._style: StyleSettings = self.styles.current()     # of the job that is running: the next visitor's may be picked already
        self._record: dict = {}                         # of the job that is running: what it was made with
        self._stop_event: Event = Event()
        self._interrupt_event: Event = Event()
        self._silent: bool = False
        self._warmed_up: bool = False
        self.is_diffusing: bool = False

    # PUBLIC
    @property
    def ready(self) -> bool:
        """Connected and warmed up: a capture handed over now is generated at once."""
        return self._warmed_up and self.client.connected

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

    def record(self) -> dict:
        """How the running visitor is made, for its saved files: description, seed, the settings of its
        style and of the portrait, and the grin as it was drawn (each value's peak and the moment of it,
        not the ranges). Call from a message callback, once the portrait is there."""
        return {'description': self._style.description, **self._record}

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
                self._warmed_up = False     # a ComfyUI that was restarted has to load its models again
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
                self._generate(image, self.settings.connection.job_timeout)
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
            loras: list[str] = list(self.client.object_info('LoraLoader')['input']['required']['lora_name'][0])
            checkpoints: list[str] = list(self.client.object_info('CheckpointLoaderSimple')['input']['required']['ckpt_name'][0])
        except Exception:
            logger.warning('Could not read the option lists from ComfyUI', exc_info=True)
            return
        self.settings.portrait.sampler_options = list(sampler['sampler_name'][0])
        self.settings.portrait.scheduler_options = list(sampler['scheduler'][0])
        self.settings.portrait.checkpoint_options = checkpoints
        for style in (self.settings.portrait, self.settings.style_a, self.settings.style_b):
            style.lora_options = [NO_LORA] + loras

    def _take_image(self) -> np.ndarray | None:
        with self._mutex:
            image, self._image = self._image, None
            return image

    def _warm_up(self) -> None:
        # A short run after every connect, if an image is there: it loads the models ComfyUI doesn't have
        # loaded (all of them when it is cold) and shows that the whole chain works before a visitor comes.
        # Until it is done the generator is not ready, and no sequence starts.
        import cv2
        image: np.ndarray | None = cv2.imread(WARMUP_IMAGE) if os.path.exists(WARMUP_IMAGE) else None
        if image is not None:
            self.settings.connection.status = 'warming up'
            self._silent = True
            try:
                failure: str | None = self._generate(image, WARMUP_TIMEOUT)
            finally:
                self._silent = False
            if failure is not None:
                # not ready: a visitor would fail the same way. The render shows why; try again shortly
                self.settings.connection.status = f'warm-up failed: {failure}'
                self._stop_event.wait(WARMUP_RETRY)
                return
            logger.info('Warm-up took %.1f s', self.settings.connection.last_duration)
        self._warmed_up = True
        self.settings.connection.status = 'ready'

    # GENERATION
    def _emit(self, message: GenMessage) -> None:
        if self._silent or self._interrupt_event.is_set():
            return
        with self._mutex:
            callbacks = list(self._callbacks)
        for c in callbacks:
            c(message)

    def _generate(self, image: np.ndarray, timeout: float) -> str | None:
        """Run one job; returns why it failed, or None when it was made (or interrupted)."""
        start: float = time.time()
        failure: str | None = None
        self._interrupt_event.clear()
        self.is_diffusing = True
        try:
            self._run_job(image, start + timeout)
        except Interrupted:
            logger.info('Generation interrupted')
        except TimeoutError as e:       # our own job_timeout (an OSError, but not a lost connection)
            logger.warning('Generation timed out: %s', e)
            failure = 'timed out'
            self._emit(GenMessage(GenState.IDLE))
        except (websocket.WebSocketException, RequestException, ConnectionError, OSError):
            self._emit(GenMessage(GenState.IDLE))
            raise
        except NoEvents as e:
            logger.warning('Generation failed: %s', e)
            failure = str(e)
            self._warmed_up = False     # not ready until a warm-up gets its events again
            self._emit(GenMessage(GenState.IDLE))
        except Exception:
            logger.exception('Generation failed')
            failure = 'failed (see the log)'
            self._emit(GenMessage(GenState.IDLE))
        finally:
            self.is_diffusing = False
            self._interrupt_event.clear()
            self.settings.connection.last_duration = round(time.time() - start, 1)
            if self.settings.connection.connected:
                self.settings.connection.status = 'ready'
        return failure

    def _check(self, deadline: float, last_event: str = '') -> None:
        if self._stop_event.is_set() or self._interrupt_event.is_set():
            self._interrupt_comfy()
            raise Interrupted()
        if time.time() > deadline:
            self._interrupt_comfy()
            raise TimeoutError(f'longer than its time limit{last_event}')

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

    def _run_job(self, image: np.ndarray, deadline: float) -> None:
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
        # debug: skip_generation leaves out the portrait and with it the grin, skip_animation only the
        # grin; the warm-up stays the full run
        no_portrait: bool = s.skip_generation and not self._silent
        no_grin: bool = no_portrait or (s.skip_animation and not self._silent)
        build = wf.build_warmup if self._silent else wf.build_describe if no_portrait else wf.build_portrait if no_grin else wf.build
        self._style = self.styles.current()
        workflow: dict = build(name, self._style, s)
        if not self._silent:
            # here, so the settings are the ones the workflow was filled with
            facts: dict = wf.run_facts(workflow)
            grin: dict = s.grin.to_dict()
            self._record = {'seed': facts['seed'],
                            'style_settings': self._style.to_dict(),
                            'portrait': s.portrait.to_dict(),
                            'grin': {**{name: grin[name] for name in RECORD_GRIN_FIELDS},
                                     'peak': facts['peak'], 'peak_at': facts['peak_at']}}
        self._capture_depth = None
        last_depth, _ = self._run_pass(workflow, deadline, play=not no_grin)
        if no_grin:
            last_depth = self._capture_depth    # no last grin frame: the capture's depth is the bloom's mask

        # grin loop or still (testing): the grin again, with the values the panel has now; not for the warm-up
        while (s.grin.loop or s.grin.still) and not self._silent and not no_grin:
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

    def _run_pass(self, workflow: dict, deadline: float, replay: bool = False, play: bool = True) -> tuple[np.ndarray | None, int]:
        """Queue the workflow and hand its grin to the player; returns, once the last frame has been shown,
        the depth of that frame and the frames made.
        A replay only makes the grin: ComfyUI reports the cached capture depth and portrait again.
        Without *play* the workflow makes no grin (skip_generation, skip_animation) and no clip is opened."""
        try:
            return self._receive_pass(workflow, deadline, replay, play)
        except BaseException:
            self.player.clear()     # after a failure or a reset nothing of the grin is left to play
            raise

    def _receive_pass(self, workflow: dict, deadline: float, replay: bool, play: bool) -> tuple[np.ndarray | None, int]:
        s: PipelineSettings = self.settings
        wf: Workflow = self.workflow
        total: int = wf.frame_count(workflow)
        if play and not self._silent:
            # the first grin after a portrait holds on its first frame (the blend); the warm-up opens no clip
            self.player.begin(total, hold=not replay)
        main_id: str = self.client.queue(workflow)

        made: int = 0
        last_depth: np.ndarray | None = None
        sampler_progress: float = 0.0
        last_event: str = 'queued'      # for the timeout message: where ComfyUI went silent
        last_time: float = time.time()
        step: str = 'queued'            # the same for the panel: the step that is running
        seen: bool = False              # an event of this workflow arrived
        silence_check: float = last_time + SILENT_AFTER

        def show(text: str) -> None:
            nonlocal step
            step = text
            if not replay:              # a replay keeps 'grin loop' / 'grin still'
                self._step(text)

        while True:
            self._check(deadline, f", no event for {time.time() - last_time:.0f} s after '{last_event}' during '{step}'")
            event: ComfyEvent | None = self.client.receive(0.05)
            if event is None or event.data.get('prompt_id') != main_id:
                # ComfyUI reports a workflow as soon as it starts: silence while it is no longer
                # in the queue means its events don't reach us
                if not seen and time.time() >= silence_check:
                    if not self.client.busy():
                        raise NoEvents(NO_EVENTS)
                    silence_check = time.time() + SILENT_RECHECK
                continue
            seen = True
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
                        self._capture_depth = depth
                        self._emit(GenMessage(GenState.CAM_DEPTH, 0.0, depth))
                elif node == wf.portrait and output.get('images'):
                    portrait: np.ndarray | None = self.client.view(output['images'][0])
                    if portrait is not None:
                        self._emit(GenMessage(GenState.DIFFUSION, 1.0, portrait, final=True))
                        self.player.hold_from_now()
                elif node == wf.description and output.get('text'):
                    self._style.description = str(output['text'][0])

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
