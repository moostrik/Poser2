import logging
import time
from abc import ABC, abstractmethod
from threading import Event, Lock, Thread
from typing import Generic, Protocol, TypeVar, cast

import torch

from .model_worker_settings import ModelWorkerSettings

logger = logging.getLogger(__name__)

Item = TypeVar('Item')


class _Dropped(Protocol):
    """The monitor each worker's settings declare last, beside the base's fields, as read on an instance."""
    dropped: int        # items replaced by a newer one before the model ran them


Settings = TypeVar('Settings', bound=ModelWorkerSettings)       # with a `dropped` field


class ModelWorker(ABC, Generic[Item, Settings]):
    """A model on its own thread and CUDA stream, which runs the newest item waiting per key.

    `_put` queues an item under a key, replacing one still waiting there, which is counted in `settings.dropped`: one
    key keeps a single slot (a batch of players), one per camera keeps each camera's newest image. The thread sets the
    model up (`_setup`, on its stream), then wakes on work, takes every waiting item and runs them (`_process`),
    catching an error per round. Each worker keeps its model's steps and its callbacks.
    """

    def __init__(self, settings: Settings, name: str) -> None:
        self.settings: Settings = settings
        self.name: str = name
        self._stream: torch.cuda.Stream | None = None      # made on the thread that runs the model, in `_prepare`

        self._thread: Thread = Thread(target=self._run, daemon=True, name=name)
        self._shutdown: Event = Event()
        self._wake: Event = Event()
        self._ready: Event = Event()

        self._lock: Lock = Lock()
        self._pending: dict[int, tuple[Item, float]] = {}      # per key: the newest item and when it was queued

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and not self._shutdown.is_set() and self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._shutdown.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            logger.warning(f"{self.name}: thread did not stop cleanly")

    def reset(self) -> None:
        """The waiting items dropped."""
        with self._lock:
            self._pending = {}

    def _put(self, key: int, item: Item) -> None:
        """An item to run, replacing the one still waiting under `key`."""
        now: float = time.perf_counter()
        with self._lock:
            replaced: tuple[Item, float] | None = self._pending.get(key)
            self._pending[key] = (item, now)
        if replaced is not None:
            cast(_Dropped, self.settings).dropped += 1     # a Field reads as its value
            if self.settings.verbose:
                logger.info(f"{self.name}: dropped an item with lag {int((now - replaced[1]) * 1000)} ms")
        self._wake.set()

    def _prepare(self) -> None:
        """The stream made and the model set up on it, on the calling thread."""
        self._stream = torch.cuda.Stream()
        self._setup()

    def _run(self) -> None:
        try:
            self._prepare()
        except Exception as e:
            logger.exception(f"{self.name}: failed to load its model: {e}")
            return
        self._ready.set()
        while not self._shutdown.is_set():
            self._wake.wait()
            if self._shutdown.is_set():
                break
            self._wake.clear()
            with self._lock:
                pending: dict[int, tuple[Item, float]] = self._pending
                self._pending = {}
            if not pending:
                continue
            try:
                self._process({key: item for key, (item, _) in pending.items()})
            except Exception:
                logger.exception(f"{self.name}: error running the model")

    @abstractmethod
    def _setup(self) -> None:
        """Load the model on `_stream` and warm it up, so no kernel is made while running."""

    @abstractmethod
    def _process(self, items: dict[int, Item]) -> None:
        """Run the model on the waiting items, by key, and hand the results on."""
