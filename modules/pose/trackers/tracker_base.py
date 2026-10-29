"""Base class for all pose trackers."""

from abc import ABC, abstractmethod

from ..frame import FrameDictCallbackMixin

class TrackerBase(FrameDictCallbackMixin, ABC):
    """Base class for tracking multiple node instances.

    Provides callback system and defines common interface for all trackers.
    """