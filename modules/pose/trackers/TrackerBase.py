"""Base class for all pose trackers."""

from abc import ABC, abstractmethod

from modules.pose.callback import PoseDictCallbackMixin

class TrackerBase(PoseDictCallbackMixin, ABC):
    """Base class for tracking multiple node instances.

    Provides callback system and defines common interface for all trackers.
    """