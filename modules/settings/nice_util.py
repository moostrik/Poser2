"""Shared NiceGUI helpers for the settings UI."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

from nicegui.elements.timer import Timer


class SafeTimer(Timer):
    """A ``ui.timer`` that stops itself when its parent slot is torn down.

    When a client disconnects or reloads, a pending timer coroutine can resume
    after its parent slot has already been deleted. The stock timer reads
    ``parent_slot`` while entering its run context and raises ``RuntimeError``
    from deep inside NiceGUI's event loop, surfacing as an unhandled
    background-task error that cannot be caught in the timer callback. Catching
    it at the context boundary lets the timer cancel gracefully — the client is
    gone, so there is nothing left to do.
    """

    def _get_context(self) -> AbstractContextManager:
        try:
            return super()._get_context()
        except RuntimeError:
            self.cancel()
            return nullcontext()
