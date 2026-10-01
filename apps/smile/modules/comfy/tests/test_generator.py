"""Tests for the generator's handling of a ComfyUI that runs a workflow but sends no events, and of a
failed warm-up (a fake client, no ComfyUI server needed)."""

import os
import unittest
from unittest import mock

from modules.comfy.client import Event
from modules.comfy.Generator import NO_EVENTS, SILENT_AFTER, Generator, NoEvents
from modules.comfy.settings import PipelineSettings
from modules.render.GrinPlayer import GrinPlayer
from modules.render.settings import SequencerSettings

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WORKFLOW = os.path.join(ROOT, 'data', 'comfy', 'smile_workflow.json')
WARMUP = os.path.join(ROOT, 'data', 'comfy', 'warmup.jpg')


class FakeClock:
    def __init__(self) -> None:
        self.now: float = 1000.0

    def time(self) -> float:
        return self.now


class FakeClient:
    """Queues anything; every receive() takes a second of the fake clock and hands out the next event."""

    def __init__(self, clock: FakeClock, events: list[Event | None], busy: bool) -> None:
        self.clock: FakeClock = clock
        self.events: list[Event | None] = events
        self.is_busy: bool = busy

    def queue(self, workflow: dict) -> str:
        return 'main'

    def receive(self, timeout: float) -> Event | None:
        self.clock.now += 1.0
        return self.events.pop(0) if self.events else None

    def busy(self) -> bool:
        return self.is_busy

    def interrupt(self) -> None:
        pass


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.settings = PipelineSettings()
        self.settings.connection.workflow = WORKFLOW
        self.generator = Generator(self.settings, GrinPlayer(SequencerSettings()))
        self.clock = FakeClock()
        patcher = mock.patch('modules.comfy.Generator.time', self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.workflow = self.generator.workflow.build('x.png', self.settings.style_a, self.settings)

    def receive(self, events: list[Event | None], busy: bool, timeout: float = 30.0):
        self.generator.client = FakeClient(self.clock, events, busy)
        return self.generator._receive_pass(self.workflow, self.clock.now + timeout, replay=False, play=False)

    def test_a_workflow_that_ran_without_events_is_no_events(self):
        with self.assertRaises(NoEvents):
            self.receive([], busy=False)
        self.assertLess(self.clock.now - 1000.0, SILENT_AFTER + 3.0)     # long before the time limit

    def test_events_of_another_workflow_do_not_count(self):
        with self.assertRaises(NoEvents):
            self.receive([Event('executing', {'prompt_id': 'other'})] * 3, busy=False)

    def test_a_workflow_still_in_the_queue_is_waited_for(self):
        with self.assertRaises(TimeoutError):
            self.receive([], busy=True)

    def test_a_workflow_that_reported_is_not_no_events(self):
        started = Event('execution_start', {'prompt_id': 'main'})
        done = Event('execution_success', {'prompt_id': 'main'})
        # silent for a long while after its first event, with an empty queue: still ours to wait for
        self.assertEqual(self.receive([started] + [None] * 20 + [done], busy=False), (None, 0))

    def test_no_events_makes_the_generator_not_ready(self):
        self.generator._warmed_up = True
        with mock.patch.object(self.generator, '_run_job', side_effect=NoEvents(NO_EVENTS)):
            self.assertEqual(self.generator._generate(None, 30.0), NO_EVENTS)
        self.assertFalse(self.generator._warmed_up)

    def test_a_failed_warm_up_is_not_ready(self):
        self.generator._stop_event.set()        # no wait before the retry
        with mock.patch('modules.comfy.Generator.WARMUP_IMAGE', WARMUP):
            with mock.patch.object(self.generator, '_generate', return_value='timed out'):
                self.generator._warm_up()
            self.assertFalse(self.generator._warmed_up)
            self.assertEqual(self.settings.connection.status, 'warm-up failed: timed out')
            with mock.patch.object(self.generator, '_generate', return_value=None):
                self.generator._warm_up()
            self.assertTrue(self.generator._warmed_up)
            self.assertEqual(self.settings.connection.status, 'ready')


if __name__ == '__main__':
    unittest.main()
