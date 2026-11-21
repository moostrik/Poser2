"""Callbacks: a write's callbacks, the buttons that fire them, and writes from several threads."""

import threading
import unittest

from modules.settings import Field, Widget

from .fixtures import RenderMode, CameraSettings, MinimalSettings, SettingsWithActions


class TestCallbacks(unittest.TestCase):
    """Tests for the callback system."""

    def test_callback_fires_on_change(self):
        s = CameraSettings()
        results = []
        s.bind(CameraSettings.exposure, lambda v: results.append(v))
        s.exposure = 2000
        self.assertEqual(results, [2000])

    def test_callback_not_fired_on_same_value(self):
        s = CameraSettings()
        results = []
        s.bind(CameraSettings.exposure, lambda v: results.append(v))
        s.exposure = 1000  # same as default
        self.assertEqual(results, [])

    def test_multiple_callbacks(self):
        s = CameraSettings()
        a, b = [], []
        s.bind(CameraSettings.exposure, lambda v: a.append(v))
        s.bind(CameraSettings.exposure, lambda v: b.append(v))
        s.exposure = 3000
        self.assertEqual(a, [3000])
        self.assertEqual(b, [3000])

    def test_remove_callback(self):
        s = CameraSettings()
        results = []
        cb = lambda v: results.append(v)
        s.bind(CameraSettings.exposure, cb)
        s.unbind(CameraSettings.exposure, cb)
        s.exposure = 5000
        self.assertEqual(results, [])

    def test_bind_unknown_field_raises(self):
        s = CameraSettings()
        with self.assertRaises(KeyError):
            s.bind(MinimalSettings.value, lambda v: None)


class TestCallbackEdgeCases(unittest.TestCase):
    """Bind idempotency, unbind no-op, broken callback in _apply."""

    def test_bind_duplicate_prevention(self):
        s = CameraSettings()
        calls = []
        cb = lambda v: calls.append(v)
        s.bind(CameraSettings.exposure, cb)
        s.bind(CameraSettings.exposure, cb)  # duplicate — should be ignored
        s.exposure = 5000
        self.assertEqual(calls, [5000])  # exactly one call

    def test_unbind_silent_noop(self):
        s = CameraSettings()
        cb = lambda v: None
        # Never bound — should not raise
        s.unbind(CameraSettings.exposure, cb)

    def test_broken_callback_doesnt_crash_apply(self):
        """A callback that raises during field set should not crash."""
        s = CameraSettings()
        results = []
        s.bind(CameraSettings.exposure, lambda v: (_ for _ in ()).throw(RuntimeError("boom")))
        s.bind(CameraSettings.exposure, lambda v: results.append(v))
        s.exposure = 9999  # should not raise
        # The second callback should still have fired
        self.assertEqual(results, [9999])

    def test_unbind_all(self):
        s = CameraSettings()
        calls = []
        cb = lambda v: calls.append(v)
        s.bind_all(cb)
        s.exposure = 2000
        self.assertTrue(len(calls) > 0)
        calls.clear()
        s.unbind_all(cb)
        s.exposure = 3000
        self.assertEqual(calls, [])


class TestBindAllWithButtons(unittest.TestCase):
    """bind_all must work when Widget.button fields are present (#1)."""

    def test_bind_all_fires_for_buttons(self):
        s = SettingsWithActions()
        results = []
        s.bind_all(lambda v: results.append(v))
        SettingsWithActions.reset.fire(s)
        self.assertIn(True, results)

    def test_bind_all_fires_for_normal_fields(self):
        s = SettingsWithActions()
        results = []
        s.bind_all(lambda v: results.append(v))
        s.exposure = 5000
        self.assertIn(5000, results)


class TestAction(unittest.TestCase):
    """Tests for Widget.button Settings (replacing the old Action descriptor)."""

    def test_action_fires_callbacks(self):
        s = SettingsWithActions()
        results = []
        s.bind(SettingsWithActions.reset, lambda _: results.append("fired"))
        SettingsWithActions.reset.fire(s)
        self.assertEqual(results, ["fired"])

    def test_action_multiple_callbacks(self):
        s = SettingsWithActions()
        a, b = [], []
        s.bind(SettingsWithActions.reset, lambda _: a.append(1))
        s.bind(SettingsWithActions.reset, lambda _: b.append(2))
        SettingsWithActions.reset.fire(s)
        self.assertEqual(a, [1])
        self.assertEqual(b, [2])

    def test_action_no_callbacks_ok(self):
        s = SettingsWithActions()
        SettingsWithActions.reset.fire(s)  # should not raise

    def test_action_is_assignable(self):
        """Widget.button fields are normal Settings — assignment is allowed."""
        s = SettingsWithActions()
        s.reset = True  # should not raise
        self.assertEqual(s.reset, True)

    def test_bind_action_unknown_raises(self):
        s = SettingsWithActions()
        with self.assertRaises(KeyError):
            s.bind(MinimalSettings.value, lambda _: None)

    def test_unbind_action(self):
        s = SettingsWithActions()
        results = []
        cb = lambda _: results.append(1)
        s.bind(SettingsWithActions.reset, cb)
        s.unbind(SettingsWithActions.reset, cb)
        SettingsWithActions.reset.fire(s)
        self.assertEqual(results, [])

    def test_actions_property(self):
        s = SettingsWithActions()
        a = s.actions
        self.assertIsInstance(a, dict)
        self.assertIn("reset", a)
        self.assertIn("hidden_action", a)

    def test_actions_property_is_copy(self):
        s = SettingsWithActions()
        a = s.actions
        a["fake"] = None # type: ignore
        self.assertNotIn("fake", s.actions)

    def test_action_repr(self):
        r = repr(SettingsWithActions.reset)
        self.assertIn("Reset all values", r)

    def test_action_class_access_returns_descriptor(self):
        self.assertIsInstance(SettingsWithActions.reset, Field)

    def test_action_broken_callback_doesnt_crash(self):
        s = SettingsWithActions()
        results = []
        s.bind(SettingsWithActions.reset, lambda _: 1 / 0)  # will raise
        s.bind(SettingsWithActions.reset, lambda _: results.append("ok"))
        SettingsWithActions.reset.fire(s)  # should not raise
        self.assertEqual(results, ["ok"])


class TestThreadSafety(unittest.TestCase):
    """Tests for concurrent access from multiple threads."""

    def test_concurrent_writes(self):
        s = MinimalSettings()
        errors = []

        def writer(n):
            try:
                for i in range(500):
                    s.value = n * 1000 + i
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        # Value should be one of the written values (deterministic only in that it's an int)
        self.assertIsInstance(s.value, int)

    def test_concurrent_callbacks(self):
        s = MinimalSettings()
        counter = {"n": 0}
        lock = threading.Lock()

        def increment(v):
            with lock:
                counter["n"] += 1

        s.bind(MinimalSettings.value, increment)

        def writer(start):
            for i in range(100):
                s.value = start + i

        threads = [threading.Thread(target=writer, args=(t * 1000,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At least some callbacks should have fired (exact count depends on ordering)
        self.assertGreater(counter["n"], 0)


if __name__ == "__main__":
    unittest.main()
