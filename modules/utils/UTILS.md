# Utils

## Broadcast

A `Broadcast` is every callback list's type (CLAUDE.md, Concurrency). It is written `Broadcast[[X]]`, the argument
types in a list, the only form that also holds a callback without arguments (`Broadcast[[]]`).

- It copies its list under its lock and calls the callbacks outside it, so a callback may add or remove one, and a
  slow callback does not block a registration.
- It is an attribute of its producer, not a base class: a producer holds as many as it has outputs, and calling one
  works wherever a single callback is expected.
- `bool()` says whether any callback is registered, so a producer can skip building an item nobody hears.

## Waiting for a moment

On Windows a wait's timeout ends on the system's 15.6 ms tick (CLAUDE.md, Time). A `utils.Waker` waits on a
high-resolution waitable timer and an event together, so a wait ends at its deadline or at a wake. Every loop on a
deadline waits on one: the recording player, `ArtNetSender`, `OscSender`, the replay camera's
tracklets, smile's camera and clip cells, the Vimba camera's control loop, `InferenceFrameSync`, `RecordedPoses`.
white_space's light clock waits on one until 1 ms before its deadline and spins the rest, as LAYERS.md's
*The light clock* measures.

A 1 ms system tick (`timeBeginPeriod(1)`) is not used: Windows 11 drops it while the window is minimized. The
high-resolution timer (`CREATE_WAITABLE_TIMER_HIGH_RESOLUTION`, Windows 10 1803 on) is not tied to the system tick;
`time.sleep` uses it since Python 3.11, and on Linux a timed `threading.Event.wait` is already this precise
**(deduction, from the documentation)**.

200 waits each in a script, and deep_flow `-sim` on `20261112-125832_measure` (pose model on, recorded tracklets,
30 s measured after 10 s warm-up, 2026-11-20):

| What (ms)                                              | No tick             | 1 ms tick           | Waker               |
|--------------------------------------------------------|---------------------|---------------------|---------------------|
| a 1 ms wait, waited, mean / p95                        | 15.6 / 16.2         | 1.6 / 2.1           | 1.5 / 1.6           |
| the same, the window minimized                         |                     |                     | 1.6 / 1.8           |
| an 18 ms wait, waited, mean / p95                      | 31.1 / 32.0         | 18.5 / 19.0         | 18.4 / 18.7         |
| the same, the window minimized                         |                     |                     | 18.4 / 18.7         |
| a wake to waking, mean / p95                           | 0.02 (notify)       | 0.02 (notify)       | 0.09 / 0.23         |
| `time.sleep(0.001)`, mean / p95                        | 1.6 / 1.7           | 1.5 / 1.7           |                     |
| the player's frame gaps, mean / p95 / max              | 20.0 / 22.2 / 38.6  | 20.0 / 22.1 / 38.4  | 20.0 / 22.2 / 38.2  |
| the same, the window minimized                         |                     |                     | 20.0 / 21.9 / 30.4  |
| the replay tracklets, late, mean / p95 / max           | 7.9 / 14.5 / 16.7   | 0.7 / 1.5 / 2.6     | 0.4 / 0.9 / 3.0     |
| the same, the window minimized, mean / p95             | 7.7 / 14.5          | 7.9 / 14.7          | 0.4 / 1.0           |
| `Event.wait(0.001)` in the app, minimized, mean / p95  | 15.6 / 16.5         | 15.7 / 16.8         | 15.6 / 16.6         |
| the poses (RAW) after their capture, mean / p95        | 41.2 / 49.1         | 39.8 / 44.6         | 39.4 / 43.9         |
| the process's CPU, cores                               | 15.75               | 15.63               | 15.64               |

The Waker's wait to a wake ran in the script; the run's sync deadline was not reached: no frame waited for its
poses. Beside a thread busy in pure Python, a wake comes up to one tick late, the Waker's and `time.sleep`'s alike
(14.6 ms for a 1 ms wait): the woken thread asks for the GIL with a timed wait **(deduction)**.

Timers on a plain wait: the session's chunk timer (`chunk_start += chunk_length` keeps it from drifting;
a split is at most one tick late, every recorder together), white_space's motor sim (its falls are interpolated
from `perf_counter`, so a late wait does not move them), smile's start loop (a 20 ms poll for loaded clips, at
start-up only).
