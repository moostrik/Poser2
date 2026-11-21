# Window

`modules/window` is the apps' window side: the window loop on the main thread, the windows, their input and the
monitors. It draws with `modules/gl` ([GL.md](../gl/GL.md)); `gl` never imports it.

## The loop

The window loop runs on the main thread: GLFW puts window creation, monitors and event processing there, and all its
callbacks run there. The launcher calls `RenderBase.run()` after `app.start()`; the loop draws on the same thread, so
nothing crosses a thread between the events and the draw. `stop()` works from any thread: it sets a close request,
GLFW's should-close and an empty event, and from another thread waits up to 2 s for the loop to end. The structure
is Cinder's: one update and a draw per window each frame, a context per window, the monitors as objects.

There is one kind of window (`Window`, `window.py`), set by its own `WindowSettings`: `enabled`, its `monitor`,
`fullscreen`, and its windowed place on that monitor (`x`, `y`, `width`, `height`). An app's root group `windows`
subclasses `WindowManagerSettings`: the loop's fields at the top of its panel, the frame rates first, then the
windows it declares, `window_0`, `window_1`, …, their number fixed at start. `v_sync_window` names one of the shown
windows (`shown_windows`, kept by the loop); a name the app does not declare falls back to `window_0`, logged.
Every window is made at start, window 0 first, the others sharing its objects. A window that is off is hidden, not
destroyed, and is neither drawn nor swapped. Fullscreen is a borderless window the size of its monitor, so the other
monitors stay on. A window whose monitor is missing waits small on the primary monitor with its settings untouched,
and returns when the monitor does. F toggles the fullscreen of the window with focus; Esc, or closing any window,
ends the app. The monitors are `Monitor`s (`monitor.py`) in `MonitorId` order: the primary first, then by x, then y.

The loop, each frame, in window 0's context:

| Step                | What it does                                                                          |
|---------------------|---------------------------------------------------------------------------------------|
| shaders             | the programs a changed file marked, compiled again (see [GL.md, Shaders](../gl/GL.md#shaders)) |
| monitors            | read again after a hot-plug (the monitor callback only marks it)                       |
| settings            | each window's settings read and only what differs applied; a user's move or resize     |
|                     | written to the settings once, as applied, and not applied back                         |
| sizes               | `on_window_resize(window, width, height)` for each framebuffer size that changed       |
| update              | the update callbacks with the frame's stamp, then `renderer.update()` once            |
| hold                | a wait on the fence of the paced swap before last: one frame in flight                 |
| draw                | each shown window: its context current, `draw(window, width, height)`                 |
| swap                | the shown windows, the `v_sync_window` last, with v-sync, and a fence after it         |
| pace                | the wait for `fps_limit`, when v-sync does not pace the loop                          |
| events              | `glfw.poll_events()`: the input callbacks                                              |

No callback calls GL: GLFW's callbacks store (a move, a size, a hot-plug) or hand over (`WindowInput`, which tells
each key, mouse and scroll callback its window, and logs a raising callback and runs the next). The renderer's
objects live in window 0's context: the loop makes it current at the top of each frame, around `allocate`,
`on_window_resize` and `deallocate`, and makes a window's own context current only for its draw and swap. A window's
size is its framebuffer's, known from `allocate` on (`RenderBase.window_size`). `default_placement` puts every window
at its default place on its own monitor, windowed; so does the start, for a window under half on the monitors.

## Pacing

With v-sync on and the `v_sync_window` shown, its swap paces the loop. Otherwise `fps_limit` does, in whole fps (0 is
off; 60 while every window is hidden): an absolute deadline on `perf_counter`, a period on from the last, waited with
the shared `utils.Waker`, a high-resolution waitable timer, without a spin. A change of `fps_limit` ends the wait at
once. A frame that ends after its deadline goes on at once;
one more than a period behind moves the deadline to now.

The Waker's lateness after its deadline, v-sync off (white_space -sim on 20261024-164009_ICK, the launcher's priority
and switch interval, 30 s from 10 s, 2026-11-20), ms:

| Where                         | Mean | p95  | p99  | Max  |
|-------------------------------|------|------|------|------|
| alone, 300 waits at 60 Hz     | 0.34 | 0.56 | 0.79 | 0.91 |
| in the app, `fps_limit` 60    | 0.98 | 3.72 | 5.23 | 6.82 |
| in the app, `fps_limit` 120   | 1.13 | 4.43 | 5.26 | 10.8 |

The timer is on time; inside the app the main thread waits for the GIL behind the app's busy Python threads after it
fires (**(deduction)**). A higher thread priority does not shorten it: inside one process the GIL, not the OS priority,
decides when a thread's Python runs. Less Python on the hot paths does. The frame time in those runs is the app's own:
17.7 ms mean at 60 fps (p95 29.8, p99 38.2), its update 9.8 ms mean (p95 26.1).

## One frame in flight

After the paced swap the loop puts a fence; before the next frame draws, it waits on the fence of the swap before
last (`_hold_frame_in_flight`), so one frame stays in flight and the driver queues no more. A GPU wait after the swap
(`glFinish`) holds the queue too, but stalls the CPU until the GPU is done, so the CPU's work and the GPU's no longer
overlap; it is not used.

v-sync on window 1 (fullscreen on a 3840 × 2160 monitor), deep_flow -sim on 20261112-125832_measure, 25 s from 10 s,
2026-11-20. "Unfinished" is the paced swaps the GPU has not passed at a frame's start, a fence polled once; ms:

| After the paced swap | Frame mean | p95  | p99  | Blocks in swap | Blocks in hold | Unfinished 0 / 1 / 2 |
|----------------------|------------|------|------|----------------|----------------|----------------------|
| nothing              | 16.9       | 30.5 | 36.2 | 0.41           |                | 9 / 1336 / 131       |
| `glFinish`           | 24.1       | 30.6 | 35.9 | 13.8           |                | 177 / 863 / 0        |
| the fence            | 16.9       | 17.6 | 33.3 | 8.65           | 0.02           | 191 / 1288 / 0       |

Measured: the fence holds the queue at one frame at 59 fps; `glFinish` holds it at 42 fps. That `glFinish`'s cost is the
lost overlap is a **(deduction)**. So is the time moving into the swap with the fence, and its p95 falling from 30.5 to
17.6 ms (one run each): the driver, with one frame queued, waits for the vblank in the swap.
