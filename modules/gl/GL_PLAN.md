# GL plan

The window manager (`modules/gl/window_manager.py`) and its panel are rebuilt on Cinder's structure, on GLFW,
keeping to GLFW's threading rules: one kind of window, set by its own settings, on the main thread. Research and
sources: `window_systems_research.md`. Windows now, Linux kept possible.

## A. GLFW on the main thread

GLFW puts window creation, monitors and events on the main thread; ours runs them on a render thread, which works
on Windows only. The launcher's main thread only waits, so it takes the loop, the drawing included (no locks, no
hand-over), as in Cinder, openFrameworks and pyglet.

- A1. `RenderBase.run()` runs the window loop on the calling thread until the windows close. The launcher calls it
  after `app.start()`, then waits for `app.is_finished` as now; Ctrl-C only calls `render.stop()`. The apps no
  longer start the render; the render thread goes. `stop()` works from any thread: a close request (`Event`),
  should-close and an empty event (GLFW allows both from any thread); from another thread it waits up to 2 s for the
  loop to end. Test: `stop()` from another thread ends the loop (glfw mocked). Run: one app, started windowed and
  fullscreen: resize, F, Ctrl-C, Esc, the timing session's stop.

## B. One kind of window

- B1. `Monitor` (`monitor.py`): a connected monitor (name, position, size, handle); the monitors in `MonitorId`
  order (primary first, then by position); a hot-plug is handled at the top of the next frame. Test: the order, a
  monitor without a video mode skipped.
- B2. The settings, the renderer and the window manager, in one step and one commit (the half-way state, the old
  manager on the new settings, is never used):
  - `WindowSettings`, one window: `enabled` (shown), `monitor` (the one it is on), `fullscreen`, `x`, `y`,
    `width`, `height` (the windowed place on that monitor).
  - `WindowManagerSettings`, the loop: `v_sync`, `v_sync_window` (default 0), `fps_limit`, `fullscreen_mouse`,
    `default_placement` (button), `avg_fps`, `min_fps`, `connected` (read: the monitors by id, name, resolution).
  - Each app's root group `windows` holds `loop` and `window_0`, `window_1`, … (as `cam_0`, `cam_1`); the render
    panel's `window` goes, in all five apps (built before the settings plan, which then leaves `windows` as it is).
  - The windows per app: hd_trio windows 1–3 its cameras; smile windows 1–2 its walls, their count fixed at start;
    deep_flow, low_flow, white_space window 0; deep_flow also a `window_1`, off, for its show.
  - The renderer: `RenderBase` takes the `windows` group; `draw(window, width, height)` replaces `draw_main` and
    `draw_secondary`. The window sizes are known in `allocate`; `on_window_resize(window, width, height)` comes
    only on a change. The sizes guessed in `__init__` (deep_flow, low_flow, white_space, hd_trio) go.
  - `FullscreenMode`, `secondary_list`, `secondary_fullscreen` go: `gl/__init__.py`, the five apps' settings and
    renderers, `smile/main.py`, `smile_render.py`, tests.
  - The 13 presets migrate, as whole trees: `secondary_list` entries become windows'
    monitors, at today's windowed place (half the monitor, at +100).
  - `Window` (`window.py`): one GLFW window with its `gl.Context` and settings. Every declared window is made at
    start, window 0 first, the others sharing its objects. A window that is off is hidden, not destroyed, and not
    drawn or swapped. A window whose monitor is missing waits small on the primary monitor, its settings untouched.
  - Fullscreen is a borderless window the size of its monitor; the other monitors stay on. Closing any window
    closes the app; F toggles the fullscreen of the window with focus; Esc closes.
  - No callback calls GL: callbacks store or hand over; the loop makes a window's context current for its draw.
  - `WindowInput` (`window_input.py`): the key, mouse and scroll callbacks, each told the window. A raising
    callback is logged and the next one runs.
  - The settings are read at the tick and only differences applied; no binds, no deferred queue (the button stays
    a bind). A user's move or resize is written to the settings once at the tick, as applied.
  - `WindowManager` keeps the loop: monitors, settings, update once, draw each shown window, swap the others, then
    the `v_sync_window` last with v-sync, then events. `_contexts`, `_secondary_fallback`, `_main_window`, the
    index lookups and the two placement paths go.
  - Tests: a raising input callback; a setting change applied once; a drag written and not applied back.

## C. Frame pacing

- C1. `fps_limit` (0 = off) when v-sync is off or every window is hidden (then 60 when it is 0): an absolute
  deadline on `perf_counter`, waited with the shared `modules.utils.Waker` (a high-resolution waitable timer);
  no busy-wait.
- C2. Run, 60 and 120 fps, v-sync off: frame-time jitter. A miss over 1 ms is fixed in `Waker` itself, with the
  thread that owns it.
- C3. Run, v-sync on a show window: frame time and the frames the driver queues, with and without a GPU wait
  (`glFinish`) after the paced swap. The wait stays if it keeps the queue at one frame.

## D. Check and document

- D1. Run each app on the PC: start windowed and fullscreen, the operator window off and on, no tearing on the
  windows without v-sync, fullscreen with the other monitors staying on, resize, drag onto another monitor and save, drag and release, F, the panel moving a
  window, every window of hd_trio and smile, a monitor unplugged and back. hd_trio's show looks as before (its
  window layout may change).
- D2. `GL.md`: a section on the windows: one kind of window, hidden not destroyed, the main thread, no GL in
  callbacks, which context is current when, the size, the placement, the settings at the tick, the pacing and the
  `v_sync_window` swapped last. gl.md's thread rule says the main thread. The window manager tests move to the new
  parts.

## Rules this plan touches

- gl.md: All OpenGL calls execute on the render thread
- gl.md: Create textures, FBOs and shaders at setup or on a resize, not per frame
- app-lifecycle.md: Start workers/services only after wiring is complete
- settings-presets.md: Do not subclass a module's Settings class in app code; use `Group`
- settings-presets.md: For renamed fields, rename the key in each JSON, preserving the value
- panels.md: A panel that holds more than one chain (an app's root) may order its groups by use
- glossary.md: An on/off switch: `enabled`
- CLAUDE.md: Pass collaborators and fixed structure (counts known at startup) through constructors
- CLAUDE.md: A callback only hands the item over and returns
- CLAUDE.md: A part that only needs the latest value, like the render, reads it at its own tick
- CLAUDE.md: A thread's loop catches an exception per item, logs it and goes on with the next
- CLAUDE.md: A failing callback does not stop the others
- CLAUDE.md: Read `time.perf_counter()` for every stamp and duration
- CLAUDE.md: A thread that waits for a moment waits on a `utils.Waker`; a wait's timeout (`Event.wait(t)`, `Queue.get(timeout=t)`) ends on Windows' 15.6 ms tick
- CLAUDE.md: Its files each hold one class or one subject, named after it
- CLAUDE.md: Keep presets and settings schemas in sync with code changes
- CLAUDE.md: A commit leaves no pyright errors
- CLAUDE.md: Remove obsolete pathways instead of leaving parallel legacy ones
