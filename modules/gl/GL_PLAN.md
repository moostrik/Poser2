# GL plan

The window manager (`modules/gl/window_manager.py`) and its panel are rebuilt on Cinder's structure, on GLFW,
keeping to GLFW's threading rules: one kind of window, set by its own settings, on the main thread. Research and
sources: `window_systems_research.md`. Windows now, Linux kept possible.

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
