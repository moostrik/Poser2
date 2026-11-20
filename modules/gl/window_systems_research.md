# Window systems: what to follow

Question: which existing system should `modules/gl`'s window manager follow? Ours is one render thread that
opens a main window and secondary windows on chosen monitors, shares one GL context's objects between them,
calls the renderer's `update` once per frame and draws each window, and runs GLFW's events.

Read from source (Nov 2026, master branches): Cinder (`AppImplMswBasic.cpp`, `AppImplMsw.cpp`, `Window.cpp`),
openFrameworks (`ofMainLoop.cpp`, `ofAppGLFWWindow.cpp`, `ofTimer.cpp`, `ofEvents.cpp`), moderngl-window
(`__init__.py`, `context/glfw/window.py`), GLFW docs (`intro.md`, `window.md`). SDL3 and Godot are from
knowledge, not checked.

## The candidates

| System          | Fit  | Why                                                                                   |
|-----------------|------|---------------------------------------------------------------------------------------|
| Cinder          | best | C++ creative coding, multiple windows on monitors, our `gl.Context` already copies it |
| openFrameworks  | good | same family, built on GLFW like us: its GLFW details are directly usable              |
| GLFW docs       | rules| not a framework; sets the threading and context rules any design must keep            |
| moderngl-window | low  | Python and GLFW, but one window, no monitors; resize is a direct callback             |
| SDL3, Godot     | idea | resize and input arrive as events in a queue the loop reads (not checked)             |
| TouchDesigner   | none | closed source; perform mode is the idea, the internals can't be read                  |

## How Cinder does it

- Parts: `App` (setup, update, draw, resize), `Window` (one OS window), `Renderer` per window (its GL context,
  shared with the first), `Display` (a monitor, as an object, with its signals).
- Loop: refresh displays if a monitor changed, `update` once, `draw` each window, then sleep until the next frame
  while handling OS messages (`AppImplMswBasic::run`).
- Every window event (draw, resize, mouse, key, file drop) first makes that window's context current
  (`Window::emitResize` → `applyCurrentContext()`), so the app never runs in the wrong context.
- At start every window gets one resize, so the app learns its size the same way as later.
- A monitor change only sets a flag (`mNeedsToRefreshDisplays`); the loop handles it at the top of the next frame.
- Frame-rate cap: an absolute next-frame time on the app clock, a Windows waitable timer for the wait, no spin;
  more than 1 s behind skips ahead.

## How openFrameworks does it

- Parts: `ofMainLoop` (the loop), `ofAppGLFWWindow` (one GLFW window), a renderer per window, an app per window;
  `shareContextWith` in the window settings shares objects.
- Loop: for each window make current, update, draw (swap inside draw); then `glfwPollEvents` once.
- Resize: GLFW's callback goes straight to the app's `windowResized` (as ours does now).
- Frame-rate cap: `ofTimerFps` waits before draw; on Windows `ofTimer` uses a waitable timer at an absolute time.
- Fullscreen: picks a monitor by index; can span several monitors (`multiMonitorFullScreen`).

## The rules from GLFW's docs

- Windows, contexts and event processing belong to the main thread, and all callbacks run there.
- A context is current on one thread at a time.
- Windows can share a context's objects; FBOs and vertex arrays are never shared (OpenGL itself).

## What this means for ours

- Our shape already matches Cinder and oF: one update per frame, a draw per window, `gl.Context` per window.
- Gap 1: we call the renderer from GLFW callbacks in whatever context is current. Cinder makes the window's context
  current first; the deferred-resize fix does this too.
- Gap 2: we create windows and run events on a render thread, not the main thread. That works on Windows, where
  each thread has its own message queue, but it breaks GLFW's rule (deduction: fine on Windows, not portable).
- Gap 3: monitors are indices kept in sync by hand; Cinder has a `Display` object with change signals.
- Gap 4: the frame-rate cap. Cinder and oF both use an absolute deadline and an OS timer, with no spin.
- Gap 5 (deduction): while a window is dragged or resized, Windows runs its own message loop inside
  `poll_events`, so our loop stops drawing until the drag ends. Cinder tracks this (`mInSizeMoveLoop`).

## Checked by the deep research (25 claims, each confirmed 3-0 or 2-1)

- GLFW: init, window and context creation, monitor queries and `poll_events` belong to the main thread, and all
  callbacks run there. Our render thread breaks that contract.
  ([intro](https://www.glfw.org/docs/3.3/intro_guide.html), [monitor](https://www.glfw.org/docs/3.3/group__monitor.html))
- GLFW: `make_context_current`, `swap_buffers` and `swap_interval` may run on any thread. The supported split is:
  the main thread owns windows, monitors and events, and a render thread makes contexts current, draws and swaps.
  Sizes and input then cross to the render thread under a lock.
- One context per window, objects shared by a share window. FBOs and VAOs are never shared (GL spec).
- pyglet makes the window's context current before it dispatches draw and resize. The framework does the switch,
  not user code (as Cinder does).
- Resize and input callbacks fire inside `poll_events`. On Windows the drag/resize loop runs inside that call and
  blocks our loop (2-1 vote). SDL3 keeps drawing with a 10 ms Win32 timer during the drag
  ([SDL](https://discourse.libsdl.org/t/sdl-allow-the-application-to-draw-while-windows-is-in-a-modal-move-resize-loop/47255)).
  With GLFW, the equivalent is to draw from the refresh callback while dragging.
- Monitor hot-plug arrives through `set_monitor_callback`, during event processing.
- pyglet caps the frame rate by sleeping until the next scheduled time on `perf_counter`, with no spin.

Not covered by verified sources: Cinder (its claims didn't pass, but I read its source above), Godot, Magnum, bgfx,
VisPy, the ImGui backends, how precise sleep is on Windows, and borderless versus exclusive fullscreen.

## Recommendation

Follow Cinder for the structure (App, Window, Renderer/context, Display, loop) and the event rule (context current
before every event). Take GLFW details (fullscreen, monitors, shared contexts) from openFrameworks, since both use
GLFW. GLFW's threading rule is an open question: first find out why the windows run on a render thread and not the
main thread. Linux is a possible later target, which makes keeping to the rule worth more.

Frame cap: an absolute deadline on `perf_counter` (next += period, resync when a period behind) and `time.sleep`
until it. Since Python 3.11, `time.sleep` on Windows uses a high-resolution waitable timer (from memory, not
checked here), so a spin may not be needed. Measure the jitter on the PC before adding one.
