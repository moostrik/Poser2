# The board's plan

The board held up to the rules (2026-11-13, the apps' plan F). Its main point: three slices only one app reads
sit in `modules/board`, where `board.md` keeps only what a module or more than one app reads; each moves into its
app's board package, as white_space's own slices are already (`apps/white_space/board/`). The rest is form. One
step per commit, each checked by the tests named and every app's start (-sim, smile on its image input, 10 s).

## 1. The slices one app reads go to that app (built)

Who reads each slice, from the code: every other slice is read by a module layer (`modules/render/layers`), so it
stays.

| Slice           | Written by               | Read by                                   | Goes to                                  |
|-----------------|--------------------------|-------------------------------------------|------------------------------------------|
| captures        | deep_flow                | deep_flow's render                        | `apps/deep_flow/board/captures.py`       |
| groups          | white_space              | its state machine and pose instrument     | `apps/white_space/board/groups.py`       |
| sequencer state | hd_trio and white_space  | hd_trio's render                          | `apps/hd_trio/board/sequencer_state.py`  |

- deep_flow and hd_trio get a board package (`board/__init__.py`), and their `render_board.py` moves into it, as
  white_space's is (`board.md`: `apps/*/board/render_board.py` when the app has slices of its own).
- white_space stops writing the sequencer state to its board, which nothing of it reads (its state machine's state
  still goes to the sound sender); its `RenderBoard` drops the mixin.
- The tests move with what they test: `modules/board/tests/test_captures.py` to deep_flow's board, and the group
  store's two tests in `modules/pose/tests/test_group_record.py` to white_space's (the group record's own tests stay
  in pose).

Checked by: the three apps' tests, the board, pose and render tests, the guard tests, and each app's start.

## 2. Form (built)

- `HasObservations`' docstring says what the slice is; why the unfused view exists (a seam's two observations
  disagree when the geometry is wrong, and the fusion would hide it) moves to white_space's
  `docs/CALIBRATION.md`, Reading the panorama, where the calibration display is described.
- The time series store's test moves from `modules/utils/tests/test_time_series.py` to `modules/board/tests/`
  (the ring's own tests stay in utils).
- `camera_images.py` drops the quotes around its annotations, as the other files do under
  `from __future__ import annotations`.

Checked by: the board and utils tests and the guard tests.

- `submit_time_sample` is `set_time_sample`: a sample goes into a ring the plot reads at its tick, which the
  naming rule's `set` is (your call, 2026-11-14).

## Rules it touches

- `board.md`: a slice only one app reads lives in that app's board package: 1.
- `board.md`: an app with slices of its own keeps its `render_board.py` in its board package: 1.
- A docstring says what the thing is; reasons sit in the document: 2.
- Its tests sit in its `tests/` folder: 2.
- Name public data-flow methods by execution timing: open.
