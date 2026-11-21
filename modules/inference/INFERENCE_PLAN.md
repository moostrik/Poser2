# The inference package's plan

What is not built in `modules/inference`, apart from the threads the app refactor moves
([REFACTOR_PLAN.md](../../REFACTOR_PLAN.md)). What is built is in [INFERENCE.md](INFERENCE.md), its times and how
the refactor's steps were checked in [TIMING.md](TIMING.md). **(deduction)** marks what is inferred. From the package
scan of 2026-11-21.

## D. Code written once

- D1. One callback list: the uploaders, the crop step, the syncs and the workers each write their callback lists by
  hand, a lock, a list, register, copy under the lock, call outside it, a failing callback logged (serial_uploader.py
  25 such lines, batch_uploader.py 14, crop_extractor.py 11, the others 4 or 5) **(seen)**. One small class, in the
  package's `engine/` or beside the uploaders, holds it: `add`, `remove`, and a call that copies under its lock and
  calls outside it, logging a failing callback and going on (CLAUDE.md Concurrency); every list in the package uses it,
  the pose estimator and RecordedPoses keeping modules/pose's `FrameDictCallbackMixin`. Checked by pyright, the
  inference and apps' tests, and a `-sim` start of hd_trio and deep_flow
- D2. One NMS: `person/decode.py` has its own `nms`, `face/decode.py` calls `cv2.dnn.NMSBoxes` **(seen)**; the face
  uses the person's. Checked by the face test on its photo (`face_expected.json`)
- D3. Names: `engine/model_types.py` → `model_type.py`, after the one class it holds; `CropExtractor.submit_snapshot`
  → `set_snapshot`, as it stores the snapshot for a later lookup (CLAUDE.md, `set` stores for polling), the apps'
  registrations following. Checked by pyright and the tests
- D4. The tests' shared helpers (`wait_for` in five files, `uploaded` in two) **(seen)** written once in a helper module
  in the package's tests. Checked by the inference tests

## The rules it touches

- `CLAUDE.md` Packages, code that does the same is written once: D1, D2, D4
- `CLAUDE.md` Concurrency, a callback list has its own lock, a failing callback does not stop the others: D1
- `CLAUDE.md` Packages, files named after their class; Composition, `set` stores for polling: D3
