# The inference package's plan

What is not built in `modules/inference`, apart from the threads the app refactor moves
([apps/APPS_PLAN.md](../../apps/APPS_PLAN.md)). What is built is in [INFERENCE.md](INFERENCE.md), its times and how
the refactor's steps were checked in [TIMING.md](TIMING.md). **(deduction)** marks what is inferred.

## A. The TensorRT models side by side

Questions:
- CUDA graphs: capture each model's `execute_async_v3` per batch size and replay it, to cut the launch time on the
  CPU. Worth it, now the models run side by side?
- The pose's stream at a higher priority than the others', as the model on the critical path?

## B. The pose model

Questions:
- The licence of the Body8 training data (AI Challenger and others) is unresolved, for ours as for every candidate
- The workers log the engine's first input type as the model's precision, FP32 for the pose's FP16 models, whose
  inputs and outputs are FP32 **(seen)**: the weights' type, or the file's name, would say what runs
- Which pose resolution the presets run, `STANDARD` or `HIGH`, both FP16: yours to save
- Each keypoint's uncertainty for the One Euro filter: in the pose package's plan ([POSE_PLAN.md](../pose/POSE_PLAN.md))

## C. The results handed on finished (from superficial_flow's plan E, 2026-11-19)

The segmenter waits for its model, then queues the mask's rescale to the crop and the foreground (crop x mask),
records `ready` and hands them on at once (`segmentation/segmenter.py`, `_infer`) **(seen)**. A GL reader cannot wait
for a CUDA event on the GPU: its PBO copy's unmap makes the GL thread wait for that CUDA work
(`modules/gl/tensor.py`, `_update_with_pbo`) **(seen)**. superficial_flow's view waits on the mask so (`ViewMask.place`):
its GPU view takes 2.9 ms against deep_flow's 0.19, the motion, bridge and fluid queued behind it **(seen, the
superficial_flow thread)**; hd_trio's mask and foreground layers wait the same way **(deduction)**. The pose and the
optical flow wait for their stream before their callbacks **(seen)**; deep_flow's and white_space's images are written
when handed on (`image_upload.py` waits on its own thread) **(seen)**.

- C1. The segmenter hands on finished masks: its one `synchronize()` moves from right after the model to after the
  foreground, so its thread waits once for all its GPU work, as the pose's and the flow's do; `ready` stays, for the
  CUDA readers. `INFERENCE.md`'s "From the frame to the model" says the masks are written when handed on
- C2. The crops stay as they are, waited for on the GPU by the model workers: a wait in `CropExtractor` would hold the
  frame source's thread behind the upload's and the crop's GPU queue. Their GL reader, `CropImageLayer`
  (`modules/render/modules/inference/crop_image_layer.py`), takes a crop only once its `ready.query()` is true and
  keeps the one before until then, so the render never waits on the crop stream
- Checked by superficial_flow `-sim` on `20261112-125832_measure` as it is, 60 s before and after, about 3 minutes:
  the GPU view time and the masks' wait (`mask_wait`) and latency; hd_trio's tests; and by eye, hd_trio's
  segmentation layers and the crop layer still show

## The rules it touches

- `CLAUDE.md` Performance, prefer the design with the lower latency, measured in the context of the app: A, C
- `CLAUDE.md` Concurrency, the part that works on the item does all its work on its own thread: C1
