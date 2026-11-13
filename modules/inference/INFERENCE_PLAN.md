# The inference package's plan

What is not built in `modules/inference`, apart from the threads the app refactor moves
([apps/REFACTOR.md](../../apps/REFACTOR.md)). How the input works is in
[INPUT_THREADS.md](../flow/INPUT_THREADS.md). **(deduction)** marks what is inferred.

## 1. The color in one kernel

The uploader's color conversion, straightening and mirroring as one CUDA kernel, as the depth's alignment is
(`depth_align.py`): from the camera's NV12 23 GPU ops become 2, about 0.4 ms less GPU a frame **(deduction)**,
measured per step by `scripts/gpu_input_timing.py`. After the live run of
[the recording plan's checks](../recording/RECORDING_PLAN.md#1-checks-waiting-for-the-camera), which measures it on
the camera.

## 2. The previous crop

The optical flow on the crops (`enable_prev_crop`, off in deep_flow, so it runs nothing) does not wait for the
crops' event, and its pairing of the current and the previous image is to be checked when it is used.

## The rules it touches

- Treat latency as a first-order concern; choose a path by its measured latency: 1, measured on the camera.
- Fix a problem where it is made: 2, the optical flow waits for the event its crops carry.
