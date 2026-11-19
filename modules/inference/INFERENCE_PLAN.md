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

## The rules it touches

- `CLAUDE.md` Performance, prefer the design with the lower latency, measured in the context of the app: A
