# The pose package's plan

What is not built in `modules/pose`. **(deduction)** marks what is inferred.

Questions:
- Each keypoint's uncertainty for the One Euro filter (noted 2026-11-18): RTMPose gives per keypoint a distribution
  over its SimCC bins, a narrow peak where the model is sure and a wide or double one where it is not; today only the
  peak's height reaches the pipeline, as the score (`modules/inference/pose/decode.py`). The distribution's width
  could set how strongly the One Euro filter smooths each point: harder where the model is unsure, lighter where it is
  sure. Cheap to compute next to the decode **(deduction)**; no paper tests it for human pose (the pose model's
  research, 2026-11-18, [INFERENCE.md](../inference/INFERENCE.md#the-models-and-their-files)). Built, it would be the width carried as a
  feature beside the score and the filter's cutoff scaled by it; whether it steadies the points more than the
  filter's own settings is to be measured

## The rules it touches

- `CLAUDE.md` Composition, a value derived from pose data is the pipeline's, and so are its settings: the width, and
  how much the filter follows it
