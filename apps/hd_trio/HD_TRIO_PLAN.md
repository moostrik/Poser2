# hd_trio's plan

What is next for hd_trio, the app in [HD_TRIO.md](HD_TRIO.md); what spans the apps is in the
[refactor's plan](../../REFACTOR_PLAN.md).

## A. The recorder

With the app's refactor ([REFACTOR_PLAN.md](../../REFACTOR_PLAN.md)), tested with the refactored app; how the recorder
works is in [RECORDING.md](../../modules/recording/RECORDING.md):

- A1. hd_trio: its wiring (the tracklets from each camera as they come, `expect_parts` per camera, `expect_poses` per
  camera and set), as white_space's is wired, and its pose `.h5` cut at the recorder's split.

## The rules it touches

- `CLAUDE.md` API evolution, update all affected call sites in the same change: A1
