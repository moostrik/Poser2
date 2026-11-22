# HD Trio

Three color OAK-D cameras, one player per camera, a show of stages narrated from Max, and per screen the player's
centered figure, RVM's mask and the 2D fluid. Each player's screen also answers the other players, as far as they
move alike (the similarity).

## Input

hd_trio's input is white_space's: the cameras' frames of one moment travel as one set, synced by the frames'
`CameraSync`, uploaded as one batch by a `BatchUploader`, and handed to the frame source with exactly those images
([INPUT_THREADS.md](../../modules/flow/INPUT_THREADS.md)). The tracker is `OnePerCamTracker`: one player per
camera. The camera's thread only hands its frames and tracklets over; every receiver works on its own thread.

The crops are two, cut by the crop step (`CropExtractor`) in one pass from the same images and boxes: `pose_crop` at the pose model's input
size (192 × 256), one antialiased resample from the camera frame rather than a larger crop the runner scales down
again, and `segmentation_crop` at RVM's (768 × 1024), for the segmentation and the board. Both are 3:4, so they
cover one region, and the mask lines up with the pose's box (`SourceSpace.CROP`). Its motion is the 2D flow on the
centered mask.

The panel pins the drops where they happen (`cadence`): the two syncs', the uploader's, the pipeline's.

## Recordings

The recordings are research data: their video and their poses are in one image, the image as the show sees it,
mirrored. The recorder runs with `mirrored` on ([RECORDING.md](../../modules/recording/RECORDING.md#what-is-recorded)),
as mp4 chunks any player plays, so every hd_trio recording, old and new, is mirrored in the file and needs no flag
read to match its poses. Nothing is straightened (the cameras' tilt is 0).

The recordings hold no tracklets, so `-sim` replays them on the host with the person detector (`camera.replay`
and `yolo` on, the detector on TensorRT); the replay hands the frames on as stored.

## The show

The sequencer plays the stages of `ShowStage` in order, each for its duration; a stage's value is its audio cue,
sent on `/global/state`, so the Max patch plays `state<value>.wav`. IDLE is the resting stage: the sequencer parks
on the playlist's last entry when the show stops, so IDLE stays last. The panel's start, stop and skip are requests
the sequencer carries out at the render's next tick, so the show changes on the render's thread only. The show's
start and stop are shared with the session's, so the session's recorders run with the show.

Each stage composes the layers and sets the cross-camera response (`similarity_scale`): none while the player
learns the movement, rising in ENJOY, full in PLAY (`render_stages.py`). The density fades faster the more the
player moves (`FADE_TIME_STILL`, `FADE_TIME_RANGE` in `render/hd_trio_render.py`).

## The intro figure

A white example figure plays over WHITE_POSE to ENJOY_IN: a recorded track, the same on every screen, on the
render's tick. It is recorded on site: armed on the panel, the next show records `record_track` from `record_start`
to `record_end`. The window is a range in show order, so leaving it also ends a recording when the show is stopped or
skipped. The intro's recorder is outside the session's shared start and stop, so it never starts the show, and it is
never armed at start, so a settings file saved during a take does not record the first show.
