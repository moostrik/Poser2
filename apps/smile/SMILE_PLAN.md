# smile's plan

What is next for smile, the app in [SMILE.md](SMILE.md); what spans the apps is in the
[refactor's plan](../../REFACTOR_PLAN.md).

## G. The OAK4 as a camera

- G4 The OAK4 live check with a visitor, on the connected OAK4 (now unturned): `-oak -lm` (landscape) and `-oak`
  (portrait, the frames turned): the face marks and the capture turned as the screen is, one visitor through each;
  at `P2160` and 30 fps the face finding and the upload keep up, no frames dropped; a recording at `P2160` and its
  `-sim`. The iGPU decoder stalled on a 3088 × 2316 recording in A3 (`VIDEO_PLAN.md` A1): if a `P2160` recording does
  not play with it, `-sim` plays it on the CPU decoder (the simulator's `decoder`) and the step says so.
  Done so far (G3): one `-oak -lm` start without a visitor, USB, 30 fps (lowest 27.6), 65 ms, 2704 × 2160, upright.

## Open questions

- Recordings of real visitors: a few recorded with the Vimba camera (or a webcam) at the PC or on site, the camera
  in portrait. The PC the plan was built on has neither; its one recording (`data/recordings/20261121-110540`, 1280 × 960) is a
  photo handed by a stand-in webcam, turned and flipped as the camera delivers.
- README.md's SMILE Online section keeps a **not yet** list: unbuilt work in a document, which `docs.md` puts in
  the plan. Move its items here as steps?
- `CaptionDraw` (`render/layers/caption_draw.py`) holds its font and title textures, where `rendering.md` has a
  `Draw` own no texture: a name that fits better, or the textures handed in?

## The rules it touches

- `CLAUDE.md` Running code, a run only on the user's go, what it measures and how long said first: G4
