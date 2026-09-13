# White Space — Layers

The light layers behind the states' mixes (see `STATES.md` for the choreography; this
document covers the layers themselves). All show layers are **indexed** below. For
`beam_playhead`, `beam_flash` and `projection_playhead` the module docstrings are the source of
truth for behavior; `beam_blue_sound`, `flood`, `beam_wind_down` and `pose_instrument` keep their
full design sections here.

Modes: **beam** layers write the four beam lights by name (`Frame.beam_lights`, indexed by
`BeamLightId`: front/back white, left/right blue) and no pixels; **projection** layers draw the
persistence-of-vision projection image. The fixture's readout mode follows the *commanded* rpm — beam
mode below 200, projection mode at or above, switching on receipt of the rpm regardless of the
bar's actual speed (`loop` in `firmware.cpp` sets `SLOW` from `RPM`). The light sender maps the beam lights to the
firmware's pixel slots exactly when the fixture reads them (`inout/osc_light_sender.py`),
and the render simulates the beams in its own layer (`render/layers/
beam_light_simulation_layer.py`). Base classes `BeamLayer` / `ProjectionLayer` encode this, and the
folders and settings groups follow the same single axis (`layers/beam/` ↔ `light.beam_layers`,
`layers/projection/` ↔ `light.projection_layers` — no separate test folder). The file path names
the layer: the class is the folder and file in CamelCase and equals the `LayerId`
(`beam/flash.py` → `BeamFlash` → `beam_flash`; `projection/test_fill.py` → `TestFill` → `test_fill`;
`projection/` adds no prefix). Debug-only layers state that role in their docstrings; the
projection block's generic patterns keep a `test_` prefix.

## Inputs

Show layers and the state machine read pose frames from the board, never the tracker's tracklets.
A person is present while their pose exists: the pose pipeline stops posing a person
`pose.tracklets.detection_timeout` (1.0 s) after their last detection, and every filter downstream
resets a track the moment its pose is missing. A layer adds no presence test of its own. A person's
azimuth is the pose's `Azimuth`, moved to the eyes at LERP; their age is the pose's `Age`. Ghosts are
published to their own board store (`get_ghosts`), not among the poses. How the tracker produces the
poses is in `TRACKING.md`, *Downstream*.

## Index — show layers

| Layer                 | Mode       | Reads                         | Writes                                   | Used by                   |
|-----------------------|------------|-------------------------------|------------------------------------------|---------------------------|
| `beam_playhead`       | beam       | — (settings only)             | front white lamp                         | S1–S6, S9, S10            |
| `beam_flash`          | beam       | LERP frames (PlayheadOffset)  | front white lamp + blue lamps            | S4                        |
| `projection_playhead` | projection | frame playhead phase          | white playhead marker                    | S6 (projecting), S7, S8   |
| `pose_instrument`     | projection | LERP frames, playhead bars    | white lines, blue anchor + between-lines | S6 (projecting), S7, S8   |
| `flood`               | projection | — (settings only)             | full-strip white                         | S8                        |
| `beam_wind_down`      | beam       | tick clock                    | both white lamps, fading                 | S9, S10                   |
| `beam_blue_sound`     | beam       | sound levels from Max (board) | left/right blue lamps                    | S1, S2, S3, S5, S10       |

`beam_flash`'s blue lamps are zeroed in the presets: S4 runs blue-none by design. `pose_instrument`
reads `Azimuth`, `BBox`, `Angles`, `LegDeviation`, `TorsoTilt` and `Similarity` from the LERP frames,
and the playhead bars in PLAYHEAD motion only.

`beam_wind_down` is `flood`'s ending and a plain beam layer: the fixture is in beam mode from
S9's first packet, so the wall while the bar is still fast *is* the two white lamps
spinning. See its section below.

Debug layers (never in a state's mix; reached via the `light.debug` select — **choosing a
layer IS turning debug on**: it shows solo at full weight and the motor auto-follows its
mode, OFF returns the show): the two beam tools `beam_haunted` (the ghost flash;
pairs with `pose.ghoster.enabled` for solo experimentation) and `beam_test` (direct
levels for the four physical lamps: front/back white, left/right blue — beam mode's
hardware check), plus the projection `test_`-prefixed patterns: `test_pose_waves` (the old
wave/void instrument, kept as a reference/montage visual), `test_harmonic`,
`test_player_lines`, `test_calibration`, `test_fill`, `test_pulse`, `test_chase`,
`test_lines`, `test_random`.

---

## beam_blue_sound (BeamLayer)

The soundscape made visible: the left and right blue lamps breathe with the actual
sound Max is playing.

- **Used by**: S2 IDLE and S3 IDLE_INTRO at full; S1 OFF_IDLE, S5 INTRO_IDLE and S10 END_IDLE fading in
- **Input**: `/WS/sound/level` from Max — two floats (left, right), 0..1, real OSC —
  received on the **OSC sound receiver** (`inout.osc_sound_receiver`) and stored on the
  board (sound-level store: levels + received-timestamp). **Max must send this message**
  (coordination item).
- **Behavior**: left level → left blue lamp, right level → right blue lamp; a gain scales
  the mapping. **Latency first**: no softening — only a minimal smoothing window of at
  most 2–3 light frames (~66–100 ms), there purely to bridge OSC-arrival vs 30 Hz tick
  timing jitter, never to shape the response (Max shapes the envelope; the lamps follow).
- **Stale input**: when no message has arrived for `stale_seconds`, the layer falls back —
  either off, or a gentle idle pulse (tunable choice) — so a silent or disconnected Max
  never freezes the lamps at a stuck level.
- **Settings**: `gain`, `smoothing_frames` (0–3, default 2), `stale_seconds`,
  fallback mode/level
- **Reset**: clears the smoothing window (lamps re-attack from the live levels)

## flood (ProjectionLayer)

Constant full-strip white — the END's wall of light. Deliberately the dumbest layer in
the pool: all dynamics (the cross-to-full) are mix weights set by the states, never
behavior inside the layer. The *ending* of the wall belongs to `beam_wind_down` — S8's flood
at 1.0 hands over to S9/S10's wind_down starting at the full wall, seamlessly.

- **Used by**: S8 END (easing in as the instrument fades)
- **Input**: none
- **Behavior**: `white[:] += level`; stateless
- **Settings**: `level` (white; blue stays 0 — the flood is a white statement)
- **Reset**: no-op

## beam_wind_down (BeamLayer)

The dying wall of light: owns the S9/S10 ending fade. It writes the two white lamps at a
fading level; everything else is physics. The fixture is in beam mode from S9's first
packet (its readout mode follows the commanded rpm), so the two lamps spin at whatever
speed the bar still has — a wall of white while fast, thinning into two beams as it
slows — and the fade rides through both. One mechanism; the layer never needs to know
when the bar is slow. The S8 → S9 hand-off is seamless at the DACs: `flood` at 1.0 in projection
mode drives the same two white outputs as this layer at 1.0 in beam mode.

- **Used by**: S9/S10 at constant weight 1.0 (reset on state entry). The states put the
  landing look underneath (`beam_playhead` at DIM/BRIGHT, `beam_blue_sound`) — it is
  *revealed* as the wall dies, so nothing has to splice or match at the hand-off.
- **Input**: the tick clock; no pose data, no playhead signals.
- **Behavior**: both white lamps at `f × level`, `f = 1 − ease(elapsed / spin_down_seconds)`,
  hand-tuned to ride the physical spin-down (the sensor is silent above 200 rpm, so the
  deceleration is not measurable). The states exit once `progress` reaches 1 and the
  playhead lock holds (`STATES.md`).
- **Settings**: `level`, `spin_down_seconds` (hidden — the visible slider is
  `states.spin_down_seconds`, next to `spin_up_seconds`, shared in via the root),
  `progress` (read-only fade readout 0..1: the states' exit, `stage_progress` and S10's
  sound-visual reveal ride it)
- **Reset**: restarts the fade at the full wall (called from S9/S10 `enter()`)

## pose_instrument (ProjectionLayer)

The heart of the piece. Each person **stands in a blue anchor** — a blue line at their
azimuth, their own presence — and around them a **mirror-symmetric pattern of white and
blue lines derived from their pose**: the visual analogue of how the sound works, pose →
pattern as pose → sound. A neutral pose is "boring": one white line each side. Arms up is
the bass: many thick lines. Everything on the strip is a *line* (the 1-D image becomes
vertical lines in the room); the vocabulary is *anchor* for the blue at the person and
*lines* for the pattern — no "spot", "marker" or "centre line".

**The line world is anchored to the people, not to the strip.** The strip is divided into
segments between neighbouring participants; each segment fits a whole number of lines
(`n = round(gap / line_spacing)`), so its actual spacing is `gap / n` — a nudge of at most
half a spacing spread over the whole gap, invisible. Every person is a mirror point of
their own pattern, and the run of lines between two people is *the same lines* counted
from either side. This is the shared phase world: the phase is shared per segment and
anchored to the people, never per person (the old `pose_waves` drifted a phase per player,
which can never merge). A fixed global grid was considered and rejected — it cannot be
symmetric about a person who is not standing on it; syncing the patterns *in the space
between people* gives symmetry and the seamless join at once.

- **Used by**: S6 (once projecting), S7, S8
- **Input contract** (six pose parameters, all read into the per-participant state every
  tick whether or not the current mapping draws with them — the composition work happens
  on these): the four arm angles (`Angles`: left/right shoulder, left/right elbow),
  `LegDeviation` (joint-weighted hip/knee deviation, 0..1), `TorsoTilt` (signed sideways
  lean against the image vertical, −1..1 — the one absolute measure; every joint angle is
  segment-vs-segment); plus pose length (BBox height), presence (the pose itself, *Inputs*) and the
  pairwise `Similarity` row. `LegDeviation` and `TorsoTilt` are also sent to Max
  (`/pose/{id}/angle/legs`, `/pose/{id}/angle/tilt`) so sound and light read the same values.
- **Initial mapping** (a starting point to tune and rework — not the design's fixed part):
  `lift` (mean |shoulder| / π) → reach (`extent_min` → `extent_max`) and line thickness
  (`line_min` → `line_max`); `bend` (mean |elbow| / π) → density, crossfading the base lines
  toward harmonic `harmonics` (a subdivision of the same spacing, so joins still match);
  `legs` → colour balance (blue between-lines `blue_min` → `blue_max`, white dimmed by
  `legs_dim`); `tilt` read but unused by the first drawing; pose length → anchor width.
- **Between people**: line parameters (thickness, density, levels) are blended by position
  along a segment, so a thick pattern thins toward a neutral neighbour with no step at the
  midpoint. As people walk, a segment's line count steps at each half-spacing; a crossfade
  band (`n_blend`) slides the mid-gap lines instead of jumping them. Overlapping patterns
  are **MAX-blended** (identical lines → a seamless union; adding would double them).
- **Sync**: above `sync_threshold` (mean of both directions' similarity) the two patterns
  **grow toward each other along the shortest arc** — reach carried across intermediate
  people segment by segment — until they meet at the arc's midpoint. No arcs or fills: the
  gap is lined by the same lines both patterns are made of.
- **Line motion** (`line_motion`): `STATIC` (default), `CONSTANT` (`line_speed` spacings/s),
  or `PLAYHEAD` (`lines_per_bar` spacings per playhead bar, from the board's bars); plus
  `line_flow`: `SYMMETRIC` (outward from every person; the flows meet and pass through each
  other at segment midpoints) or `GLOBAL` (one way round the strip; lines approach a person
  on one side and depart on the other). Any motion breaks instantaneous symmetry — it holds
  exactly at phase 0 and ½ — so the moving modes are for evaluation on the machine: the
  important thing is that people recognise their own presence. Lines are born from the
  anchor (a whole-line gate over the first half spacing), so no line ever sits on the person.
- **Presence**: per participant attack (`attack_seconds`) and release (`release_seconds`:
  the last pose is held while fading; a fading person still bounds segments so neighbours'
  lines don't re-space at the moment of leaving).
- **Settings**: `line_spacing`, `line_motion`, `line_flow`, `line_speed`, `lines_per_bar`,
  `line_phase`, `n_blend`; `extent_min`/`extent_max`, `line_edge`; `line_min`/`line_max`,
  `line_soft`, `harmonics`; `level`, `legs_dim`, `blue_min`/`blue_max`; `anchor_width`,
  `anchor_level`; `sync_threshold`; `attack_seconds`, `release_seconds`
- **Reset**: forgets every participant (S6's entry — a fresh instrument per cycle); the
  line phase is a world property and keeps running
- **Relation to `pose_waves`**: the old wave/void instrument is **not** renamed or
  extended — it lives on as `test_pose_waves` (debug override), a reference/montage visual.

---

## Open

- **beam_blue_sound**: fallback choice (off vs idle pulse); exact `/WS/sound/level` scaling agreed
  with Max (linear 0..1 vs dB)
- **pose_instrument**: which line motion reads best on the machine (static join vs the moving modes'
  midpoint collision)
- **pose_instrument**: the mapping itself — the composition work
- **pose_instrument**: whether `tilt` should drive anything in the light
- **pose_instrument**: camera level — a pitched wide-FOV camera reads a small spurious tilt near the
  frame edges; check a straight person at the edge
- **Sync thresholds**: `states.sync.threshold` (per participant, INTRO → INTRO_PLAY) and
  `pose_instrument.sync_threshold` (pairwise, pattern growth) measure different quantities and are
  tuned separately; whether they should share one value
