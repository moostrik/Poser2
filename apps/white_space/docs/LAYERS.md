# White Space — Layers

The light layers behind the states' mixes (see `STATES.md` for the choreography; this
document covers the layers themselves). All show layers are **indexed** below. For
`beam_playhead`, `beam_flash` and `projection_playhead` the module docstrings are the source of
truth for behavior; `beam_blue_sound`, `flood` and `beam_wind_down` keep their full design sections
here; `pose_instrument`'s design (its voice, controls, patch and poses) is `POSE_INSTRUMENT.md`, and
its section here covers the layer alone.

Modes: **beam** layers write the four beam lights by name (`Frame.beam_lights`, indexed by
`BeamLightId`: front/back white, left/right blue) and no pixels; **projection** layers draw the
persistence-of-vision projection. The fixture's readout mode follows the *commanded* rpm — beam
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

Show layers and the state machine read pose frames from the board, never the tracker's tracklets. The
state machine also reads the board's hit streak (`HitSync`, `STATES.md` *Vocabulary*: in sync); the live
`Similarity` feature stays the instrument's (the window opening) and the sound's.
A person is present while their pose exists: the pose pipeline stops posing a person
`pose.tracklets.detection_timeout` (1.0 s) after their last detection, and every filter downstream
resets a track the moment its pose is missing. A layer adds no presence test of its own. A person's
azimuth is the pose's `Azimuth`, at their eyes: raw from CLEAN, smoothed at SMOOTH, predicted at
PREDICT and interpolated at LERP. The LERP stage runs inside the light tick at `light_rate`, one live root
field that the conductor's clock, the layers' playhead step (`Tick.interval`) and the LERP interpolators all
read. Show layers and the state machine read LERP poses, the stable eye azimuth and its `PlayheadOffset`. Their angles are the calibrator's (`POSE_INSTRUMENT.md`, *The
body*): 0 at neutral, π at raised. Their age is the pose's `Age`. Ghosts are published to their own
board store (`get_ghosts`), not among the poses. The dummy (`POSE_INSTRUMENT.md`, *The dummy*)
joins the poses before the LERP filters at its own id, `max_players`, so to every layer it is a
person; the ghosts' ids start above it. The render draws every LERP pose as a figure over the
projection row at its azimuth (`render.pose_figures`). How the tracker produces the poses is in
`TRACKING.md`, *Downstream*.

## Index — show layers

| Layer                 | Mode       | Reads                         | Writes                                   | Used by                   |
|-----------------------|------------|-------------------------------|------------------------------------------|---------------------------|
| `beam_playhead`       | beam       | — (settings only)             | front white lamp                         | S1–S6, S9, S10            |
| `beam_flash`          | beam       | LERP frames (PlayheadOffset)  | front white + blue lamps; board flashes  | S4                        |
| `projection_playhead` | projection | frame playhead phase; LERP frames (Azimuth) | white playhead marker, dim in a mask | S6 (projecting), S7, S8 |
| `pose_instrument`     | projection | LERP frames (PlayheadOffset)  | white and blue lines, dim blue masks     | S6 (projecting), S7, S8   |
| `flood`               | projection | — (settings only)             | whole projection white                   | S8                        |
| `beam_wind_down`      | beam       | tick clock                    | both white lamps, fading                 | S9, S10                   |
| `beam_blue_sound`     | beam       | sound levels from Max (board) | left/right blue lamps                    | S1, S2, S3, S5, S10       |

`beam_flash`'s blue lamps are zeroed in the presets: S4 runs blue-none by design. `pose_instrument`
reads `Azimuth`, `Angles`, `LegDeviation`, `TorsoTilt`, `AngleSymmetry`, `Similarity` and
`PlayheadOffset` from the LERP frames.

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

The whole projection constant white — the END's wall of light. Deliberately the dumbest layer in
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
  playhead lock holds (`STATES.md`). The fade advances only while the layer is drawn, so a
  debug solo pauses it.
- **Settings**: `level`, `spin_down_seconds` (hidden — the visible slider is
  `states.spin_down_seconds`, next to `spin_up_seconds`, shared in via the root),
  `progress` (read-only fade readout 0..1: the states' exit, `/global/state/progress` and S10's
  sound-visual reveal ride it)
- **Reset**: restarts the fade at the full wall (called from S9/S10 `enter()`)

## pose_instrument (ProjectionLayer)

The heart of the piece: each person stands in a dim blue **mask** at their azimuth, and around
them lies a mirror-symmetric pattern of full white and full blue **lines** drawn from their pose.
The design, the vocabulary, the meanings, the connections and the pattern, is `POSE_INSTRUMENT.md`;
this section is the layer: what it reads, how it composes people, and what it exposes.

- **Used by**: S6 (once projecting), S7, S8
- **Input**: per person from the LERP frames, the measures: the four arm angles (`Angles`:
  left/right shoulder, left/right elbow), `LegDeviation` (joint-weighted hip/knee deviation, 0..1),
  `TorsoTilt` (signed sideways lean against the image vertical, −1..1) and `AngleSymmetry` (signed
  left minus right per pair, −1..1); plus presence (the pose itself,
  *Inputs*), the pairwise `Similarity` row (`SIMILARITY.md`: the arms, raw, weighted at neutral; the window
  opens from `window.sync_threshold` up to 1 = identical arms) and `PlayheadOffset`. `AngleSymmetry`, `LegDeviation` and
  `TorsoTilt` are also sent to Max (`/pose/{id}/angle/sym`, `/pose/{id}/angle/legs`,
  `/pose/{id}/angle/tilt`) so sound and light read the same values. The layer adds no smoothing: the
  LERP poses are the pipeline's smoothed output.
- **Per person**: `PoseInstrument.connect` turns the measures into the pattern's parameters
  (`POSE_INSTRUMENT.md`, *Sources and connections*). The pattern is mirrored about the person's
  centre pixel, so it is symmetric exactly and moves with them as one piece; a neighbour walking
  never re-spaces it. On top of the pose each colour drifts by its own `drift`, white outward and
  blue inward. Every pixel is 0 or 1 per channel; only the mask is dim.

### Between people

- **Union**: overlapping patterns combine per channel (a pixel is lit when any pattern lights it).
  Two patterns of different intervals or centres make a moiré.
- **The visual limit**: `max_lines` per revolution, line and gap equal, so no line and no gap is
  narrower than half its period. Each person's pattern is made visible on its own
  (`LinePattern.visible`: gaps under the limit fill, then lines under it drop). A union of legal
  patterns can only add narrow gaps, so the union fills gaps under the limit. A line or gap appears
  and disappears at the limit's width, never thinner.
- **Masks**: every mask goes over every pattern, in both channels, and lights dim blue
  (`mask.brightness`, `mask.width` wide). A mask or a window edge cuts
  a line where it falls, so lines slide out from behind the mask and into view at the window edge.
- **Sync**: above `window.sync_threshold` (mean of both directions' similarity) a pair's window opens
  toward each other along the shorter arc, eased, until each pattern reaches the partner: full sync
  is full overlap, one pattern. It opens over any intermediate person, whose own pattern is
  unchanged. Sync shows more of the pattern; it never changes the lines. A pair with a person at neutral
  reads 0 and does not open (`STATES.md`, *Vocabulary*).

### Hit

On the ticks the playhead is closest to a person (`PlayheadCrossing` in `pose/playhead_offset.py`,
the same closest-tick rule as `beam_flash`, `events.hit_frames` of them), the person is marked: the
mask flashes to `mask.flash_brightness`; each colour's lines take the other colour by its tint
(`events.tint_white`, `events.tint_blue`: the central fraction of every line, 1 the swap); and the
push raises the drift by `events.push_strength`, settling back over `events.push_seconds` while
the lines keep what they gained (`POSE_INSTRUMENT.md`, *Events*). The crossing is measured in
playhead steps at `beam_rpm`, the rate the content playhead free-runs at in PROJECTION.

### Presence, tuning, reset

- **Presence**: per player attack (`presence.attack_seconds`: the window opens from the mask)
  and release (`presence.release_seconds`: the last pose is held while the window closes and the mask
  dims). A pose with a NaN azimuth has no place in the projection and counts as absent (it releases),
  the one exception to *Inputs*' no-presence-test rule.
- **Tuning**: the values are the root `PI` settings group (`POSE_INSTRUMENT.md`, *Settings*), live
  from the panel and saved in the preset; the connections are code. `connect`, the drawing math
  (`PoseInstrument`, `LinePattern`) hot-reload on save; adding a setting needs a restart.
- **Settings**: the layer's own group holds only `blend`; everything else is `PI`
- **Reset**: forgets every player and pass (S6's entry, a fresh instrument per cycle)
- **Relation to `pose_waves`**: the old wave/void instrument lives on as `test_pose_waves` (debug
  override), a reference/montage visual.

---

## Open

- **beam_blue_sound**: fallback choice (off vs idle pulse); exact `/WS/sound/level` scaling agreed
  with Max (linear 0..1 vs dB)
- **pose_instrument**: see `POSE_INSTRUMENT.md`, *Open*
- **Hit ownership**: `PlayheadCrossing` (`pose/playhead_offset.py`) runs in three instances (`beam_flash`,
  `pose_instrument`, `HitSync`) on the same LERP `PlayheadOffset` and the same step (`beam_rpm` at the
  tick rate), so they agree by construction. One light-side detector publishing per-width hit sets on the
  board would give the hit one owner; left as is because the layers gain nothing by it
- **Ghosts and the playhead boundary**: the Ghoster is deprecated (off in `studio.json`; a nice-to-have).
  It is what keeps `PlayheadOffset` a pose feature (its beat reads it, it stamps it on ghosts) and is the
  pose package's one playhead dependency. Removing the Ghoster removes `beam_haunted`, `GhostFeature`,
  `GhostState`, the ghost store, the sound's ghost slots and the `playhead_data` dropdown with it; the
  playhead offset (azimuth − playhead, stateless) can then be computed by its consumers — the flash, the
  instrument, `HitSync`, the sound sender when its frames arrive — and leave the frame, and the pose
  pipeline takes the light's clock (the LERP stage runs inside the light tick) and none of its data
