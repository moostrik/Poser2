# White Space — Layers

The light layers behind the states' mixes (see `STATES.md` for the choreography; this
document covers the layers themselves). All show layers are **indexed** below. For
existing layers the module docstrings stay the source of truth for behavior; the **new**
layers (`sound_light`, `flood`, and the `pose_instrument` placeholder) get full design
sections here — they have no code yet, so this is their specification.

Regimes: **low** layers write the four bar lights by name (`Frame.bar_lights`, indexed by
`BarLightId`: front/back white, left/right blue) and no pixels; **high** layers draw the
persistence-of-vision ring. The fixture's readout mode follows the *commanded* rpm — slot
mode below 200, ring mode at or above, switching on receipt of the rpm regardless of the
bar's actual speed (`firmware.cpp` line 475). The light sender maps the bar lights to the
firmware's pixel slots exactly when the fixture reads them (`inout/osc_light_sender.py`),
and the render simulates the bar at low speed in its own layer (`render/layers/
bar_light_simulation_layer.py`). Base classes `LowLayer` / `HighLayer` encode this, and the
folders and settings groups follow the same single axis (`layers/low/` ↔ `light.low_layers`,
`layers/high/` ↔ `light.high_layers` — no separate test folder). Debug-only layers state
that role in their docstrings; the high block's generic patterns keep a `test_` prefix,
while the low tools are named as playhead tools (`playhead_haunted`, `playhead_test`).

## Index — show layers

| Layer             | Regime | Reads                                                  | Writes                        | Used by |
|-------------------|--------|--------------------------------------------------------|-------------------------------|---------|
| `playhead_low`    | low    | — (settings only)                                      | front white lamp              | S1–S5, S8, S9 |
| `playhead_flash`  | low    | LERP frames (PlayheadOffset, Dwell), tracklets         | front white lamp + blue lamps (blue zeroed in presets — S3 runs blue-none by design) | S3 |
| `playhead_high`   | high   | frame playhead phase                                   | white ring marker             | S5 (post-un-lock), S6, S7 |
| `pose_instrument` | high   | LERP frames (Azimuth, BBox, Angles, Similarity), tracklets | white bands + sync arcs, blue markers *(placeholder — see below)* | S5 (post-un-lock), S6, S7 |
| `flood`           | high   | — (settings only)                                      | full-strip white              | S7 |
| `wind_down`       | low    | tick clock                                             | both white lamps, fading (the wall while the bar is still fast) | S8, S9 |
| `sound_light`     | low    | sound levels from Max (board)                          | left/right blue lamps         | S1, S2, S4, S9 |

`wind_down` is `flood`'s ending and a plain low layer: the fixture is in slot mode from
S8's first packet, so the wall while the bar is still fast *is* the two white lamps
spinning. See its section below.

Debug layers (never in a state's mix; reached via the `light.debug` select — **choosing a
layer IS turning debug on**: it shows solo at full weight and the motor auto-follows its
regime, OFF returns the show): the two low tools `playhead_haunted` (the ghost flash;
pairs with `pose.ghoster.enabled` for solo experimentation) and `playhead_test` (direct
levels for the four physical lamps: front/back white, left/right blue — the lamp regime's
hardware check), plus the high `test_`-prefixed patterns: `test_pose_waves` (the old
wave/void instrument, kept as a reference/montage visual), `test_harmonic`,
`test_player_lines`, `test_calibration`, `test_fill`, `test_pulse`, `test_chase`,
`test_lines`, `test_random`.

---

## sound_light (new — LowLayer)

The soundscape made visible: the left and right blue lamps breathe with the actual
sound Max is playing.

- **Used by**: S1 IDLE and S2 IDLE_INTRO at full; S4 INTRO_IDLE and S9 END_IDLE fading in
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
- **Open questions**: fallback choice (off vs idle pulse); exact `/WS/sound/level`
  scaling agreed with Max (linear 0..1 vs dB)

## flood (new — HighLayer)

Constant full-strip white — the END's wall of light. Deliberately the dumbest layer in
the pool: all dynamics (the cross-to-full) are mix weights set by the states, never
behavior inside the layer. The *ending* of the wall belongs to `wind_down` — S7's flood
at 1.0 hands over to S8/S9's wind_down starting at the full wall, seamlessly.

- **Used by**: S7 END (easing in as the instrument fades)
- **Input**: none
- **Behavior**: `white[:] += level`; stateless
- **Settings**: `level` (white; blue stays 0 — the flood is a white statement)
- **Reset**: no-op
- **Open questions**: —

## wind_down (LowLayer)

The dying wall of light: owns the S8/S9 ending fade. It writes the two white lamps at a
fading level; everything else is physics. The fixture is in slot mode from S8's first
packet (its readout mode follows the commanded rpm), so the two lamps spin at whatever
speed the bar still has — a wall of white while fast, thinning into two beams as it
slows — and the fade rides through both. One mechanism; the layer never needs to know
when the bar is slow. The S7 → S8 hand-off is seamless at the DACs: `flood` at 1.0 in ring
mode drives the same two white outputs as this layer at 1.0 in slot mode.

- **Used by**: S8/S9 at constant weight 1.0 (reset on state entry). The states put the
  landing look underneath (`playhead_low` at DIM/BRIGHT, `sound_light`) — it is
  *revealed* as the wall dies, so nothing has to splice or match at the hand-off.
- **Input**: the tick clock; no pose data, no playhead signals.
- **Behavior**: both white lamps at `f × level`, `f = 1 − ease(elapsed / spin_down_seconds)`,
  hand-tuned to ride the physical spin-down (the sensor is silent above 200 rpm, so the
  deceleration is not measurable). The states exit once `progress` reaches 1 and the
  motor has locked at LOW.
- **Settings**: `level`, `spin_down_seconds` (hidden — the visible slider is
  `statemachine.spin_down_seconds`, next to `spin_up_seconds`, shared in via the root),
  `progress` (read-only fade readout 0..1: the states' exit, `stage_progress` and S9's
  sound-visual reveal ride it)
- **Reset**: restarts the fade at the full wall (called from S8/S9 `enter()`)
- **Open questions**: —

## pose_instrument (new — HighLayer, **placeholder**)

The instrument is the heart of the piece and the most complicated layer — it gets its
own major work package **later, once the rest of the system works**. What we build now
is a deliberate placeholder: a layer that already **receives the complete input
contract** the real instrument will need, and visualises each input in the simplest
legible way. The real instrument then grows inside this layer with all plumbing in place.

**Design direction for the real instrument** (the deferred work package):
- Each person **stands in a blue light** — their spot on the ring.
- Around them, a **line pattern derived from their pose** — white and blue — the
  visual analogue of how the sound works: pose → pattern as pose → sound.
- All patterns live in **one shared phase world**: line spacing and phase are anchored
  to a global reference, never per-person, so when two people's patterns fill the space
  between them they **match up seamlessly** — the sync fill is then not an overlay but
  matched patterns meeting and joining. (Note: the old `pose_waves` phases per player —
  `left/right_pattern_time` per person — which can never merge; the shared phase world
  is a foundational difference, not a refinement.)

- **Used by**: S5 (post-un-lock), S6, S7
- **Input contract** (all wired now, so the future instrument changes only the drawing):
  - per participant: azimuth strip position, pose length (BBox), the four arm angles,
    presence/age (tracklets)
  - per pair: pairwise `Similarity` (the sync fill's driver)
- **Placeholder visualisation**:
  - white: a simple band per participant at their azimuth (width from pose length,
    brightness modulated plainly by the arm angles — enough to see the data move)
  - white: a flat **arc between each similarity-matched pair** (similarity ≥
    `fill_threshold`, shorter arc, soft ends) — the simplest version of the sync fill
  - blue: a plain marker per participant (the "stands in a blue light" spot)
  - any striping/pattern the placeholder draws is anchored to the **shared phase world**
    from day one (a global grid, never per-person phase) — validating the foundational
    idea early, before the real instrument is built on it
- **Settings**: band width/level, `fill_threshold`, `fill_level`, `fill_edge`
- **Reset**: clears per-participant state (S5's entry reset covers it)
- **Relation to `pose_waves`**: the old wave/void instrument is **not** renamed or
  extended — it moves to the test layers (debug override) as a reference/montage visual;
  `pose_instrument` is a fresh file with the clean input surface.
- **Open questions**: the real instrument's full design (the deferred work package —
  see the design direction above); overlapping fill arcs add vs MAX-blend — decide when
  the real fill is built
