# White Space — Pose Instrument

The instrument the participants play with their bodies: the light half of pose → sound. Four
parts. **Vocabulary** is the language of the instrument, borrowed from the synthesizer. **Meaning**
is what each measure of the body, alone and with the others, should say in it. **Connections** is
how the measures are wired to the pattern so that the meaning comes out. **Implementation** is how the layer draws it
today. Meaning and connections are one loop: meaning sets the targets, the targets pick the
connections, the connections have consequences, and the consequences are new meaning to judge.
While one is held still the other is worked on.

The layer's place among the other layers is in `LAYERS.md`; the states that run it are in
`STATES.md`.

## The instrument

Each person is a source of light in the projection. At their azimuth sits a dim blue **mask**: it
goes over everything at the person, the patterns of everyone and the light of the other layers;
the playhead passes through it dimmed by a setting. Around the mask, mirrored about the person,
their **pattern**: lines of full white and full blue, generated behind the mask and coming out from
under it. Only a **window** of the pattern is visible each side of the person; sync opens the
window wider.

The ground rules, as set:

- About 90 lines per revolution, line and gap equal, is the visual maximum **(site fact)**. It is a
  guide, not a wall: a good reason may go past it.
- No brightness: every pixel of white and of blue is off or full. The mask is the one dim
  thing.
- No blinding: the instrument never blinds a person with white; people must be able to see each
  other. The mask is how: it goes over everything at the person, so full white can fill a window
  and never fall on a face. This rule is the instrument's alone, not the other layers'.
- White and blue are projected separately by the fixture **(site fact)**, so where both are on the
  overlap reads as a subtly different tone. The palette is four tones: dark, blue, white, both.
- No jitter: the layer adds no flicker of its own.
- Lines follow the pose. On top of that they drift by themselves, slowly and peacefully, blue
  inward and white outward: a low setting that may be 0 (see *Drift*).
- The output each side of a person is symmetric.
- The pattern is LFO based; the six pose values control it; every combination of the six draws
  differently; the arms come first, legs and body bend second.
- Arms down is neutral: full blue over the window. Arms up: full white. In between, the pattern.
- Blue and white can behave differently; line thickness and interval can change.
- A hit (the playhead crossing the person) marks that person for one frame. Wider lines do not
  read **(site fact)**; the mark is open, see *The hit*.
- Sync makes more of the pattern visible: the window opens, the lines stay what they are.

---

# Part 1 — Vocabulary

## The oscillators

The pattern is two low-frequency oscillators, one for white and one for blue. They are spatial:
they oscillate over the distance from the person, not over time. The vocabulary is the LFO's.

The projection is an oscilloscope of one dimension. It draws sine waves as the old green screens
did, but it has no height to draw them with, so it draws where the wave rises above a level: a bar
for every crest. The thickness of a bar is how long the wave stays above the level; the interval is
the wavelength.

| Synth term     | In the instrument                                                         |
|----------------|---------------------------------------------------------------------------|
| waveform       | sine, triangle or saw, thresholded into bars (see below)                  |
| interval       | the distance between lines, in degrees: the LFO's rate, spatial           |
| detune         | blue's interval leaving white's: 0 locked to one grid, more and they beat |
| pulse width    | the thickness of a line as a fraction of the interval: 0 none, 1 solid    |
| phase          | where the lines sit relative to the mask                                  |
| cutoff         | which overtones pass: none, or up to 2, 3 or 4 per interval               |
| resonance      | how strongly the passing overtone is emphasised: sub-lines, uneven lines  |
| overtone phase | where the overtone sits within the interval                               |
| drift          | own motion: the phase advancing over time, blue inward, white outward     |

A **filter** shapes the wave before the level: the cutoff says which overtones pass, the resonance
how strongly the passing overtone speaks. Open, the bars split into sub-lines or grow uneven;
closed, there is a plain bar per crest. In code the filter is additive, one overtone at the cutoff
mixed in by the resonance; to the eye it is the same. An overtone is not a second oscillator: it
sits at a whole number of cycles per interval and can never leave its fundamental, so the pattern
still repeats every interval. Only detune lets one wave leave another. The **waveform** decides how thickness
answers the level: a sine or a triangle grows a bar from its centre both ways, the triangle
linearly; a saw grows it from one edge only, toward the person or away from them.

**Registration** is the organ's word for which drawbars are out. The fundamental's drawbar is the
pulse width, the harmonic's drawbar is the resonance: one thickens the lines, the other the
sub-lines. Full organ, every drawbar out, fills the window; all drawbars in is silence in that
colour.

Each colour has its own oscillator: its thickness, phase, filter and drift are its own. The
interval is one, shared, and **detune** lets blue's interval leave white's; at 0 the two are locked
to one grid. Locked, with phases half an interval apart, they alternate white and blue. With equal
phases they stack into the overlap tone with dark between. Detuned, the two beat: a moiré within
one person's pattern. The colours meet as AND, the overlap tone where both are lit; they could also
meet as XOR, dark where they coincide, lines that cancel: the ring modulation of the palette.

## Drift

**Drift** is the lines' own motion: the phase advancing per second, so the oscillator has two
rates, the interval over distance and the drift over time. Phase is where the lines sit; drift is
how fast they travel. It is a low setting, possibly a connection (the body bend, or something else,
could set its speed): blue toward the person, white away from them.

A **push** is a drift event: for a moment the drift speeds up, the lines travel faster, then it
settles back to the slow drift. The lines keep the distance they gained; nothing comes back.

## The voice

Each person is a **voice**; the number of people is the polyphony. Presence is the note: attack
when a person arrives and release when they leave, both on the window. A **decay** and a
**sustain** would add an arrival flourish, the window opening wide and settling to its rest width.
The push is an attack-release envelope on the drift, and the accent may be a trigger with a short
decay if one frame proves too short to see.

**Expression** is how fast and how hard a key is played: velocity and aftertouch. The pose pipeline
measures motion (angle motion, motion time), and under the principle of *The body* those are
sources like any feature: a fast arm can give a stronger push or a wider tint.

**Unison** is sync: two voices stacked. Two nearly equal intervals in unison beat, which is the
chorus of unison detune: the moiré, and the reason it must stay small.

## The hit

The playhead crosses a person once per revolution of the content sweep. On that frame the person is
marked: the accent. The mark lasts one frame (or the few closest frames, as the beam flash), and
has to read within the palette of four tones. Candidates, each per colour where it applies:

| Mark         | For the frame                                                        | Settings                          |
|--------------|----------------------------------------------------------------------|-----------------------------------|
| tint         | each colour's lines take the other colour: blend 0 none, 1 swap     | blend 0..1 per colour             |
| mask flash   | the mask goes brighter blue; white is beam mode's flash              | the blue level                    |
| push         | the drift speeds up for a moment and settles back                    | strength, settle time             |
| window pulse | the window opens wider for the frame: more pattern shows             | the extra width                   |

Tint keeps the pattern and changes its colour, and at blend 1 it is the swap; the mask flash is
the person's own light; push and window pulse speak the drift and sync vocabulary. Chosen to try:
tint, push and the mask flash.
The layer's mark today is in Part 4.

---

# Part 2 — Meaning

## What we hold on to

Two meanings are fixed. Neutral, arms hanging and standing straight, is full blue over the window:
the blue ping. Raised arms is full white over the window: the bass. Both follow the sound
(`STATES.md`, S6: a glass ping for neutral, a heavy bass for arms raised). Between the two the
magic happens,
and this part describes the magic: an instrument that draws a beautiful pattern for every
combination of its parameters. So each measure of the body, alone and with the others, has to mean
something, and nothing but the two fixed points is tied to a pose.

## The rules of meaning

1. Only measures mean something: the six measured values, and what the pose pipeline derives from
   them (the symmetry of a pair). The instrument derives nothing.
2. A meaning belongs to a measure, never to a pose. The table says what happens when a measure
   changes. No pose owns a meaning; the two fixed points are the only exceptions.
3. Groups of measures (both arms, both elbows) and poses are consequences of the measures'
   meanings. They are read back in Part 3, never written here.
4. Raising one arm never means the same as raising the other, and the two arms are connected by a
   relation from the vocabulary of Part 1: not by a colour, and not by whatever two parameters were
   free.
5. The same holds for every pair of joints: a pair acts through a musical relation, or through its
   symmetry.
6. A meaning is spoken in the vocabulary: it names a synth term.
7. Every combination of the measures draws differently.
8. Few connections, much meaning. A measure may mean nothing on its own; then its symmetry may.
   Not every parameter of the vocabulary needs a measure.
9. Primary measures make the note, secondary measures modulate it. A secondary meaning needs a
   note to act on, so it is judged on a pattern, never on the fixed points.

## What each measure could mean

Each measure is a musical term. The first best guess, with the alternatives beside it.

**Primary** measures make the note: the arms are the organist's hands on the white registration,
the left the fundamental's drawbar and the right the harmonic's, the relation between them the
harmonic series, a note and its overtone; each elbow places what its shoulder pulls. **Secondary**
measures modulate the note: detune, pitch bend and drift are what a player does to a note that is
already sounding, and with the arms hanging there is no note to modulate. The drift is a setting
until a secondary measure takes it.

| Measure           | Term                      | Role      | Or                                  |
|-------------------|---------------------------|-----------|-------------------------------------|
| left shoulder     | the fundamental's drawbar | primary   |                                     |
| right shoulder    | the harmonic's drawbar    | primary   |                                     |
| left elbow        | the fundamental's phase   | primary   | the interval                        |
| right elbow       | the overtone's phase      | primary   | the overtone's order                |
| leg deviation     | detune                    | secondary | the interval; the drift             |
| body bend         | pitch bend                | secondary | the drift's speed; the phases apart |
| shoulder symmetry | unconnected at first      | secondary | a solo; a phase apart               |
| elbow symmetry    | unconnected at first      | secondary | syncopation                         |

Two events sit over every measure: the **accent** of a hit (*The hit*) and the **unison** of sync,
the windows of two people opening toward each other until the two patterns overlap and become one.

## Composition

The aim is a balanced composition that people recognise they influence. Not every parameter of the
vocabulary has to be used; they are there to be tried, because several may mean the same thing to
the eye, or the pattern's own workings may already produce what a parameter would add. A connection
earns its place when a person can find it with their body. Few connections, much meaning: a feature
may drive more than one parameter, and a parameter may stay fixed.

---

# Part 3 — Connections

## Pose results

The poses we check against: each with the light it should produce under the meanings of Part 2.
The list is the acceptance set, small on purpose; on the machine each row is a thing to stand in
and look at, and later a test. Every pose between the rows is unique and is not described.

| Pose                                        | Result                                                         |
|---------------------------------------------|----------------------------------------------------------------|
| arms hanging, standing straight             | full blue over the window: the blue ping                       |
| both arms straight up                       | full white over the window, no blue: the bass                  |
| arms out level, a T                         | white lines with sub-lines, blue between: half registration    |
| left arm up, right hanging                  | thick white lines, no sub-lines: the fundamental alone         |
| right arm up, left hanging                  | thin white sub-lines only: the harmonic alone                  |
| a T, both elbows folded                     | the lines and the sub-lines moved out from the mask            |
| a T, left elbow folded                      | the lines moved out, the sub-lines in place                    |
| a T, right elbow folded                     | the sub-lines moved off the lines                              |
| a T, leaning                                | the whole pattern's interval bent, back when upright           |
| a T, in a crouch                            | blue lines beating against the white: detuned                  |

The elbow and the secondary rows stand on the T because the elbows place white lines and the
secondaries modulate them: with the arms hanging there are no white lines to place or modulate, so
hands on the hips draw the blue ping and hands on the head draw the bass.

## The body

The instrument's sources are pose features and nothing else: values the pose pipeline measures
and publishes with every frame, smoothed. The instrument never derives a value of its own. The one
derivation the pipeline adds is the symmetry of a pair, as it derives the leg deviation from the
hip and knee angles.

| Feature           | Measure                                | In the pipeline |
|-------------------|----------------------------------------|-----------------|
| left shoulder     | 0 hanging → π straight up              | `Angles`        |
| right shoulder    | 0 hanging → π straight up              | `Angles`        |
| left elbow        | 0 straight → π folded                  | `Angles`        |
| right elbow       | 0 straight → π folded                  | `Angles`        |
| leg deviation     | 0 standing → 1 bent, stretched         | `LegDeviation`  |
| body bend         | −1 left → 0 upright → 1 right          | `TorsoTilt`     |
| symmetry, a pair  | signed: how unequal the two sides are  | to come         |

A **connection** is one feature into one pattern parameter. A feature may feed several parameters;
a parameter has one source; two features never sum into one parameter. The patch is a
**modulation matrix**: features down one side, parameters along the other, a depth and a polarity
in each cell; a feature feeding several parameters is a macro. The connections stay few, and each
carries a meaning a person can find with their body.

Everything else said about the body is description, of two kinds. A **target** is a meaning of
Part 2: what a measure should say, and the two fixed points. The connections are chosen so that
the targets come out. A **consequence** is what the chosen connections then draw for any shape.
"Both arms", "one arm higher", "the elbows together" are neither parameters nor connections; they
are consequences.

## Symmetry

The light is mirrored, so left and right in the body never show as left and right in the light. A
joint on its own can only show as a different sound, so the two joints of a pair are given two
sounds that belong together (rule 4). How unequal a pair is carries meaning of its own, as the leg
deviation and the body bend do. The **symmetry** feature holds the meaningful pairs (the shoulders,
the elbows, the arms as a whole, the two halves of the body), each signed, so that downstream the
signed value or its absolute can be used. Which arm is which matters: a person and their mirror
image draw differently.

## The first connections

The connections that make the meanings of Part 2 come out: feature into pattern parameter. The
white registration is the two shoulders. The bass is a consequence: full organ has to fill the
window, so the fundamental's drawbar alone reaches at most half a window of white and the harmonic's
drawbar fills the rest (how the two drawbars sum into a solid is Part 4's). The blue ping and the
blue's fading as the arms rise is open: blue pulse width has one source, so either one shoulder also
fades the blue, or the blue recedes by construction, drawn where the white registration leaves room.

| Feature           | Connection                                               | Alternatives                                                   |
|-------------------|----------------------------------------------------------|----------------------------------------------------------------|
| left shoulder     | white pulse width 0 → ½ (the fundamental's drawbar)      |                                                                |
| right shoulder    | white resonance 0 → 1 (the harmonic's drawbar)           |                                                                |
| left elbow        | white phase 0 → ½                                        | the interval                                                   |
| right elbow       | white overtone phase 0 → ½                               | white cutoff 2 → 4                                             |
| leg deviation     | the detune 0 → its maximum                               | the interval; the drift                                        |
| body bend         | the interval, signed about its rest                      | the two phases in opposite directions; the drift               |
| shoulder symmetry | unconnected                                              | one colour's interval or phase                                 |
| elbow symmetry    | unconnected                                              | the two phases against each other                              |
| the blue          | open: one shoulder, or by construction                   |                                                                |

Deviation is the legs' word: the rest interval is the standing pose, and the further the legs
deviate from standing, the further the interval is pushed. Pitch bend is the body's: signed, and
back to centre when the person stands straight.

## Consequences

Once the first connections run on the machine, the *Pose results* are read back here: what each
row draws, against what it should. Empty until then; what the deprecated patch draws is in Part 4.

---

# Part 4 — Implementation

## The layer

`pose_instrument` (`light/layers/projection/pose_instrument.py`) draws one boolean mask per colour,
mirrored about each person's centre pixel, cut to the window each side, unioned over people, masked
by every mask, and written as 0 or 1 into the frame. How people compose (union, visibility,
the masks, sync growth, the hit, presence) is in `LAYERS.md`, *pose_instrument*. The line
math is `LinePattern` (`line_pattern.py`). The waveform is a cosine; the filter is additive: one
overtone at `harmonic_order` (the cutoff), mixed in by `harmonic` (the resonance), placed by
`harmonic_phase`.

The waveform per colour, `x` the distance from the person in pixels:

```
u   = x / interval + phase
lfo = (1 − harmonic) · cos 2πu + harmonic · cos 2π(order · u + harmonic_phase)
lit = lfo ≥ cos(π · thickness)
```

The visual limit is a minimum width: no line and no gap narrower than `min_feature` (2°, the 90
lines per revolution). The interval never goes below twice it; a thickness that would make a line
or gap narrower is clamped, below that the colour is off, above it solid. A mask or the
window's edge cuts a line as it is, so lines slide out from behind the mask and into view at the
window. The hit today widens every line of the person by `hit_widen` each side for one frame, the
mark that does not read. Each colour has its own interval range: the shared interval with detune,
drift, the push, and the mask over the other layers' light are not implemented.

## Names

The settings today, the concept each carries, and the name that would carry it. The renames are
open (see *Open*); the code uses the current names.

| Concept                                | Current name     | Carries the concept        |
|----------------------------------------|------------------|----------------------------|
| the dim blue mask at the person        | `band_width`     | `mask_width`               |
| its brightness                         | `band_level`     | `mask_brightness`          |
| the visible part of the pattern        | `reach`          | `window`                   |
| the visual limit                       | `min_feature`    | `max_lines` (90)           |
| line thickness, fraction of interval   | `duty`           | `thickness`                |
| the hit's mark                         | `hit_widen`      | after the mark chosen      |
| lines and gaps at least the limit wide | legible          | visible                    |
| no pose control on a parameter         | `CONSTANT`       | `NONE`                     |
| the sources                            | derived controls | the features               |
| the filter's cutoff                    | `harmonic_order` | `cutoff`                   |
| the filter's resonance                 | `harmonic`       | `resonance`                |
| the overtone's place                   | `harmonic_phase` | `overtone_phase`           |
| the interval and the phase             | as named         | as named                   |

## The controls (deprecated)

The code today derives controls inside the instrument (`PoseControl`), against the principle of
*The body*; they go when the features take their place (*Open*):

| Control      | Value                                  |
|--------------|----------------------------------------|
| `CONSTANT`   | 1: the parameter holds its `high`      |
| `LIFT`       | mean \|shoulder\| / π                  |
| `ARM_SPLIT`  | \|\|left\| − \|right\|\| shoulder / π  |
| `BEND`       | mean \|elbow\| / π                     |
| `BEND_SPLIT` | \|\|left\| − \|right\|\| elbow / π     |
| `LEGS`       | `LegDeviation`                         |
| `TILT`       | (`TorsoTilt` + 1) / 2, upright 0.5     |

`LIFT`, `ARM_SPLIT`, `BEND` and `BEND_SPLIT` are derived: the sum and the difference of a pair.

## The patch (deprecated)

A patch (`PatchSettings`) routes one control into one parameter:
`parameter = low + (high − low) × control^curve`. Each colour has five patches (`duty`,
`interval`, `harmonic`, `harmonic_phase`, `phase`) and its interval range (`interval_min`,
`interval_max`) and cutoff (`harmonic_order`). The panel re-routes live; the preset
stores the routing. A parameter takes one source.

The patch in `studio.json` is a first routing and is deprecated: only its two targets, arms down
full blue and arms up full white, carry over. The rest is an example of what a routing looks like
until Part 3's connections are chosen.

| Colour | `duty`     | `interval`           | `harmonic`   | `harmonic_phase`   | `phase`         |
|--------|------------|----------------------|--------------|--------------------|-----------------|
| white  | LIFT 0 → 1 | LEGS 0.5 → 0         | BEND 0 → 1   | BEND_SPLIT 0 → 0.5 | TILT 0 → 1      |
| blue   | LIFT 1 → 0 | ARM_SPLIT 0.5 → 0.15 | BEND 0 → 0.6 | CONSTANT 0.5       | TILT 0.5 → −0.5 |

Intervals span 4° to 24° (patch value 0 → 1), cutoff 2. Lift is the fader, legs push
the white interval, an arm split tightens the blue lines, folded elbows add overtones, the body bend
shifts the two colours' phases in opposite directions.

## Shapes and what they draw now

A few shapes under the deprecated patch:

| Shape                                       | Draws now                                                  |
|---------------------------------------------|------------------------------------------------------------|
| arms hanging, standing straight             | solid blue over the window                                 |
| arms out level (a T)                        | white and blue lines alternating, equal, 14° interval      |
| both arms straight up (a V)                 | solid white over the window                                |
| one arm up, the other hanging               | white lines at half thickness over tight blue lines        |
| hands on hips, elbows folded, shoulders low | the blue splits into sub-lines                             |
| hands on the head, arms up and folded       | sub-lines in both colours, white thicker                   |
| one arm folded, one straight, both level    | the white sub-lines sit off the blue's                     |
| any of the above, leaning                   | white and blue slide across each other: overlap and dark   |
| any of the above, in a lunge or a crouch    | the white interval tightens toward 4°: the colours beat    |

## Tuning

The patch, the ranges and the levels are settings: live from the panel, saved in the preset. The
drawing (`PoseInstrument`, `LinePattern`) hot-reloads on save. Adding a control, a parameter or a
setting needs a restart.

---

## Open

- Parts 1 to 3 are revisited after the first build runs on the machine; until then the theory
  stands as written and the pose results are what the build is measured against
- The elbows at the fixed points: hands on the hips draw the blue ping and hands on the head the
  bass, because the elbows place white lines; whether the elbows should also touch the blue is a
  meaning question for that revisit
- The composition: which connections people find with their bodies (the work on the machine)
- The waveform as a parameter: sine, triangle or saw
- Tuning: intervals from a scale (4°, 6°, 8°, 12°, 24°, the series of the 90-line limit), so that
  synced people sit in simple ratios and their union reads as a chord; a scale means steps, and
  portamento between them is motion of its own
- Oscillator sync between people: one pattern's grid adopting the other's where they meet, an
  alternative to the union
- Expression: the motion features as sources for the push and the tint
- XOR as a second way for the colours to meet
- Decay and sustain on the window: the arrival flourish
- The hit's mark: how tint's blend is realised in the binary palette (which pixels of a line
  take the other colour between 0 and 1), the push's strength and settle time, the mask flash's
  blue level
- The playhead's dimming at the mask: a setting; and how the mask reaches the other layers' light
  (the compositor, or the board)
- The renames in *Names*, in code and preset, once the concepts settle
- From the machine **(site facts)**: the mask is too narrow and the window too wide at the preset's
  values; the render hides blue under full white, so the two colours need their own visualisation;
  the settings panel should warn when one control is linked twice; the instrument's settings should
  become their own root group, `PI`
- Drift: its low default per colour (0 = none), and whether its speed is patched from the body
  bend or another control
- The symmetry feature in the pose pipeline: its pairs (shoulders, elbows, arms, the body's
  halves), signed
- How the blue fades as the arms rise: one shoulder as its source, or by construction
- The instrument's derived controls replaced by the features as sources
- Whether the body bend should bend the pitch (interval) or shift the phase as now
- Whether the legs should detune blue from white or push both intervals
- Pitch bend and deviation both want the interval; a parameter has one source, so one of them goes
  to the detune or the drift
- Whether a line appearing at the limit width flickers when a pose hovers at its threshold
  **(deduction: the pipeline's smoothing should hold it)**
- Whether the 90-line limit holds with the moiré of two synced patterns on the machine
- Camera level: a pitched wide-FOV camera reads a small spurious body bend near the frame edges, a
  pitch bend nobody played; check a straight person at the edge
