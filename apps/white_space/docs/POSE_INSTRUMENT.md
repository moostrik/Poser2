# White Space — Pose Instrument

The instrument the participants play with their bodies. An instrument is what a player plays: a
keyboard synth is its keyboard, its synth engine and the patch that ties them. Here a person's
body takes the place of the keyboard.

| Part                           | In a keyboard synth      | Here                                                       |
|--------------------------------|--------------------------|------------------------------------------------------------|
| what the player acts on        | keyboard, wheels, pedals | the body, as the pose pipeline measures it (*The body*)    |
| what makes the result          | the synth engine         | the light synth (`LIGHT_SYNTH.md`)                         |
| what ties them                 | the patch                | which measure goes into which parameter, and by how much   |
| the whole                      | the instrument           | the pose instrument                                        |

One body plays two engines: the same measures drive the sound synth in Max and the light synth.
This document is the light's. It is the bridge between the pose data and the light synth, in four
parts. **Vocabulary** fixes the words. **Meaning** is what each measure of the body should say.
**Connections** is how the measures and the events are wired to the synth so that the meaning
comes out. **Implementation** is the bridge's design and the layer's state today.
Meaning and connections are one loop: meaning sets the targets, the targets pick the connections,
the connections have consequences, and the consequences are new meaning to judge. While one is
held still the other is worked on.

The layer's place among the other layers is in `LAYERS.md`; the states that run it are in
`STATES.md`.

## The instrument

Each person makes light in the projection. At their azimuth sits a dim blue **mask**: it
goes over everything at the person, the patterns of everyone and the light of the other layers;
the playhead passes through it dimmed by a setting. Around the mask, mirrored about the person,
their **pattern**: lines of full white and full blue, generated behind the mask and coming out from
under it. Only a **window** of the pattern is visible each side of the person; sync opens the
window wider.

The ground rules, as set:

- About 90 lines per revolution, line and gap equal, is the visual maximum **(site fact)**. It is a
  guide: a good reason may go past it. It bounds the pitch and not the width of a line or a gap
  (`LIGHT_SYNTH.md`, *The rules*).
- Full on: every pixel of white and of blue is off or full, since the fixture projects on whatever
  is around it and subtle brightness is lost (`LIGHT_SYNTH.md`, *A light synth*). The mask is the
  one dim thing.
- No blinding: the instrument never blinds a person with white; people must be able to see each
  other. The mask is how, so full white can fill a window and never fall on a face. This rule is
  the instrument's alone, not the other layers'.
- White and blue are projected separately by the fixture **(site fact)**, so where both are on the
  overlap reads as a subtly different tone. The palette is four tones: dark, blue, white, both.
- No jitter: the pose pipeline's smoothing is the only smoothing. The instrument adds no smoothing
  and no steps, since steps are jitter of another kind, and no flicker of its own: a smooth change
  of the body is a smooth change on the projection.
- Lines follow the pose. On top of that they travel by themselves, slowly and peacefully, blue
  inward and white outward: a low speed that may be 0.
- The output each side of a person is symmetric, except the window's reach: sync opens one side.
- The pattern is the light synth's; the six measures play it; every combination of the six draws
  differently; the arms come first, legs and body bend second.
- Arms down is neutral: full blue over the window. Arms up: full white. In between, the pattern.
  Full is solid over the window, thinning at its ends (`LIGHT_SYNTH.md`, *The window*).
- Blue and white can behave differently; line thickness and interval can change.
- A hit (the playhead crossing the person) marks that person. Wider lines do not read, and a tint
  of the lines did not read **(site facts)**; the mark is the push and the mask's flash
  (*Events*), with the playhead's marker passing the person; each mark is optional through its
  levels.
- Sync makes more of the pattern visible: the window opens, the lines stay what they are.

---

# Part 1 — Vocabulary

The language of the pattern is the light synth's and is in `LIGHT_SYNTH.md` (*Terms and what
belongs where*): oscillator, parameter, slot, source, output, pitch, interval, pulse width, phase,
speed, hardness, LFO, envelope, window, reach, taper, push, voice. Standard synth terminology
where the synth is a synth; our own word where the wall differs. The words of the bridge:

| Word       | What it is                                                                            |
|------------|---------------------------------------------------------------------------------------|
| measure    | a value of the body, measured by the pose pipeline (*The body*); ours: the body is the controller |
| source     | what is plugged into a parameter's slot: a measure, an LFO's or an envelope's output  |
| connection | a measure as the source in a slot: the measure, the parameter, the base, the amount   |
| patch      | everything set for the synth, the connections among it; shared by all voices          |
| voice      | one person's instance of the light synth                                              |
| event      | what the instrument triggers in a voice: presence, the hit, sync (*Events*)           |

What is the instrument's and what is the synth's (`LIGHT_SYNTH.md` states the same split from the
synth's side): the instrument decides what each source is, which measure, and any shaping of a
measure before it is handed over (a dead zone, a remap, a mean of two joints, each of which makes
one source out of measures); when the gates open (presence, the hit); the reaches; which output
is white and which blue; and the mask. Everything from the slot on is the synth's, and the synth
never changes a source before the slot and never adds two.

---

# Part 2 — Meaning

## What we hold on to

Two meanings are fixed, the ground rules' neutral and raised arms: the blue ping and the bass, as
the sound has them (`STATES.md`, S4: a glass ping for neutral, a heavy bass for arms raised).
Between the two the magic happens, and this part describes it: a pattern for every combination of
the measures, so each measure of the body has to mean something, and nothing but the two fixed
points is tied to a pose.

## The rules of meaning

1. Only measures mean something: the six measured values of the body, the distance, and what the
   pose pipeline derives from them (the symmetry of a pair). The instrument derives nothing.
2. A meaning belongs to a measure, never to a pose. The table says what happens when a measure
   changes. No pose owns a meaning; the two fixed points are the only exceptions.
3. Groups of measures (both arms, both elbows) and poses are consequences of the measures'
   meanings. They are read back in Part 3, never written here.
4. Raising one arm never means the same as raising the other, and the two arms are connected by a
   relation from the vocabulary of the light synth: not by a colour, and not by whatever two
   parameters were free.
5. The same holds for every pair of joints: a pair acts through a musical relation, or through its
   symmetry.
6. A meaning is spoken in the vocabulary: it names a term of the light synth.
7. Every combination of the measures draws differently.
8. Few connections, much meaning. A measure may mean nothing on its own; then its symmetry may.
   Not every parameter of the synth needs a measure.
9. Primary measures make the note, secondary measures modulate it. A secondary meaning needs a
   note to act on, so it is judged on a pattern, never on the fixed points.

While the instrument is being built the rules are aims and not gates. Two things are fixed: arms
hanging is full blue and arms raised is full white. Every combination drawing differently (rule 7)
is the aim in the end, not a test of each step. A calculation on the measures may live in the
instrument while it is tried; once the result is liked, what belongs in the pose pipeline moves
there (rule 1). The connections of today depart from rule 4 as such an experiment: the two arms are
told apart by a colour, each arm playing one oscillator.

## What each measure means

Each measure is a musical term of the light synth: a parameter of an oscillator (pitch, pulse
width, phase, speed), the level of an LFO, a reach. **Primary** measures make the note: the arms,
each arm one oscillator, the left the white and the right the blue. **Secondary** measures
modulate a note that is already sounding: the legs and the body bend. The distance, how far the
person stands from the fixture, is the one measure that is not of the body's pose.

| Measure           | Role      | Term                                                        |
|-------------------|-----------|-------------------------------------------------------------|
| left shoulder     | primary   | the weight of the white: its pulse width                    |
| right shoulder    | primary   | the weight of the blue: its pulse width                     |
| left elbow        | primary   | the pitch of the white: how fine its lines are              |
| right elbow       | primary   | the pitch of the blue: how fine its lines are               |
| leg deviation     | secondary | the sway: the level of an LFO, a vibrato in space           |
| body bend         | secondary | the flow: the speed of both, outward or inward by the lean  |
| distance          | secondary | open                                                        |
| shoulder symmetry | secondary | open                                                        |
| elbow symmetry    | secondary | open: it shows unconnected, as the colours in or out of tune |

Two events sit over every measure: the **accent** of a hit and the **unison** of sync, the windows
of two people opening toward each other until the two patterns overlap and become one (*Events*).

## Composition

The aim is a balanced composition that people recognise they influence. Not every parameter of
the synth has to be used; they are there to be tried, because several may mean the same thing to
the eye, or the pattern's own workings may already produce what a parameter would add. A
connection earns its place when a person can find it with their body. Few connections, much
meaning: a measure may drive more than one parameter, and a parameter may stay at its base.

---

# Part 3 — Connections

## Pose results

The poses we check against: each with the light it should produce under the meanings of Part 2.
The list is the acceptance set, small on purpose; on the machine each row is a thing to stand in
and look at, and a test. The rows are the dummy's saved poses (`data/poses.json`, *The dummy*),
so each is one pick in the panel. The first two are the calibrator's reference poses, `neutral`
(arms hanging, standing) and `raised` (arms up); the others are built on them, the T halfway along
the calibrator's shoulder arc from neutral to raised. Every pose between the rows is unique and is
not described.

| Pose                                        | Result                                                           |
|---------------------------------------------|------------------------------------------------------------------|
| neutral                                     | full blue over the window: the blue ping                         |
| raised                                      | full white over the window, no blue: the bass                    |
| arms out level, a T                         | white and blue lines, each half the interval wide, alternating   |
| left arm up, right hanging                  | white and blue both solid: the overlap tone over the window      |
| right arm up, left hanging                  | dark but for the mask                                            |
| a T, both elbows folded                     | the T's lines, finer, white and blue still in tune               |
| a T, left elbow folded                      | the white lines finer than the blue: the colours slide apart     |
| a T, right elbow folded                     | the blue lines finer than the white: the colours slide apart     |
| a T, leaning                                | the T's lines flowing, outward one way and inward the other      |
| a T, in a crouch                            | the white lines swaying against a still blue                     |

## The body

The instrument's measures are pose features and nothing else: values the pose pipeline measures
and publishes with every frame, smoothed. The instrument never derives a value of its own. The one
derivation the pipeline adds is the symmetry of a pair, as it derives the leg deviation from the
hip and knee angles. The distance is the tracker's, read from where the feet meet the floor
(`TRACKING.md`, *The pose's distance*).

| Feature           | Measure                                                  | In the pipeline |
|-------------------|----------------------------------------------------------|-----------------|
| left shoulder     | 0 hanging → ±π straight up, the sign the side it passes  | `Angles`        |
| right shoulder    | 0 hanging → ±π straight up, the sign the side it passes  | `Angles`        |
| left elbow        | 0 straight → ±π folded                                   | `Angles`        |
| right elbow       | 0 straight → ±π folded                                   | `Angles`        |
| leg deviation     | 0 standing → 1 bent, stretched                           | `LegDeviation`  |
| body bend         | −1 left → 0 upright → 1 right                            | `TorsoTilt`     |
| distance          | 0 the zone's near edge → 1 its far edge                  | `Distance`      |
| symmetry, a pair  | signed, left minus right: how unequal the two sides are  | `AngleSymmetry` |

The angle extractor measures the geometric angle between body segments, the arm against the
torso line, mirrored on the right so a symmetric pose reads equal on both sides; what a body
reads at neutral is its geometry, and a vertical arm is not π from it. The angle calibrator
(`modules/pose/nodes/filters/AngleCalibrator.py`, settings `pose.angle_calibrator`) then maps
each joint from two reference poses, the body with the arms hanging and with the arms raised,
each held as the raw readings: every reading at neutral becomes 0; the shoulder's raised reading
becomes π, so the fixed points are the feature's 0 and π for sound and light alike; the elbow's
raised reading is where its straight is with the arm up, so a relaxed straight elbow reads 0 in
both poses and π is the fold from there. The readings are tuned in the panel, against a person or
the dummy. The sign of an angle is the side of the body the limb passes, outward or across, and
means nothing to the instrument: a measure is the absolute over π, 0..1, so an arm raised outward
and one raised across the body read alike.

## Symmetry

The light is mirrored, so left and right in the body never show as left and right in the light;
which arm is which shows only as a different sound. The **symmetry** feature (`AngleSymmetry`)
holds the meaningful pairs, the four joints (shoulder, elbow, hip, knee) and the arm and the leg as
a whole per side (`arms`, `legs`), each signed, left minus right, so that downstream the signed
value or its absolute can be used. A person and their mirror image draw differently.

## The connections

A **connection** is one measure as the source in one parameter's slot (`LIGHT_SYNTH.md`,
*Modulation*): the measure is the source, 0..1, and the slot's base and amount carry the rest,
the range and the direction. A measure may feed several parameters; a slot has one source; two
measures never sum into one parameter.

Each arm plays one oscillator, the left the white and the right the blue. The bases and amounts
are the preset's starting values, tuned on the machine:

| Source         | Range | Parameter         | Base     | Amount       | At full                                       |
|----------------|-------|-------------------|----------|--------------|-----------------------------------------------|
| left shoulder  | 0..1  | white pulse width | 0        | 1            | solid white                                   |
| right shoulder | 0..1  | blue pulse width  | 1        | −1           | no blue                                       |
| left elbow     | 0..1  | white pitch       | 25.7     | 46.3 lines   | 72 lines, one every 5°: finer as the arm folds |
| right elbow    | 0..1  | blue pitch        | 25.7     | 46.3 lines   | 72 lines                                      |
| body bend      | −1..1 | white speed       | 3.9°/s   | 15°/s        | both colours flowing the way of the lean      |
| body bend      | −1..1 | blue speed        | −4.5°/s  | 15°/s        | as white's                                    |
| leg deviation  | 0..1  | the LFO's level   | 0        | 1            | the LFO at full swing                         |
| the LFO        | −1..1 | white phase       | 0        | ¼ interval   | white swaying ¼ interval each way, at 0.5 Hz  |

Unconnected: the symmetries, the distance, blue's phase, the hardness.

## Events

What the instrument triggers in a person's voice, beside the measures:

| Event    | When                                                        | Acts on                                                     |
|----------|-------------------------------------------------------------|-------------------------------------------------------------|
| presence | the person is seen: a gate open while they are there        | the presence envelope, on both reaches                      |
| hit      | the playhead crosses the person: the tick closest to it     | each oscillator's push, and the mask's flash                |
| sync     | the similarity of a pair is over its threshold              | the reach on the partner's side: full reaches the partner   |

Presence and the pushes are envelopes of the synth (`LIGHT_SYNTH.md`, *The envelope*). The
reaches are the bridge's: it gives each side's reach to the voice as a value, the rest width grown
toward the partner by sync, since only the bridge knows where the partner stands. The mask's
flash is the bridge's own envelope of the same block, opened by the hit at once and falling back
over its release: the mask goes to its flash levels and returns to its own. The mask is the
bridge's and not the synth's.

## Consequences

What follows from the connections, before the machine has been judged **(deductions)**:

- The four corners of the two shoulders are the four tones of the palette: both hanging blue, both
  raised white, the left up alone the overlap tone, the right up alone dark. Every combination of
  the two arms draws differently.
- The elbows' symmetry shows by itself, unconnected: equally folded, white and blue share one
  interval and stay in tune; unequally, the colours slide past each other with distance
  (`LIGHT_SYNTH.md`, *In the pose instrument*).
- An elbow shows nothing while its own colour is solid or dark: the pitch needs a note.
- The legs and the bend show nothing where both colours are solid or dark, as rule 9 expects.
- A lean also moves the leg deviation (*Open*), so leaning brings some sway with the flow.

Once the connections have been played on the machine, the *Pose results* are read back here: what
each row draws, against what it should.

---

# Part 4 — Implementation

## Design

The bridge gives every person a voice of the light synth, sends its two outputs to white and to
blue, feeds the measures of *The body* into the slots of *The connections*, triggers the *Events*,
and draws the mask over the result. The synth, its building blocks and its rules are
`LIGHT_SYNTH.md`'s.

## State

### The layer

`pose_instrument` (`light/layers/projection/pose_instrument.py`) is the bridge. It gives each
person a `Voice` of the light synth (`light/synth`: `Oscillator`, `Envelope`, `Slot`, `Voice`),
paints the voice's output 1 into white and its output 2 into blue over the person's window, the
fuller of overlapping voices showing, and draws every mask over the result. Distances are taken
from the person's own azimuth and not from their centre pixel, so a walking person's lines move
smoothly. How people compose (union, the masks, sync, the hit, presence) is in `LAYERS.md`,
*pose_instrument*. The instrument draws the playhead's marker over the masks, dimmed inside them
(`PlayheadMarker`, `playhead_marker.py`), and the render shows the overlap of the two colours as
a tone of its own
(`render/shaders/lightsimulation.frag`). A tick costs about 0.2 ms per person at 3600 pixels.

### Sources and connections

The layer reads pose features from the LERP frames and derives nothing. The sources are the
elements of `Angles` (the four arm joints), `LegDeviation`, `TorsoTilt`, `Distance`, and
`AngleSymmetry` (`modules/pose`): one signed scalar per pair, left minus right, normalised to −1..1, made by its
own extractor (`AngleSymExtractor`) and windowed, graphed and sent to Max like the leg deviation.
The pairs: the shoulder, the elbow, the hip and the knee (the joint angles), the arms (shoulder and
elbow together per side), the legs (hip and knee together per side, weighted as the leg deviation
is, by its settings).

The connections are code, not settings: one method, `PoseInstrument.connect`, takes a person's
measures and returns the sources of the white and the blue oscillator's slots, and it hot-reloads
on save. It is *The connections* written out; `connect_lfo` beside it gives the LFO's level its
source, and runs first each tick so `connect` can read the LFO's output. The bases and the amounts,
the range and the direction of every connection, are settings, so the panel keeps the values and
the code keeps the routing. The preset carries values only.

### People

The bridge sets each side's **reach** every tick: `reach.width`, grown by sync toward every
similarity-matched partner along the shorter arc, full reaching the partner; the partner's
presence scales the growth, so a partner leaving lets go smoothly. The voice multiplies the reach
by presence (attack from 0, release to 0) and thins the lines to nothing over the taper
(`LIGHT_SYNTH.md`, *The window*). On release the window closes and the mask fades with it.

The **mask** is a band `mask.width` wide at the person, each channel at its level (`mask.white`,
`mask.blue`) times presence, and goes over everything at the person: the patterns of every voice.
In the preset the white is 0, so the mask is the dim blue band. The playhead's **marker**
(`PI.playhead`: its width, its white and blue) is drawn by the instrument over the masks, dimmed
to `playhead.at_mask` inside one, so the marker never blinds and no other layer shares a mix with
the instrument at a person.

### The hit

On the tick the playhead is closest to a person (`PlayheadCrossing`, `PoseInstrument.HIT_TICKS`,
as the beam flash), each of the person's oscillators is pushed (its `push`, falling back over its
`push_release_seconds`) and the mask goes to its flash levels (`mask.flash_white`,
`mask.flash_blue`), falling back to its own over `mask.flash_release_seconds`. Every mark has a
white and a blue level, and 0 is off.

### Settings

`PI` is a root settings group of the app (`apps/white_space/settings.py`), not a layer group
(`PoseInstrumentSettings`, `light/layers/projection/pose_instrument.py`), grouped by what is
tuned together, knobs throughout:

| Group           | What it holds                                                                     |
|-----------------|-----------------------------------------------------------------------------------|
| `max_lines`     | the visual limit: the pitch ceiling                                               |
| `mask`          | the mask: width, its white and blue; the flash: its white and blue, its release  |
| `playhead`      | the playhead's marker: width, its white and blue, its level inside a mask         |
| `window`        | how far the pattern shows and when: the shape (taper, attack, release) and the reach (width, its bypass, the sync threshold) |
| `white_lines`, `blue_lines` | an oscillator: its On switch and Bypass All button, a slot per parameter, its push (amount, release) |
| `lfo`           | the LFO: rate, phase, and the level's row                                         |
| `dummy`         | *The dummy*                                                                       |

The groups run from the person outward: the mask at the person, the marker that passes them, the
window around them, then what fills the window, then the tool. Wherever a mark has a level per
channel the two settings are `white` and `blue`; the oscillators are the lines.

A slot is a row titled with its parameter's name and reads Base · Amount · Curve · Bypass
(`LIGHT_SYNTH.md`, *Modulation*); every other row has its title too (Push, LFO, Shape, Reach,
Mask, Flash, Playhead), so the panel reads the same way throughout.

The synth's settings classes (`OscillatorSettings`, `LfoSettings`, `WindowSettings`) are extended
by the bridge's where a concept spans both (`window`); the voice reads only its own fields. The
layer keeps its `blend`. No setting routes anything: an amount does nothing until `connect` gives
its parameter a source.

### Hot reload

The composition is worked on the machine in two ways. Values are tweaked in the panel and saved in
the preset. Connections are edited in code and seen at once: `HotReloadMethods` (`modules/utils`)
watches `pose_instrument.py` and the synth's files and, on save, re-executes their class bodies
and module constants into the running app, so `connect`, the drawing methods and the synth's
classes change under the layer without a restart.

What the reloader patches: methods of a class (instance, static, class), class constants, module
constants. What it does not: module-level functions, imports, settings fields, and the objects
already built (a settings instance keeps its fields). Hence the rules:

- Everything that may change lives in a class method or a module constant, never in a module-level
  function.
- The synth's `Parameter` is an `IntEnum`, so a source's key still matches after a reload redefines
  the class.
- The numbers of a connection are its slot's base and amount, settings, never literals in
  `connect`, so the preset keeps describing the show.
- A new setting, feature or parameter needs a restart; a new connection does not.
- The table of *The connections* is `connect` in words; when one changes, the other follows in
  the same change.

### Playing by hand

A parameter's base is already the hand's value, so playing by hand is bypassing modulation. Every
slot has a **Bypass** (`PI.white_lines`, `PI.blue_lines`, and the level of `PI.lfo`): bypassed, the parameter
is its base while every other parameter keeps following the body, so a pose can be taken apart
parameter by parameter; lifted, it follows again with its amount as it was. An oscillator's
**Bypass All** button sets its five ticks at once, and pressed again with all set clears them, so
the panel draws that colour for everyone, live people and the dummy alike, while the other colour
and the LFO keep following the body; the ticks are the one state, and can be changed one by one
after. An oscillator's **On** switch, off, draws nothing for that colour.
`window.width_bypass` holds both reaches at the width without a partner, presence still opening
and closing them. What the body gives is read in the render's data graphs, per person.

### Tests

`apps/white_space/tests`: the synth in `test_synth_oscillator.py` (among them the no-jumps rule,
each parameter swept in small steps), `test_synth_envelope.py` and `test_synth_voice.py`; the bridge
in `test_pose_instrument.py`.

### The pose results now

The rows of *Pose results* draw what that table says, in the unit tests
(`tests/test_pose_instrument.py`: the four corners of the shoulders, one shoulder moving its own
colour only, an elbow making its own colour finer, equal elbows in tune, the lean's flow both
ways, the legs' sway against a still blue, and a small move of any arm measure a small change of
the picture). What they draw on the machine has not been judged yet.

### The dummy

The dummy (`pose/dummy.py`, settings `PI.dummy`) stands in for a person while the instrument is
judged: a figure of the pipeline's 17 landmarks whose joints are set by the panel, each joint's
degrees the angle the angle extractor reads at it when the figure is upright, the signed angle
from the segment above to the limb below, the right side mirrored as the extractor mirrors it
(the shoulder 0 hanging, 90 across the body, 180 up, 270 out; the elbow 180 straight, 0 folded;
the hip 180 standing, 90 leg out level; the knee 180 straight, 90 bent), standing at `azimuth`.
What is set is what the extractor reads. The `torso` leans the upper body over standing legs, as
a person leans: the arms, measured against the leaning torso line, still read as set; the hips
read off neutral by the lean, as a person's do. Its sides are named as the pipeline names
people's, the left on image-right.
Its frame joins the interpolated
poses before the LERP filters at its own id, `max_players`, between the live players and the
ghosts (which start one above it), so it is a pose like any other from there: the filters stamp
the playhead offset, the symmetries, the leg deviation and the bend on it, the render draws its
figure over the projection row where it stands (as it does every pose, `render.pose_figures`)
and its skeleton with the same data graphs as a player's in the pose row's last column,
the azimuth overlay marks it, Max plays it at its slot and the instrument lights it. Only the
joints are set; the leg deviation and the bend are the pipeline's, derived from the figure as for
a person, and its angles pass through the angle extractor and the calibrator as a person's do.
Where it stands is set, as no tracker sees it: the `azimuth`, and the `distance`, 0..1 as the
`Distance` feature is.
With `solo` the live players' frames are left out at the merge, so from the LERP filters on the
dummy is the only pose; the tracker and the earlier stages still see them.
What the pipeline reads of its poses is what the calibrator is read against: the preset's
calibration is the dummy's raw readings, so its hanging arm reads 0 and its vertical arm π.

Its poses are named in `data/poses.json`, the rows of *Pose results*, picked in the `pose`
select and saved under `name` with `save`. A change of any measure morphs over `morph` seconds
(`easeInOutSine`), the azimuth and the arms the shortest way round, an exact opposite going up
through the front on both sides. `enabled` comes back as the preset saved it; in the show the
dummy is never in the room, so the site's preset is saved with it off.

---

## Open

The open questions of the synth itself are in `LIGHT_SYNTH.md`, *Open*.

Meaning and connections:

- Whether the connections of Part 3 can be found with the body: each arm an oscillator, the flow,
  the sway; and what the symmetries should play
- Whether the two extreme corners of the shoulders, the overlap tone everywhere and dark, read as
  striking or as broken on the projection
- Which of the instrument's departures from the rules of meaning stay, and which calculations move
  into the pose pipeline, once the result is liked
- The lean moves the hips' reading by the lean, on the dummy as on a person, so at full lean the
  leg deviation reads about 0.75 and whatever it plays sounds with the bend; whether the leg
  deviation should be taken against the vertical instead of the torso line
- The symmetries: which of the four pairs earns a connection, and what it means
- The distance: what it plays, and whether the reading is steady enough to play it; it comes off
  the detection box's bottom row and is weakest at the zone's far edge (`TRACKING.md`, *The pose's
  distance*)
- The composition itself: which connections people find with their bodies
- Whether the wiring of a connection is chosen in the panel or stays code, as `connect` is

Not yet in the design:

- Tuning between people: intervals in simple ratios, so that synced people read as a chord
- Oscillator sync between people: one pattern's grid adopting the other's where they meet, an
  alternative to the union
- Expression: the motion features (angle motion, motion time) as sources, a fast arm giving a
  stronger push
- Decay and sustain on the window: the arrival flourish

From the machine **(site facts)**:

- The mask is too narrow and the window too wide at the preset's values
- Whether the 90-line limit holds with the moiré of two synced patterns
- A pitched wide-FOV camera reads a small spurious body bend near the frame edges, a bend nobody
  played; check a straight person at the edge
