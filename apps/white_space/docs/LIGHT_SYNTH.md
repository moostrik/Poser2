# White Space — The Pose Instrument's Light Synth

The light of the pose instrument is made by a light synth. This document describes that part and
nothing else: the lines the oscillators draw, their parameters and what each looks like on the
projection, how a parameter is connected, and the building blocks. What plays the parameters is
not part of it: the instrument, its measures, meanings, connections and events, are in
`POSE_INSTRUMENT.md`. The code is the package `light/synth`: `Oscillator`, `Envelope`, `Slot` and
`Voice`.

The synth has three levels, and only the last knows of colour or of pose data:

| Level                | What it is                                                         | Knows colour |
|----------------------|--------------------------------------------------------------------|--------------|
| oscillator, envelope | the building blocks (*The oscillator*, *The envelope*)             | no           |
| voice                | two oscillators, one time, one amp stage, its LFO; two outputs     | no           |
| pose instrument      | the bridge between pose data and synth (*In the pose instrument*)  | yes          |

## Terms and what belongs where

The terms are a synth's wherever the synth is a synth, and our own where the wall differs. One
value walked through, the left shoulder:

| Term          | What it is                                                                          |
|---------------|-------------------------------------------------------------------------------------|
| **measure**   | what the pose pipeline says about the body: the left shoulder's angle, 0 hanging, π raised. The body's, whether or not the light uses it; the sound reads the same one. Ours: the body is our controller |
| **source**    | that measure once the bridge hands it to the synth to move something: 0..1, plugged into a slot. Also an LFO's or an envelope's output. The word is a role: whatever is in a slot's socket, one per slot |
| **parameter** | a knob of the oscillator, the thing being moved: pitch, pulse width, phase, speed, hardness; the LFO's level. Its value each tick is its base plus what its source adds |
| **slot**      | the parameter's modulation row, its mechanism: base, amount, curve, bypass, and the one socket the source goes into |
| **output**    | what the oscillator makes of its parameters: on or off per pixel along the wall; two per voice, sent to white and to blue |

In one line: the measure (the angle) is made a source by the bridge (0..1, into white's pulse
width slot), moves a parameter through its slot (`base + amount × curve(source)`), and the
oscillator turns its parameters into an output (white lines of that width). The wall's own words
are kept where a synth's would mislead: the interval is a pitch in degrees, the speed a rate in
degrees per second; reach, taper, window, presence, push and hardness are the amp stage and the
slew, which here act on width instead of level (*A light synth*).

What belongs where:

- **The synth owns** how a source becomes a parameter's value (the slot), how the amp stage acts
  on the parameters (the window, presence), how an LFO or an envelope makes its output, and how
  the parameters become an output (the waveform). A slot has one source, so the synth never adds
  two sources; and it never changes a source before the slot: what arrives is what it uses.
- **The instrument owns** what each source is: which measure, and the shaping of that measure
  before it is handed over (a dead zone as `PI.measures`, a remap); when the gates open (presence,
  the hit); the reaches; which output is white and which blue; the mask (`POSE_INSTRUMENT.md`,
  Part 1).

## A light synth

The design is a synthesizer's, part for part, so it can be reasoned about as one.

| Synth                          | Here                                                          |
|--------------------------------|---------------------------------------------------------------|
| oscillator, a pulse wave       | an oscillator (*Parameters*); its pitch in lines per revolution |
| pulse width                    | pulse width                                                   |
| LFO                            | an LFO: an oscillator used as a source (*Modulation*)         |
| envelope with a gate           | the envelope: push, presence (*The envelope*)                 |
| amp envelope and VCA           | the window, over distance, on the pulse width (*The window*)  |
| modulation matrix              | the slot: `parameter = base + amount × curve(source)`         |
| slew, lag                      | hardness                                                      |
| clock                          | the time, which can run faster or slower                      |
| a voice, its two oscillators   | a voice (*The voice*): one per person, while present          |
| keyboard and controllers       | what plays the parameters: not in this document               |

Where it is not a synth:

- **The building blocks run along the wall as well as in time.** An oscillator or an envelope is
  given positions and gives a value for each, so a whole wave is seen at once, travelling
  (*Distance and time*). A synth's run in time only.
- **There is no level.** The fixture projects on whatever is around it, not on a screen, so subtle
  differences of brightness are lost **(site fact)**: a pixel of an output is off or full. A thin
  line is faint **(site fact)**, so thinness does the work a synth gives to level: the window, a
  synth's amp envelope, acts on the pulse width. The pulse width is therefore both the timbre and
  the level of an output.
- **A voice has a place.** A synth mixes its voices into one output. Here a voice is drawn where
  its person stands and moves with them, and two voices meet only where their windows overlap
  (*The voice*).
- **The outputs do not sum.** A synth adds its oscillators into one signal. A voice's two outputs
  stay two: white and blue, which meet as four tones (*In the pose instrument*).
- **The highest pitch is low.** About 90 lines per revolution is where lines stop being lines
  (*The rules*), as hearing ends a synth's range; it leaves a few lines each side of a person, not
  thousands of cycles.
- **Smooth in, smooth out, without exception.** A synth avoids clicks in its sound but steps
  freely in its control: square and sample-and-hold modulators, quantisers, hard sync. Here a step
  in control is a step on the wall, so every source is smooth: the sine, the eased envelope. Only a
  deliberate event, as the push, changes something at once, and it changes a rate, not the picture.

## The voice

A **voice** is one person's pattern, for as long as they are present. It knows nothing of colour:
it has two outputs, and what each is projected in is the instrument's choice. There is one synth
and it is polyphonic. The patch is shared: what is connected to which parameter, with what base
and amount, is set once and holds for every voice. The values are each voice's own: what flows
through a connection comes from that person, so each person's body drives their own pattern, and
the same patch draws differently for every person. A voice also has its own time and its own
presence. Put another way, each person gets their own instance of the light synth, and every
instance is given the same settings.

| Part       | How many | What it is                                                          |
|------------|----------|---------------------------------------------------------------------|
| oscillator | 2        | one per output, the same parameters, each its own values (*Parameters*) |
| time       | 1        | shared by the oscillators and the LFO (*Distance and time*)         |
| amp stage  | 1        | each side's window, and presence; for both outputs (*The window*)   |
| LFO        | 1        | in time; its output a source for the caller to wire (*Modulation*)  |

An oscillator draws a line, a gap, a line, a gap, outward from the person, the same on both
sides (or, unmirrored, one grid passing behind the person; *Distance and time*). A line is full
and a gap is off. The two sides are not voices or oscillators: a mirrored oscillator is given each
pixel's position without its sign, so it cannot tell left from right, and only the window knows
the side.

Where the windows of two voices overlap their lines join per output: a pixel of an output is lit
where either voice's oscillator lights it.

Two pitches in a simple ratio (1:1, 1:2, 2:3) are tuned: with equal phases and equal speeds the
two outputs' lines coincide at regular places, every 1, 2 or 3 of the shorter interval; with
unequal speeds the places travel. They stay tuned only while their sources move them alike: a
pitch's amount is in lines, so one source into both pitches keeps their difference and not their
ratio (*Open*).

## Parameters

Per oscillator, five parameters and nothing else. Each has a base value and can be connected to a
source (*Modulation*). The hardness is a parameter like the others, its base 1, hard.

| Parameter   | Unit                 | What it is                                              |
|-------------|----------------------|---------------------------------------------------------|
| pitch       | lines per revolution | how fine the lines are: 25.7 is a line every 14°        |
| pulse width | fraction of interval | the thickness of a line: 0 none, 1 solid                |
| phase       | intervals            | where the lines sit relative to the person              |
| speed       | degrees per second   | how fast the lines travel: positive outward, 0 still    |
| hardness    | 0..1                 | the flanks of a line: 1 hard (default), 0 softest       |

An oscillator also has its push (*Distance and time*): `push`, the speed a hit adds to it, and
`push_release_seconds`, the release of its own push envelope; the amount and the time of an
envelope, not a parameter, so no slot. And it has two switches: `enabled`, off, its output is
dark whatever its slots say; and `mirror` (*Distance and time*), on, both sides of the person
draw the same, off, one pattern passes behind them. The switches and the bypasses are the
panel's, not the body's, so flipping them is the one allowed step.

**Pitch** is how many lines fit in a revolution, the site fact's own unit; the **interval** is the
spacing it gives, `360° / pitch`, in which phase and pulse width are measured. The pitch never
goes above the visual limit (`max_lines`) and never below 2, one line per half turn. A change of
pitch moves a far line more than a near one, as an accordion opens from the person.

**Pulse width** is the thickness of every line, `pulse width × interval` wide. At 0 the output is
dark, at 1 it is solid, and halfway line and gap are equal.

**Phase** places the lines. At 0 a line is centred on the person; at ½ a gap is. A whole interval
further the picture is the same.

**Speed** is how fast the lines move across the projection. It is the same whatever the interval,
so a change of interval changes the spacing and not the travel. At interval 10° and speed 2° per
second the lines are centred at 0°, 10°, 20°; a second later at 2°, 12°, 22°; after five seconds
at 10°, 20°, 30°, the same picture, a new line having come out at the person.

**Hardness** shapes the flanks of the lines and nothing else. At 1 a line ends at its edge and
every pixel of an output is off or full. Below 1 the flank is a smooth fall centred on the edge,
never wider than the line or the gap has room for: the centre of a line stays full, the centre of
a gap stays off, pulse width 0 stays dark and 1 stays solid. At hardness 0 and pulse width ½ the
lines are a sine. Levels between off and full exist only in these flanks.

## Distance and time

Two more things go into every oscillator and are not played.

| Goes in  | Unit    | What it is                                                         |
|----------|---------|--------------------------------------------------------------------|
| position | degrees | a pixel's angle from the person, signed: what the lines run along  |
| time     | seconds | what makes the lines travel: normally the clock                    |

**Position** is each pixel's angle from the person's azimuth, negative on their left. It is the
one thing the person's position feeds: when they walk, the picture goes with them. Mirrored
(`mirror` on, the default), the oscillator takes the position without its sign, so both sides
draw the same and the lines come out of the person, or go in, on both sides. With `mirror` off it
takes the signed position: one pattern across the person, the lines passing behind them, a
positive speed moving every line toward the positive side. To the oscillator a position is a
phase offset: the pixel at the person shows the wave as it is, the pixel one interval out shows
it one cycle late, which is the same. That is how a wave in time becomes lines along the wall.

**Time** enters only through the speed: with the speed at 0 it has no effect and the lines stand
as a row. A voice has one time. Only its steps are used, so the time can run faster or slower,
stand still or run backward, all smoothly.

A **push** is a moment of added speed: each oscillator's own envelope over time (*The envelope*),
opened by a hit at once, that adds the oscillator's `push`, in degrees per second and signed, to
its speed, and falls back over the oscillator's release. The lines
keep the distance they gained and never move back, and standing lines are moved too, since the
push is added to the speed and not multiplied into it. The push changes how fast the lines travel
at once, which is not a step on the projection: where the lines are stays continuous.

The two combine as a difference, the position less what the lines have travelled, because the
lines move outward: a pixel further out shows what a nearer pixel showed a moment before.

## The window

Only a window of the lines shows each side of the person. The window is an envelope over distance
(*The envelope*: rise 0, length the reach, fall the taper's part of it): 1 from the person on,
falling smoothly to 0 over the last part of the **reach**. Every line's pulse width is multiplied
by it, so the lines are alike over most of the window, thin out at its end and stop at nothing. No
line is cut: a cut would leave a last line of a width no parameter asked for. A thin line is faint
(*The rules*), so the pattern fades out with every pixel still off or full.

| Parameter   | Unit    | What it is                                                               |
|-------------|---------|--------------------------------------------------------------------------|
| reach left  | degrees | how far the lines extend on the person's left                            |
| reach right | degrees | how far the lines extend on the person's right                           |
| taper       | 0..1    | the last part of a reach over which the lines thin out: 0.2 by default   |

```
reach at a side       = reach × presence
pulse width of a line = pulse width × window(the line's centre)    the window after the pulse width's slot
```

The window is read once per line, at the line's centre, and not per pixel: read per pixel it
would thin the outer half of a line more than the inner, and a line in the taper would be lopsided
and sit off its place. Read per line, a line in the taper is thinner, whole and still centred. A
solid output is lines that touch, so its solid part ends at a line's boundary within half an
interval of the taper and not exactly at it.

The reach is the one thing that may differ between the two sides of a person: mirrored, the lines
are the same on both, the window need not be. With both reaches equal the picture is symmetric. The
reaches have no slot: they are given to the voice as values, each tick, by whoever uses it. A
reach that has to land on something, as sync's on a partner, cannot be reached through a base and
an amount without reading the patch back, so it is set whole.

The window and **presence** are the amp stage, a synth's amp envelope, and sit after the slots
and not in them, so the pulse width stays free to be connected. Presence is an envelope over time
(*The envelope*) that multiplies both reaches: the window opens from the person when they arrive
and closes to the person when they leave. The pulse width is the most a line can be; the window
only thins it, so a dark output stays dark. A solid output is solid over most of the window and
opens into thinning lines over the taper. A change of reach is smooth: the lines in the taper grow
or thin a little, and nothing appears or disappears.

## The rules

- Each parameter does one visible thing, whatever the others are.
- Every parameter is continuous: a smooth change of a parameter is a smooth change on the
  projection. Nothing appears, disappears or changes width in a step.
- A thin line is faint on the fixture **(site fact)**, and there is no reason not to show it. That
  is how a line arrives and leaves.
- Many small lines do not register as lines **(site fact)**: about 90 per revolution, line and gap
  equal, is the most. This bounds the interval, which never goes below one period of it (4°). It
  does not bound a line's or a gap's width.
- The oscillators add no smoothing and no steps of their own.

## Modulation

Every parameter has a modulation slot, a modulation matrix row, as a synth's parameters have.
What the source is does not matter to the slot. In the panel a slot is a row titled with the
parameter's name and reads Base · Amount · Curve · Bypass, the matrix's own words.

| Part   | What it is                                                                          |
|--------|-------------------------------------------------------------------------------------|
| base   | the parameter's own knob: its value with nothing connected, and the hand's value    |
| amount | how far the source moves the parameter from its base, in the parameter's unit, signed |
| curve  | how the source's magnitude is eased, its sign kept: one of pytweening's easings     |
| source | what is connected: a value 0..1, or −1..1 for an LFO; one per tick or one per pixel |
| bypass | bypassed, the modulation is off: the parameter is its base, and the amount is kept  |

```
parameter = base + amount × curve(source)         amount in the parameter's unit
```

At pulse width base 0.2 and amount 0.6 the lines are 0.2 wide with the source at 0 and 0.8 wide
with the source at 1. At pitch base 25.7 and amount 46.3 the lines go from one every 14° to one
every 5°: a row reads in one unit, and the 90-line limit is the knob's own scale.

- A slot has one source. A source may feed several parameters, each with its own amount.
- The bypass is how one parameter is played by hand while the others follow their sources: a base
  is already the hand's value, so bypassing the modulation is all that is needed, and lifting it
  brings the source back with the amount as it was.
- A source has a synth's ranges: an LFO swings both ways, −1..1, so the parameter moves around its
  base; an envelope and everything else is 0..1 and moves the parameter one way from its base. Any
  source fits any parameter, and the amount alone carries the unit and the direction.
- A per-pixel source on the phase bunches and spreads the lines along the wall, a synth's FM, and
  can take them past the visual limit locally. The synth allows it; whether it is used is the
  instrument's choice.
- Pulse width, phase and hardness take a source per tick or per pixel; pitch and speed take one
  per tick, since the travelled (*The oscillator*) is one count.
- Where a parameter ends: pulse width and hardness stop at 0 and 1, the pitch stops at the
  visual limit, phase wraps, speed has no ends. A stop is a standstill, not a step.
- The slot adds no smoothing and no steps. Its curve is the one shaping it does, so a
  connection's feel is tuned in its row. A curve shapes a value, what a source is worth at each of
  its positions; an envelope shapes time (*The envelope*): two different things that share the
  same easings. The curves are `pytweening`'s: linear, and ten families (quad, cubic, quart,
  quint, sine, expo, circ, back, elastic, bounce) each as ease in (little at first), ease out
  (much at first) and ease in-out. A curve is continuous and odd: 0 and ±1 stay where they are
  and the sign is kept, so a bipolar source stays symmetric. Back and elastic overshoot on the
  way, so a parameter may pass its base or its end for a moment, and bounce turns back on itself;
  pulse width and hardness still stop at 0 and 1. `pytweening`'s functions take one number, so a
  curve is sampled once into a table and read between, which also serves a source per pixel; the
  table's ends are pinned, since `pytweening`'s elastic ends a hair past 1.

### An LFO as a source

An **LFO** is an oscillator used as a source: its output feeds a parameter of another oscillator.
An LFO is not drawn: its output is a sine, −1..1, so what it feeds moves around its base without
a step. It has one parameter more, its **level**, 0..1, which scales its output and has a slot of
its own. At level 0 the LFO is silent and the parameter it feeds is at its base: the level is what
is played, as a synth's mod wheel brings in the vibrato, and an envelope into it is a fade-in.

A voice has one LFO, **in time**: it has one position, so no interval and no speed, and what it
has is a **rate**, in cycles per second, and a phase. Its output is one value per tick, and where
it goes is the wiring's choice (`POSE_INSTRUMENT.md`, *The connections*).
An LFO along the wall, with an interval and a speed and one value per pixel, is not built.

| An LFO         | Into a pulse width                                  | Into a phase                              |
|----------------|-----------------------------------------------------|-------------------------------------------|
| in time        | all the lines breathe together                      | the lines sway about their place          |
| along the wall | the thickness varies smoothly from line to line     | the lines bunch and spread (a synth's FM) |
| both           | a swell of thickness travels through standing lines | a ripple travels through the lines        |

### An envelope as a source

An envelope's output is a value 0..1, so it is a source like any other: over time one value per
tick, over distance one per pixel. Three envelopes act outside a slot: the window and presence
are the amp stage (*The window*), and the push adds to the speed after its slot (*Distance and
time*).

## The oscillator

The oscillators that draw and the LFOs are one building block, reusable elsewhere. It knows
nothing of people, colours, mirroring or the mask: positions and a time step go in, values come
out.

An ordinary LFO is a slow wave over time with a rate and a phase, giving one smooth value now. This
oscillator is the same, with one addition: it is given a set of **positions**, each a phase offset
(*Distance and time*), and gives a value for every one of them, every tick. With one position it
is the ordinary LFO; with a row of positions it is the wave laid out along the row, travelling as
the time runs.

The **core** knows where each position is in the cycle; the **waveform** reads that and makes the
output.

```
travelled += speed × dt / interval             each tick; how far the wave has moved, in cycles
cycle      = position / interval − phase − travelled

sine   value = level · cos(2π · cycle)                    highest at every whole cycle
pulse  on where |frac(cycle + ½) − ½| ≤ pulse width / 2   centred on every whole cycle
```

- Interval, phase and speed are the core's parameters, in the units of the positions. The
  ordinary LFO's rate is a consequence: `speed / interval` cycles per second.
- The travelled is counted in cycles and not in position units, so a change of interval opens the
  wave from position 0 and not from wherever it has travelled to.
- The sine gives a smooth value −1..1 and has one parameter more, the level. The pulse gives on or
  off and has one parameter more, the pulse width.

| Use                       | Positions                   | Waveform | Output                                             |
|---------------------------|-----------------------------|----------|----------------------------------------------------|
| drawing lines             | every pixel's distance      | pulse    | on or off per pixel                                |
| an LFO along the wall     | every pixel's distance      | sine     | a value per pixel, into an oscillator's parameter  |
| an LFO in time            | one position                | sine     | one value per tick, into any parameter             |
| elsewhere                 | whatever the user's axis is | either   | a value per position                               |

The hardness is not part of the core: the pulse waveform applies it as a slew on its output, as a
lag after an LFO softens a square.

```
d     = |frac(cycle + ½) − ½|              the distance from the nearest line's centre, in intervals
edge  = pulse width / 2                    where a hard line ends
ramp  = (1 − hardness) × 2 × min(edge, ½ − edge)
level = full where d ≤ edge − ramp / 2, off where d ≥ edge + ramp / 2, a smooth fall between
```

## The envelope

The second building block, reusable as the oscillator is. An oscillator repeats; an envelope goes
up once, holds and comes down once. It knows nothing of people or colours: positions or time steps
go in, values 0..1 come out, smoothly.

An envelope gives a shape over time to something that has none of its own. A synth needs one for
every note, since a key is only down or up. Here the player is a body, which moves through time by
itself, so a played parameter needs no envelope: the arm's own motion is the shape, and an
envelope on top of it would be a second motion (`POSE_INSTRUMENT.md`, *The instrument*: no
smoothing but the pipeline's). Envelopes exist only where there is an event with no motion of its
own: a person arriving or leaving, the playhead crossing, and the end of the window, an edge given
a shape along the wall. That is why there are three, and why there will be no more unless an event
is added.

| Parameter | What it is                                                                       |
|-----------|----------------------------------------------------------------------------------|
| rise      | how long the way up takes: 0 is at once                                          |
| fall      | how long the way down takes                                                      |
| length    | over positions: where the envelope ends                                          |
| gate      | over time: open or closed, opened and closed from outside                        |

```
over positions   level = position / rise, up to 1; from length − fall on, (length − position) / fall
over time        level moves to 1 at 1 / rise per second while the gate is open,
                 and to 0 at 1 / fall per second while it is closed
value            = ½ − ½ · cos(π · level)          the level eased, so the ends are smooth
```

Over positions the envelope is a shape with a fixed length, in the units of the positions. Over
time it follows its gate, so a gate that closes early turns the rise into a fall from where it is,
without a step. What opens a gate is not part of this document.

| Envelope | Over     | Rise, hold, fall                                       | Acts on                         |
|----------|----------|--------------------------------------------------------|---------------------------------|
| window   | distance | 0; to the reach less the taper; the taper              | the pulse width, after its slot |
| push     | time     | 0; a gate open for a tick; the oscillator's release    | its speed, after its slot; one per oscillator |
| presence | time     | the attack; a gate open while present; the release     | both reaches                    |

## In the pose instrument

The pose instrument is the bridge between the pose data and the synth (`POSE_INSTRUMENT.md`). It
gives every person a voice, sends the voice's two outputs to white and to blue, connects the
body's measures to the parameters, triggers the events (presence, the hit, sync) and draws the
mask.

Where white and blue are both lit the overlap is a tone of its own: the palette is dark, blue,
white, both (`POSE_INSTRUMENT.md`, *The instrument*). How the two meet is a consequence of their
oscillators' parameters, not a parameter of its own:

| The two oscillators                            | On the projection                                   |
|------------------------------------------------|-----------------------------------------------------|
| equal intervals, phases half an interval apart | white and blue alternate                            |
| equal intervals, equal phases                  | they stack into the overlap tone, with dark between |
| unequal intervals                              | they slide past each other with distance            |
| unequal speeds                                 | they slide past each other over time                |

At the person sits the instrument's **mask** (`POSE_INSTRUMENT.md`, *The instrument*), not part of
the synth. The lines come out from under it, and it covers the place where the two mirrored sides
meet.

## Open

- More LFOs than the one in time per voice: one along the wall; one shared by all voices, all
  patterns moving in step
- Whether blue half an interval from white is the instrument's rest, or a preset value
- Both solid (the overlap tone everywhere) and both dark: whether each is a sound of the
  instrument or a silence to avoid
- A pitch's amount in lines is linear in pitch: an LFO on a pitch would swing it further up than
  down to the eye, and one source into two pitches keeps their difference, not their ratio. If a
  pitch is ever given a symmetric or a shared source, the synth's answers are an amount in
  octaves, or detune: blue's pitch set relative to white's, as a ratio, so the two stay tuned
