# White Space — The Pose Instrument's LFOs

The light of the pose instrument is made of low-frequency oscillators. This document describes
that part of the instrument and nothing else: the lines each person's LFOs draw, the inputs they
have, what each input looks like on the projection, and the LFO they are built from. What plays
the inputs is not part of it. The instrument as built today, its meanings and its connections are
in `POSE_INSTRUMENT.md`; this document is the design the instrument is rebuilt from.

## The lines

Each person has two LFOs, one drawn in white and one in blue. An LFO draws a line, a gap, a line,
a gap, outward from the person, the same on both sides. A line is full and a gap is off. The two
LFOs have the same inputs, each its own values.

White and blue are projected separately by the fixture **(site fact)**, so where both are lit the
overlap is a tone of its own: the palette is dark, blue, white, both.

## Inputs

Per colour, five inputs and nothing else. Four are there to be played: each has a base value and
can be connected to a source (*Modulation*). The hardness is a setting: hard by default, and
probably never connected to anything.

| Input       | Unit                 | What it is                                              |
|-------------|----------------------|---------------------------------------------------------|
| interval    | degrees              | the distance from one line to the next                  |
| pulse width | fraction of interval | the thickness of a line: 0 none, 1 solid                |
| phase       | intervals            | where the lines sit relative to the person              |
| speed       | degrees per second   | how fast the lines travel: positive outward, 0 still    |
| hardness    | 0..1                 | the flanks of a line: 1 hard (default), 0 softest       |

**Interval** spaces the lines. A change of interval moves a far line more than a near one, as an
accordion opens from the person.

**Pulse width** is the thickness of every line. At 0 the colour is dark, at 1 it is solid, and
halfway line and gap are equal. A line is `pulse width × interval` wide.

**Phase** places the lines. At 0 a line is centred on the person; at ½ a gap is. A whole interval
further the picture is the same.

**Speed** is how fast the lines move across the projection, outward or inward. It is the same on
the projection whatever the interval, so a change of interval changes the spacing and not the
travel. At interval 10° and speed 2° per second the lines are centred at 0°, 10°, 20°; a second
later at 2°, 12°, 22°; after five seconds at 10°, 20°, 30°, the same picture, a new line having
come out at the person. A **push** makes the lines travel faster for a moment and settle back;
they keep the distance they gained.

**Hardness** shapes the flanks of the lines and nothing else. At 1 a line ends at its edge and
every pixel of a colour is off or full. Below 1 the flank is a smooth fall centred on the edge, as
wide as the hardness allows and never wider than the line or the gap has room for: the centre of a
line stays full, the centre of a gap stays off, pulse width 0 stays dark and 1 stays solid. At
hardness 0 and pulse width ½ the lines are a sine. Levels between off and full exist only in
these flanks.

## Distance and time

Two more things go into every LFO. They are not played: they are what the lines run along, and
what makes them travel.

| Goes in  | Unit    | What it is                                                         |
|----------|---------|--------------------------------------------------------------------|
| distance | degrees | how far a pixel is from the person: what the lines run along       |
| time     | seconds | what makes the lines travel: normally the clock                    |

**Distance** is each pixel's angle from the person's azimuth, without its sign, so both sides of
the person draw the same. It is the one thing the person's position feeds: when they walk, the
picture goes with them. To the LFO a distance is a phase offset: the pixel at the person shows the
wave as it is, the pixel one interval out shows it one cycle late, which is the same, and the
pixels between show everything between. That is how a wave in time becomes lines along the wall.

**Time** only enters through the speed: the lines travel `speed × time`. With the speed at 0 the
time has no effect and the lines stand as a row. It is normally the clock, but only its steps are
used, so the time can run faster or slower, stand still or run backward, all smoothly. The push is
the time running faster for a moment.

The two combine as a difference, the distance less what the lines have travelled, because the
lines move outward: a pixel further out shows what a nearer pixel showed a moment before.

## The two colours

How white and blue meet is a consequence of their inputs, not an input of its own:

| The two LFOs                                   | On the projection                                   |
|------------------------------------------------|-----------------------------------------------------|
| equal intervals, phases half an interval apart | white and blue alternate                            |
| equal intervals, equal phases                  | they stack into the overlap tone, with dark between |
| unequal intervals                              | they slide past each other with distance            |
| unequal speeds                                 | they slide past each other over time                |

## The rules

- Each input does one visible thing, whatever the others are.
- Every input is continuous: a smooth change of an input is a smooth change on the projection.
  Nothing appears, disappears or changes width in a step. A line grows from nothing to solid
  through every width between.
- A thin line is faint on the fixture **(site fact)**, and there is no reason not to show it. That
  is how a line arrives and leaves.
- Many small lines do not register as lines **(site fact)**: about 90 per revolution, line and gap
  equal, is the most. This bounds the interval, which never goes below one period of it (4°). It
  does not bound a line's or a gap's width.
- The LFOs add no smoothing and no steps of their own.

## Modulation

The instrument is a light synth, and its inputs are connected as a synth's are. Every input has a
modulation slot: what the input is by itself, what is connected to it, and how far that moves it.
What the source is does not matter to the slot.

| Part   | What it is                                                                          |
|--------|-------------------------------------------------------------------------------------|
| base   | the input's value with nothing connected                                            |
| source | what is connected: a value 0..1, one per tick or one per pixel                      |
| amount | how far the source moves the input from its base, in the input's unit, signed       |

```
input = base + amount × source
```

With amount 0, or nothing connected, the input is its base. At pulse width base 0.2 and amount
0.6 the lines are 0.2 wide with the source at 0 and 0.8 wide with the source at 1, and everything
between is smooth.

- An input has one source. A source may feed several inputs, each with its own amount.
- A source is always 0..1, so any source fits any input and the amount alone carries the unit and
  the direction.
- A source is one value per tick, or one value per pixel. Pulse width, phase and hardness take
  either; interval and speed take one value per tick, since the travelled is one count.
- Where an input ends: pulse width and hardness stop at 0 and 1, the interval stops at the visual
  limit, phase wraps, speed has no ends. A stop is a standstill, not a step.
- The slot adds no smoothing, no curve and no steps. A source that needs shaping is shaped in the
  source.

### An LFO as a source

An LFO's output can be the source of another LFO's input. A modulating LFO is not drawn: its
output is a smooth value, a sine, so what it feeds changes without a step.

| A modulating LFO | Into a drawn LFO's pulse width                                       |
|------------------|----------------------------------------------------------------------|
| along the wall   | the lines' thickness varies smoothly from line to line               |
| in time          | all the lines breathe together                                       |
| both             | a swell of thickness travels through the lines, which stay in place  |

## The LFO

The drawn and the modulating LFOs are one LFO, used in different ways; it can be used elsewhere
too. It knows nothing of people, colours, mirroring or the mask: positions and a time step go in,
values come out.

An ordinary LFO is a slow wave over time with a rate and a phase: it gives one smooth value now,
and that value moves something else. This LFO is the same, with one addition: it is given a set of
**positions**, and gives a value for every one of them, every tick. A position is a phase offset:
a position one interval further shows the wave one cycle late, which is the same. With one
position it is the ordinary LFO. With a row of positions it is the wave laid out along the row,
and as the time runs the wave travels along it.

It has two parts. The **core** only knows where each position is in the cycle. The **waveform**
reads that and makes the output.

```
travelled += speed × dt / interval             each tick; how far the wave has moved, in cycles
cycle      = position / interval − phase − travelled

sine   value = ½ + ½ · cos(2π · cycle)                    highest at every whole cycle
pulse  on where |frac(cycle + ½) − ½| ≤ pulse width / 2   centred on every whole cycle
```

- **Interval**, **phase** and **speed** are the core's inputs, in the units of the positions. The
  ordinary LFO's rate is a consequence: `speed / interval` cycles per second.
- **Time** enters only as `dt`, the step since the last tick (*Distance and time*).
- The travelled is counted in cycles and not in position units, so a change of interval opens the
  wave from position 0 and not from wherever it has travelled to.
- The **sine** gives a smooth value 0..1. The **pulse** gives on or off, and has one input more,
  the pulse width: the on part of a cycle.
- Interval and speed are one value per tick. Phase and pulse width may be one value, or one per
  position, which is how a modulating LFO along the wall feeds them.

| Use                       | Positions                   | Waveform | Output                                      |
|---------------------------|-----------------------------|----------|---------------------------------------------|
| drawing lines             | every pixel's distance      | pulse    | on or off per pixel                         |
| modulating along the wall | every pixel's distance      | sine     | a value per pixel, into another LFO's input |
| modulating in time        | one position                | sine     | one value per tick, into any input          |
| elsewhere                 | whatever the user's axis is | either   | a value per position                        |

What the instrument adds around it:

- The positions are the **distances** of *Distance and time*, in degrees, so the interval is in
  degrees and the speed in degrees per second.
- The pulse's on part is a line, its off part a gap.
- The hardness is not part of the LFO: it is a slew on the pulse's output, as a lag after an LFO
  softens a square.

```
d     = |frac(cycle + ½) − ½|              the distance from the nearest line's centre, in intervals
edge  = pulse width / 2                    where a hard line ends
ramp  = (1 − hardness) × 2 × min(edge, ½ − edge)
level = full where d ≤ edge − ramp / 2, off where d ≥ edge + ramp / 2, a smooth fall between
```

## Open

- Which modulating LFOs the instrument has, and what each feeds
- Whether blue half an interval from white is the instrument's rest, or a preset value
- Both solid (the overlap tone everywhere) and both dark: whether each is a sound of the
  instrument or a silence to avoid
- Where the pattern ends each side of the person: the window's cut is a step and has to go
