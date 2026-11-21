# Inout

The network output and input the apps share: OSC out to the sound and the show's other programs, OSC in from
them, and Art-Net out to the lights.

## The parts

| Part           | File               | What it does                                                              |
|----------------|--------------------|---------------------------------------------------------------------------|
| `OscSound`     | `osc_sound.py`     | the pose data, one bundle per pose tick, to the sound (a Max patch)       |
| `OscSender`    | `osc_sender.py`    | what its owner queues, one bundle per tick at `fps`                       |
| `OscClient`    | `osc_client.py`    | the UDP socket under both senders                                         |
| encoder        | `osc_encoder.py`   | a message's and a bundle's bytes                                          |
| `OscReceiver`  | `osc_receiver.py`  | calls what is bound to an address; can echo each message back             |
| `ArtNetSender` | `artnet_sender.py` | its owner's DMX packet to one or more consecutive universes               |

| App         | Sends                                              | Receives                     |
|-------------|----------------------------------------------------|------------------------------|
| hd_trio     | `OscSound`; `ArtNetSender` ×3 (`ArtNetBars`)       | `OscReceiver` (session)      |
| deep_flow   | `OscSound`                                         | –                            |
| low_flow    | `OscSound`                                         | –                            |
| white_space | `OscSound` with `SoundExtras`                      | `OscReceiver` (sound levels) |
| smile       | `OscSender` (sound, walls); `ArtNetSender` (lights) | –                            |

deep_flow and low_flow hold an `osc_control` receiver group that no receiver is built from **(seen)**.
white_space's light output (`OscLightSender`, `UdpLightReceiver`) is its own: the fixture firmware's wire contract
does not fit a bundle sender.

## OSC out

Both senders build their messages, encode them (`osc_encoder.py`) and hand them to an `OscClient`, which sends them
as one bundle with the time tag "immediately". The client binds its owner's `ip` and `port` at start and unbinds them
at stop; an invalid address keeps the current one. A failed send is logged, at most once every 10 s, and the next one
goes out: a sender never stops for the run.

The encoder writes an int as `i`, a float as `f`, a str as `s` and a float array as one `f` per value, from
`astype('>f4')`: the bytes python-osc wrote for the same messages (`tests/test_osc_encoder.py`). A bool is refused; an
app sends it as an int.

### The sound

`OscSound` sends a bundle for each frame dictionary of its `stage`. It sends `slot_count` slots (`max_players` unless
the app passes more); the per-player rows stay `max_players` wide. An app adds its own messages through
`OscSoundExtras`: globals, a present player's messages, an empty slot's messages and the angle values. Switched off
(`enabled`), it sends one blackout and then nothing; stop sends a blackout too. The blackout is every global at 0
(`/global/state` −1, so no stage's cue fires) and every slot's reset.

| Address                         | Arguments              | Value                                                   |
|---------------------------------|------------------------|---------------------------------------------------------|
| `/global/state`                 | int                    | the sequencer's stage; −1 in the blackout               |
| `/global/state/progress`        | float                  | the stage's progress, 0..1                              |
| `/global/progress`              | float                  | the show's progress, 0..1                               |
| `/pose/{id}/active`             | int                    | 1 present, 0 for the first two ticks after it left      |
| `/pose/{id}/bbox`               | 4 floats               | center x, center y, width, height; NaN without one      |
| `/pose/{id}/time/motion`        | float                  | motion time, about 1 a second while moving              |
| `/pose/{id}/time/age`           | float                  | the track's age in seconds                              |
| `/pose/{id}/angle/rad`          | 9 floats               | the angles, −π..π, in `AngleLandmark` order             |
| `/pose/{id}/angle/vel`          | 9 floats               | the angles' velocities                                  |
| `/pose/{id}/angle/sym`          | floats                 | left minus right per pair, −1..1                        |
| `/pose/{id}/angle/legs`         | float                  | leg deviation, 0 standing, 1 bent                       |
| `/pose/{id}/angle/tilt`         | float                  | torso tilt, −1 left, 1 right                            |
| `/pose/{id}/similarity/pose`    | `max_players` floats   | pose similarity to each player, 0..1; NaN sent as 0     |
| `/pose/{id}/similarity/gate`    | `max_players` floats   | the motion gate per player, 0..1                        |
| `/pose/{id}/similarity/motion`  | `max_players` floats   | the same row as `similarity/pose`                       |
| `/pose/{id}/similarity/leader`  | `max_players` floats   | −1 this one leads, 1 the other leads                    |
| `/pose/{id}/similarity`         | `max_players` floats   | the same row as `similarity/pose`, deprecated           |

An empty slot's reset sends every address of a player with zeros, the row lengths a player sends. white_space's own
addresses are in its [SOUND.md](../../apps/white_space/docs/SOUND.md).

### The sender

`OscSender` sends what was queued since its last tick, on a fixed-rate grid at `fps`: of a state address only the
newest value, of an event address (`events`) every message. An event goes out at once, with what else is queued. A
tuple or list goes as separate arguments, which Max takes where it would not take an OSC array. Nothing is queued
while it is switched off or not running.

## OSC in

`OscReceiver` serves `port_in` and calls the callbacks bound to an address, a `utils.Broadcast` each, on the server's
thread: a failing callback is logged and the next still runs. With `return_messages` it echoes each message to
`ip_out`:`port_out`. A port change restarts the server with the same bindings. An address nothing is bound to is
counted and warned about once.

## Art-Net out

`ArtNetSender` calls its owner's `fill` before every tick, on its own thread, and sends the packet's part for each
universe as an ArtDmx packet to port 6454, on one socket; a failed send is logged, at most once every 10 s. It sends a
universe when it differs from what that universe last sent, and at least every 0.9 s: Art-Net 4 has a controller
resend an unchanged universe every 800 to 1000 ms **(site fact)**. It ticks at `fps`, at most 44 a second, the most a DMX universe
refreshes. A blackout goes out at start, when switched off and at stop. With the flash on (smile), a flash, all
channels full for one tick, goes before each blackout and when it is switched on or readdressed; hd_trio's bars run
without it.
