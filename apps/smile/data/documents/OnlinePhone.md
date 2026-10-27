# SMILE Online: the phone

What a visitor's phone does during SMILE Online, as the visitor sees it (section 1) and as the machine does
it (section 2). It is the reference for the phone side: the relay (`relay/`), SMILE's online side
(`modules/online/`) and the page (`relay/web/`). The setup is in the README's SMILE ONLINE section and in
`relay/README.md`.

## 1. The interaction

A visitor outside gets the room's eye when nobody inside has it and goes through the same sequence as a
visitor inside. The phone shows the grid of the newest portraits and, in its middle, a window: the visitor's
camera until the flash, the installation's main screen after it. The page explains nothing and shows no
control before the thing it controls exists.

**Opening the page.** The grid of stills is there, built once from the relay's list; it does not change while
the page is open. The window shows one button, SMILE. The line under the window says what the eye is doing:
looking for faces, looking at someone inside, warming up, busy inside, asleep when SMILE is not there.

**Consent.** The button opens the terms; one tap on I AGREE TO SMILE unlocks the sound and opens the camera.
Without a camera the line says so and the page goes no further.

**Waiting.** The window shows the visitor's own camera, mirrored, black and white, in coarse pixels, coarser
the further back in line; the line under it says how many are before them. The room's laughter plays, a laugh
every second or two, left and right, while nothing else sounds.

**The turn.** The window is the visitor's live image, mirrored, black and white. It is in pixels while the
face is not right (too far, too near an edge, turned away, eyes shut), coarser the worse, and sharp when the
face is right. Nothing is drawn on the image. The page behind the grid goes white as the charge fills,
quickly at first and slowly over the last stretch, like the ring light in the room, and falls again when the
face is lost. A turn in which no face qualifies within [the window](#the-judging) passes and the visitor goes
to the back of the line ("missed you"); the same when the connection drops ("lost you") or the operator
resets.

**The flash.** The visitor's own image goes white in the room's flash-in curve while the track starts from
standstill: it spins up with the wobble of the room's tape, muffled, the pitch rising. At full white the
window becomes the main screen of the installation and the white fades as the room's flash fades. From here
the visitor watches the room: the capture coming out of the flash, the generation steps, the grin, the
portrait standing after it, the bloom. The caption, the artist's name and the visitor's line, fades in with
the first generated image, as on the main screen. The track opens up through the transformation and runs at
full speed through the grin. The laughter is silent through the turn.

**The end.** The room's picture stops at the bloom (the fade would show the room's camera), the portrait
stands, the track plays on. When the visitor's own video is on the phone the track winds down to a stop and
the video takes over: the grin, looping, with its own sound (the track on the tape, spun up at the start and
wound down at the end) and a small watermark bottom right, a circled "ai" and the artist's name. The video
has no caption. The share button appears with the file and opens the share sheet at once, with the file and
the text (the name, the line, the tags). A browser that will not start the video's sound by itself plays it
silent; a tap on the video turns the sound on.

**After.** The grin loops. The visitor closes the page or reloads it for another turn.

**The room meanwhile.** An online turn takes the room's camera and nothing else: the main screen shows the
phone's image and the sequence as always, the music and the lights follow it, the wall gets the clip. A reset
ends the turn. A turn starts only when the room has had no face for a few seconds and the generator is
ready, so whoever is inside keeps the eye.

## 2. The machine

### The parts and the messages

The **relay** serves the page, keeps the line, forwards messages and holds the files; it decides nothing.
**SMILE** (`Online`, stepped once per render frame from `Main.sequence_callback`) decides every turn's start
and end. The **page** draws and sounds and keeps no state the other two do not know.

The messages of a turn, in order, as the page sees them (`relay/server.py` has the protocol):

| From  | Message                     | When                                | On the page                          |
|-------|-----------------------------|-------------------------------------|--------------------------------------|
| relay | `state`                     | every change of the line or the eye | the status line                      |
| relay | `turn {window}`             | SMILE took this phone               | state `you`, the judging starts      |
| SMILE | `music {id, url}`           | the turn's start                    | fetch and decode the track           |
| SMILE | `face {charge, faces}`      | every detector frame                | the pixels and the white             |
| SMILE | `stage flash {in, out}`     | the flash starts                    | own image to white, the track starts |
| SMILE | `stage {name, progress}`    | every change, else every 0.25 s     | the white and the tape follow        |
| SMILE | binary `SHOW` frame         | flash to fade, one in flight        | the window, from full white on       |
| SMILE | `result {phone, grin, ...}` | the files are built                 | download the video                   |
| SMILE | `turn_end {done}`           | the sequence is back at the camera  | wind down into the video             |
| page  | `seen`                      | every `SHOW` frame received         | the relay sends the next frame       |
| page  | `timing`                    | the video plays, the file is here   | the relay logs it                    |

A phone that is not the turn's gets `state` only, and the grid once at load.

### The judging

The page sends its camera frames to SMILE (640 x 800 JPEG, 15 a second, mirrored as the phone shows them,
so the room and the portrait match the phone) and SMILE's face finder judges them as it judges the room's.
The page draws its own live video and takes two numbers from each `face` message: the subject's distance
from good (1 without a face), which sets the pixel size of the whole image, and the charge, which sets the
white as `1 - (1 - charge)²`. The live image never lags; the judgement does. A turn in which no face
qualifies within `online.turn_window` (10 s) passes.

### The flash and the stream

At `stage flash` the page keeps drawing its own image, sharp, and whitens it with the room's flash-in
progress; at full white it switches to the stream, so the room's face marks are never shown. The stream is
the main screen as rendered, taken in `SmileRender._render_credit` before the status line and from a credit
pass without the QR code (`ScreenGrab`; `online.stream_width`, 4:5, `online.stream_fps`; JPEG at
`Link.STREAM_QUALITY`, encoded on the link's thread). The relay keeps one frame in flight per phone: the next
goes when the phone has answered `seen`, or after `FRAME_PATIENCE`, and the newest replaces one still
waiting. A slow phone so gets fewer frames, never late ones, and the messages behind the frames never queue.
While the result stands (the timeout) the stream runs at `Online.STILL_FPS`, 2 a second, because the phone is
downloading its video then. The frames stop at the fade.

### The sound

Everything goes through one `AudioContext`, created on the consent tap: the track through source → low-pass
filter → gain → destination, the laughs as buffer sources with a stereo pan. The track is not played through
a media element: browsers silence one at the tape's low speeds, and Safari refuses some MP3s **(site fact)**.

- **The track** is the room's: SMILE picks the visitor's style and track at the turn's start
  (`Main.pick_style` from `Online._begin_turn`) and names it in `music`. The relay holds the tracks as AAC;
  SMILE uploads one, transcoded, the first time it is used (`Link._track_aac`). The page fetches and decodes
  it during the judging; one that arrives after the flash starts that far in (`offset`).
- **The tape** follows the room's stage and progress with the Max patch's curves (`p speedVar`,
  `p musicMixer`): flash-in, the speed from 0 along the wobble and the filter closing; flash-out and
  diffusion, the filter opening; grin and after, full speed and open. A speed under 0.3 (`STANDSTILL`) is
  standstill: the source is muted, not slowed to 0. The filter never goes under 1.2 kHz (`PHONE_FLOOR`) and its
  resonance is 2.5 (`PHONE_Q`): a phone speaker has nothing under that, and the patch's resonance of 11 is a
  boom it cannot make. The bloom and the fade do not wind the track down on the phone; it plays on until the
  video is here, then winds down over 1.2 s (`windDown`) and stops.
- **The laughter**: the room's samples, uploaded once to the relay and listed by `/api/library`, one every
  0.8 to 2.5 s at a random pan while there is no track, no video and no turn running, following the panel's
  laughter switch (`laughter` in `state`).
- **The video's sound** is in the file.

### The video

The recording closes when the last grin frame is saved (`Main.diffusion_callback`), while the grin still
plays on the screens: the last frame exists before the sequence ends (see [Time](#time)). The saver builds the
phone's clip first (`modules/save/share.py`: the grin re-encoded lighter, the track from the second the room
heard at the grin's start, on the tape, spun up over its first half second and wound down over its last
second, the watermark, no caption) and lands the players' silent clip after it. `Online` uploads the three
files as soon as they exist and sends `result`, addressed to the phone, so a phone finds it after its turn as
well. The page downloads the video once, alone (nothing else is fetched meanwhile), plays it from that
download after the wind-down and uses the same bytes for the share sheet; the completed download is what
shows the share button. The video element is shown only once it plays (it is black before), so the last
frame of the screen stands until then. A result that has no files `Online.SAVE_TIMEOUT` (20 s) after the turn
ends the turn without a video.

### Time

Measured 2026-10-26: preset `Smile` (250 grin frames at 12 fps; stream 512 px wide at 10 fps), Quick Sync
encoder, this PC as room and relay, the phone over Tailscale on a direct connection (`tailscale status`).

| Step                                        | Time       |
|---------------------------------------------|------------|
| portrait and grin frames, ComfyUI           | 34 to 35 s |
| recording closed after the last grin frame  | under 1 s  |
| share clip built                            | 1.3 s      |
| result on the phone, before the turn's end  | 2.4 s      |
| video playing, after the turn's end         | 1.5 s      |

### Logs

- `relay/data/relay.log`: SMILE and phones connecting; per turn `phone … timing: result, play, playing,
  file` (seconds after the turn's end) and `frames to the phone: n, flight median, max`; the access log of
  every file served.
- SMILE's log (`output/logs/smile_*.log`): `turn n starts`, `turn ends (reason)`, the track uploads, the
  saver's `Saved`, warnings.
- The page with `?debug` in its address: its state, the audio context, the track (named, decoded, playing,
  error), the tape's rate, gain and filter, the video.

## Open

- The still at the grin's peak for the grid; the portrait stands in.
- The caption for the outside and the QwenVL line on the machine.
- The QR screen's stream in the players.
- The capture at the phone's full resolution.
- The page on its own domain; the relay on the VPS.
