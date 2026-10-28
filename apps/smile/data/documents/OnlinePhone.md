# SMILE Online: the phone

What a visitor's phone does during SMILE Online, as the visitor sees it (section 1) and as the machine does
it (section 2). It is the reference for the phone side: the relay (`relay/`), SMILE's online side
(`modules/online/`) and the page (`relay/web/`). The setup is in the README's SMILE ONLINE section and in
`relay/README.md`.

## 1. The interaction

A visitor outside gets the room's eye when nobody inside has it and goes through the same sequence as a
visitor inside. The phone shows one portrait frame, as the installation's main screen: the visitor's camera
until the flash, the installation's main screen after it. Above it the title of the machine, SMILE, in the
caption's comic font, with the ring of the share video's watermark, the "ai", raised beside it like a (R) and
standing out a little above the letters; under the frame, flush with its right edge, the artist's site,
oostrik.net. Behind everything the page shows the same picture as a field of smileys, black and white: a dot
where it is dark, a ring, a face with eyes and a smile where it is light, breathing slowly: the camera, then the
screen, then the video. The page explains nothing and shows no control before the thing it controls exists.

**Opening the page.** The page asks for the camera as it opens and shows it in the frame, in coarse pixels,
with the terms over it (the heading and the list each sized so their longest line spans the frame, `fitTerms`);
the bar under the frame is white and says I AGREE TO SMILE: the one tap is the consent,
it unlocks the sound and joins the line, and the bar goes black for the wait. A page that got no camera
as it loaded (a prompt dismissed by a tap elsewhere is a refusal on iOS) asks once more inside the tap; without
a camera then, the terms go, the bar says so and the page goes no further. The page is a portrait thing, like the room's screen:
a phone turned on its side sees TURN YOUR PHONE and nothing else until it is turned back (a desktop window
gets the column fitted to its height). The bar is the page's one button and one
line of text, there from the start: the agree button (white), then black with white text the wait (WAITING FOR SMILE, the number before you, SMILE IS
ASLEEP when SMILE is not there: made on the page from the state's `place` and `eye`; the relay's own `status`
sentence is not shown), then what the face still needs, then, once the flash's white is full, TURN YOUR VOLUME UP (the track has just started, and a phone on silent would otherwise
never know there is something to hear; VOLUME grows now and then, the words beside it moving aside and the bar growing with it), from the generation on BULLSHITTING
over the loading bar (poops fly out of the word in every direction, two or three at a time, and the letter jolts as they leave it, `flyPoops`), PREPARING YOUR SMILE over the timeout, then SHARE YOUR SMILE. Wherever a text names the machine (the title,
the terms, the bar) SMILE carries its mark, the ring with "ai" in it.

**Waiting.** The window shows the visitor's own camera, mirrored, black and white, in coarse pixels, coarser
the further back in line; the bar under it says how many are before them. The room's laughter plays, a laugh
every second or two, left and right, while nothing else sounds.

**The turn.** The window is the visitor's live image, mirrored, black and white. It is in pixels while the
face is not right (too far, too near an edge, turned away, eyes shut), coarser the worse, and sharp when the
face is right. Nothing is drawn on the image; the bar under the frame says what the face still needs (SHOW ME
YOUR FACE, COME CLOSER, INTO THE MIDDLE, LOOK STRAIGHT AT ME, CHIN UP or DOWN, HEAD STRAIGHT, OPEN YOUR EYES,
and HOLD IT when it is right). The page behind the frame goes white as the charge fills,
quickly at first and slowly over the last stretch, like the ring light in the room, and falls again when the
face is lost. A turn in which no face qualifies within [the window](#the-judging) passes and the visitor goes
to the back of the line ("missed you"); the same when the connection drops ("lost you") or the operator
resets.

**The flash.** The visitor's own image goes white in the room's flash-in curve while the track starts from
standstill: it spins up with the wobble of the room's tape, muffled, the pitch rising. At full white the
window becomes the main screen of the installation, and shows the room's flash out as the room draws it: the
capture appearing from the white by its depth, while the page behind the frame goes dark with the same curve.
From here the visitor watches the room: the generation steps, the grin, the
portrait standing after it. The caption, the artist's name and the visitor's line, fades in with
the first generated image, as on the main screen. The track opens up through the transformation and runs at
full speed through the grin. The laughter is silent through the turn.

**The end.** When the portrait has stood its time the caption fades out and the portrait stands, without
the room's bloom and fade (the fade would show the room's camera), the track plays on. When the visitor's
own video is on the phone the track winds down to a stop and
the video takes over: the grin, looping, with its own sound (the track on the tape, spun up at the start and
wound down at the end) and a small watermark bottom right, SMILE, the circled "ai" and by the artist's name. The video
has no caption. The bar under the frame says BULLSHITTING from the flash on and fills once with every stage of
the animation; once the file is here it reads SHARE YOUR SMILE and is the button: it opens the share sheet at once, with the file,
named `smile_by_matthias_oostrik.mp4` (the credit's name), and the text (the name, the line, the tags). A browser that will not start the video's sound by itself plays it
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
| relay | `state`                     | every change of the line or the eye | the bar's text (from `place`, `eye`) |
| relay | `turn {window}`             | SMILE took this phone               | state `you`, the judging starts      |
| SMILE | `music {id, url}`           | the turn's start                    | fetch and decode the track           |
| SMILE | `face {charge, faces}`      | every detector frame                | the pixels and the white             |
| SMILE | `stage flash {in, out}`     | the flash starts                    | own image to white, the track starts |
| SMILE | `stage {name, progress}`    | every change, else every 0.25 s     | the white and the tape follow        |
| SMILE | `… flash_in/out {value}`    | the flash stages                    | the white, as the room eases it      |
| SMILE | binary `SHOW` frame         | flash to the result's end, one in flight | the window, from full white on  |
| SMILE | binary `LAST` frame         | the result has stood its time       | fade to it, keep it                  |
| SMILE | `result {phone, grin, ...}` | the files are built                 | download the video                   |
| SMILE | `turn_end {done}`           | the sequence is back at the camera  | wind down into the video             |
| page  | `seen`                      | every `SHOW` frame received         | the relay sends the next frame       |
| page  | `timing`                    | the video plays, the file is here   | the relay logs it (with the download) |

A phone that is not the turn's gets `state` only, and `grid` (the newest stills) once at load, which the page
ignores: the relay keeps and serves the stills (`/api/grid`), the page shows none.

### The judging

The page sends its camera frames to SMILE (640 x 800 JPEG, 15 a second, mirrored as the phone shows them,
so the room and the portrait match the phone) and SMILE's face finder judges them as it judges the room's.
The page draws its own live video and takes two numbers from each `face` message: the subject's distance
from good (1 without a face), which sets the pixel size of the whole image, and the charge, which sets the
white as `1 - (1 - charge)²`. The live image never lags; the judgement does. A turn in which no face
qualifies within `online.turn_window` (10 s) passes.

### The flash and the stream

At `stage flash` the page keeps drawing its own image, sharp, and whitens it with the room's flash-in
`value` (the eased progress, the white as `FlashStage` draws it); at full white it switches to the stream, so
the room's face marks are never shown. The flash out is in the stream (the capture appearing from the white by
its depth, `RevealStage`): the page draws no white of its own over it, and the white behind the frame goes
down with `1 - value`. The stream is
the main screen as rendered, taken in `SmileRender._render_credit` before the status line and from a credit
pass without the QR code (`ScreenGrab`; `online.stream_width`, 4:5, `online.stream_fps`; JPEG at
`Link.STREAM_QUALITY`, encoded on the link's thread). The relay keeps one frame in flight per phone: the next
goes when the phone has answered `seen`, or after `FRAME_PATIENCE`, and the newest replaces one still
waiting. A slow phone so gets fewer frames, never late ones, and the messages behind the frames never queue.
While the result stands (the timeout) the stream runs at `Online.STILL_FPS`, 2 a second, because the phone is
downloading its video then. The frames stop at the end of the timeout with the turn's last frame: the render takes
it at once and without the caption (`ScreenGrab.last`, `clean`), it travels as `LAST`, and the page fades
from the frame it shows to it over the caption's fade time (`CAPTION_FADE`, `CreditLayer.FADE_S`) and keeps
it; the room's bloom and fade are not shown (the bloom would brighten the standing picture, the fade shows
the room's camera). A last frame that never came stops the frames at the fade.

### The halftone

The background is one WebGL draw a frame (`drawHalftone`): the picture the frame shows (the camera canvas, the
stream's canvas, the video) goes up as a texture and the fragment shader draws one smiley per cell of `CELL`
css pixels, its radius from the brightness there (a dot under 0.3, a ring, eyes and a smile over 0.5), white on
black at 40 %. Nothing per cell on the CPU, so a weak phone keeps up; a browser without WebGL, or one that loses
the context, gets a black background and the same page otherwise.

### The bar

The share button under the frame is the page's one button and one line of text (`setLabel`; there is no
status line in the frame). Under the terms it is full and says I AGREE TO SMILE (`agree`, the one tap). Before
the turn it shows the wait (`lineText`, from the relay's state). During the judging it says
what the face still needs (`faceHint`: the first word of the `reject` label of the subject, or of the largest
face when none qualifies, mapped to a line; the pitch's sign picks CHIN UP or CHIN DOWN), black with white
text. Its ripple starts once the camera is allowed. The bar is empty through the flash. The diffusion and the grin
are one bar (`stageProgress`): the diffusion fills it to 0.1, and whenever its progress then stands still (the portrait waits for the grin:
the room's player holds it until the frames come in fast enough, a wait the page cannot know, and no stage is
sent meanwhile) the bar creeps on from where it is toward 0.25, slower and slower, on the page's own clock,
never standing still; the grin takes it from there to full. The timeout is a bar
of its own (the fill jumps back, `snap`) that says PREPARING YOUR SMILE, still, without the poops; full from
the bloom on; it says TURN YOUR VOLUME UP once the white is full, for 5 s at least, and BULLSHITTING from
the diffusion on, the switch waiting for VOLUME to be back at its size (`whenLoudStill`). The text seen is two
copies of the label laid over the bar, the white one clipped to the black part and the black one to the fill
(no blending), so the poops can fly out from behind the letters. Once the video is on the phone it reads SHARE YOUR SMILE and works
(`setShare`). All the page's text is in the caption's font.

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
download after the wind-down and uses the same bytes for the share sheet, as a file named `smile_by_<name>`
from the credit's name (`fileName`), with the file's extension; the completed download is what shows the
share button. The video element is shown only once it plays (it is black before), so the last
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
  file` (seconds after the turn's end), then the video's download as the phone saw it (`first byte`, its size,
  the KB received at every second), and `frames to the phone: n, flight median, max`; the access log of every
  file served.
- SMILE's log (`output/logs/smile_*.log`): `turn n starts`, `turn ends (reason)`, the track uploads, the
  saver's `Saved`, warnings.
- The page with `?debug` in its address: its state, the audio context, the track (named, decoded, playing,
  error), the tape's rate, gain and filter, the video, and the download (KB so far of the total, seconds,
  the first byte).

## Open

- The still at the grin's peak (the relay keeps a still per turn; the portrait stands in).
- The caption for the outside and the QwenVL line on the machine.
- The QR screen's stream in the players.
- The capture at the phone's full resolution.
- The page on its own domain; the relay on the VPS.
