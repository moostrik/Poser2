# Video

The video package writes and reads clips through ffmpeg. Its root holds ffmpeg itself, a file per subject: the
process and its pipes (`process.py`), the chips (`hardware.py`), the encoders (`encoder.py`), what ffprobe says
(`probe.py`), the clip writer (`clip_writer.py`), the clip decode (`clip_decode.py`) and audio (`audio.py`).
`stream/` holds the camera recordings' streams (`stream_writer.py`, `stream_reader.py`, `depth_frame.py`); `screen/`
the screen recorder (`screen_recorder.py`, `screen_format.py`, `screen_frame.py`, `screen_frame_store.py`,
`screen_size.py`). This document holds the streams' and the screen recorder's design; the screen capture, which
hands the recorder its frames, is render's ([SCREEN_RECORDING.md](../render/general/screen_recording/SCREEN_RECORDING.md)).

Every process is started with `popen`: the apps run under pythonw, so it is started with CREATE_NO_WINDOW and
without inheriting stdin or stderr, which are invalid handles there. stderr is a pipe read as it comes
(`ErrorReader`), so a failure has a message and a process that writes a line per frame never fills it. A process
whose output fits in memory is run to its end with `run`, which kills it after its timeout.

`probe` is the one question to ffprobe: a video stream's size, frame rate and exact frame count (the count an mp4
stores, else the packets counted, which reads an mkv or a raw stream without decoding it) and the clip's comment
(`StreamInfo`).

`hardware.py` names the chips video is encoded and decoded on (`Hardware`: CPU, iGPU, GPU, and AUTO, to encode on
the first whose trial passes) and the codecs (`ClipCodec`); what ffmpeg calls an encoder follows from the two
(`encoder_name`), and a hardware decode from the chip (`hwaccel_args`): d3d12va on the chip's DXGI adapter, which
decoded a recording's chunk (2999 frames, 992 × 800 H.264) on the iGPU in 6.3 s against d3d11va's 8.3 s **(measured
2026-11-20)**. The player decodes on the iGPU: the CPU decodes a chunk faster (1.2 s), but the player needs only
50 frames a second, and in deep_flow `-sim` both kept it on time while the CPU cost about half a core more
**(measured 2026-11-20, 20 s at 50 fps)**:

| Player's decoder | Frames late | Frame p95 (ms) | System CPU | Ticks over 25 ms |
|------------------|-------------|----------------|------------|------------------|
| iGPU (d3d12va)   | 0 of 1000   | 22.5           | 70.3 %     | 43 of 1195       |
| CPU              | 0 of 1000   | 22.5           | 72.7 %     | 50 of 1197       |

A camera recording's color part is an `.mkv` whatever its codec: a raw `.hevc` has no timestamps, and
the merge into the chunk refuses it.

The chip a clip is encoded on is picked by a one-frame trial encode (`encoder_works`, `pick_encoder`): the encoder
list alone does not prove the driver works. Audio is transcoded or read by one `run` each (`audio.py`); AAC
(`transcode_aac`) because every phone decodes it, where Safari refuses some MP3s, at about a third of the MP3's size.

Every video file is written by `ClipWriter`: raw frames of one pixel format (`ClipInput`) through the 8 MB pipe into one
ffmpeg process, its error lines read while it writes, with the output options of one encode: `h264_encode` (smile's
clips), `master_encode` (the screen recorder's HEVC master, below) or `stream_encode` (a camera recording's stream,
live: the codec on its chip, its constant quality, an exact keyframe interval, no B-frames where
[Depth](#depth) says).
`StreamWriter` feeds one from a queue on its own thread: color as BGR, NV12 or gray, depth as P012 ([Depth](#depth)).
It writes every frame it is given, in order: a stream of a recording must hold every frame its other streams and its
metadata hold, so whatever is dropped is dropped whole, by the recorder, before any part of it is written; the bytes
it holds are the recorder's bound. The chip each
writer encodes on is its own, so the camera records on the iGPU while the screen recorder encodes on the GPU. In
deep_flow live (the camera at 50 fps recording its color and depth on the iGPU, a 15 s UHD 4:4:4 screen take held and
saved on the GPU meanwhile, 60 Hz with v_sync) **(measured 2026-11-20)**:

| Phase                         | Ticks | Mean (ms) | p95 (ms) | Max (ms) | Over 25 ms |
|-------------------------------|-------|-----------|----------|----------|------------|
| Idle                          | 181   | 16.66     | 20.2     | 30.3     | 3          |
| Camera recording              | 180   | 16.69     | 20.6     | 31.3     | 3          |
| Camera and the screen take    | 905   | 16.66     | 20.1     | 42.2     | 16         |
| Camera and the screen save    | 1473  | 16.70     | 22.2     | 34.6     | 30         |

The chunk held 2285 color and 2285 depth frames against 2285 metadata lines, at 50 fps, with nothing dropped or
missed; the screen take dropped no frame and saved in 24.6 s.

A camera recording's stream is read by `StreamReader`: one decoder started with `popen`, its frames
through the 8 MB pipe, its error lines logged until the reader ends the decoder itself (at its end frame, the
stream's end, `stop` or a failed play), as a decoder cut off mid-write says so. A mono pipeline gets gray, the luma straight from ffmpeg without a
color conversion and at a third of BGR's bytes in the pipe; NV12, as a camera delivers it, is half of BGR's bytes.
It waits for each frame's moment on a `modules.utils.Waker`, which
`stop` wakes: a timed `Event.wait` returns on Windows' 15.6 ms timer ticks. Its frame gaps in deep_flow `-sim`
(20261112-130105_show, 50 fps, 20 s) **(measured 2026-11-20)**:

| Pacing                   | Mean (ms) | p95 (ms) | Max (ms) | Over 30 ms |
|--------------------------|-----------|----------|----------|------------|
| `Event.wait` alone       | 19.99     | 31.8     | 33.5     | 267        |
| `Event.wait`, then sleep | 20.00     | 22.5     | 26.6     | 0          |

## Depth

Depth is a 12-bit HEVC stream on the iGPU (`depth_frame.py`), the only hardware path past 8 bits, so recording depth
costs the installation nothing. Millimeters go into the Y plane in 3 mm steps (`DEPTH_STEP_MM`), 12 m full scale, over
a neutral chroma plane. It is lossy but close: on a synthetic hard-edged frame the largest error was 24 mm, and
frame-to-frame noise grew from 8.6 to 9.0 mm **(measured)**.

- Depth is encoded at constant quality (`-q:v`), not `global_quality`: at low values that breaks down to errors in
  meters.
- Depth is written without B-frames, and so is a stream that seeks with it (the depth's confidence): with them its
  frames came out in the wrong order **(site fact)**.

## Screen recorder

Each render tick the layers an app records are drawn into the screen frame, an Fbo of the recording's size;
`ScreenCapture` converts it to its YUV planes (`chroma`: 4:4:4 or 4:2:0; `bit_depth`: 8 or 10), reads them back
into RAM and hands each finished frame (`ScreenFrame`) to the `ScreenRecorder`, whose thread copies it into a store in RAM. When the
store is full, the duration is reached or Stop is pressed, recording stops and the take is held: in and out are set
on it, and each `save` has the recorder's thread encode the frames between them to an HEVC mp4 with NVENC;
`discard` frees the memory. The main thread never waits on the GPU, the copy or the encoder. What is drawn into the
frame, the capture and the preview are render's: [SCREEN_RECORDING.md](../render/general/screen_recording/SCREEN_RECORDING.md).

### What gets recorded

The recorder's settings are the group `screen_recorder` at the root of an app's settings, so it has a tab
of its own on the panel (deep_flow and low_flow). It holds four groups, in the order the recorder uses them:

| Group     | Fields                                                 | Readouts                 |
|-----------|--------------------------------------------------------|--------------------------|
| `frame`   | main_window, size, portrait, width, height, snap, fill |                          |
| `quality` | chroma, bit_depth, frame_step                          | max_duration             |
| `record`  | start, stop, look_back, duration, ram_reserve          | state, recorded, dropped |
| `take`    | in_out, play, save, discard, name, quality             | playhead                 |

Each button has one meaning: Stop ends a recording and holds the take, in look back too; `save` writes the held take
between in and out.

### Size

`width` and `height` are the recording's size. A preset (`size`) or the `portrait` switch writes them; when
they change, the preset becomes the one that matches in either orientation, else CUSTOM, and `portrait`
becomes height > width. Every write lands on the value already there, so the three never loop. `snap` sets
the size to the first source's shape, or the window's with `main_window`. Sizes are rounded down to even numbers, which
4:2:0 needs. The size, the chroma and the bit depth are read at Start and hold for that recording.

| Preset | Landscape   | Portrait    |
|--------|-------------|-------------|
| HD     | 1920 × 1080 | 1080 × 1920 |
| QHD    | 2560 × 1440 | 1440 × 2560 |
| UHD    | 3840 × 2160 | 2160 × 3840 |
| CUSTOM | as set      | as set      |

### How many seconds fit

Quality is traded against length. `chroma` and `bit_depth` (`ScreenFormat`, `modules/video/screen/screen_format.py`) pick
the frame's layout, BT.709 video range always, one (rows, width) array of planes, top row first:

| `chroma`, `bit_depth` | Planes                                          | Bytes/pixel | ffmpeg input | For                          |
|-----------------------|-------------------------------------------------|-------------|--------------|------------------------------|
| YUV444, 8 (default)   | Y, Cb, Cr, each at full size, bytes             | 3           | `yuv444p`    | sharp colored edges, masters |
| YUV420, 8             | Y, then Cb Cr interleaved at half size, bytes   | 1.5         | `nv12`       | twice the length             |
| YUV444, 10            | one plane, Cb Y Cr packed in 32 bits            | 4           | `xv30le`     | smooth gradients, masters    |
| YUV420, 10            | as YUV420 8 in 16-bit samples, the 10 bits high | 3           | `p010le`     | smooth gradients, length     |

4:2:0 halves the color resolution, which blurs sharp colored edges in graphic content; 4:4:4 keeps it. 10 bits keep
the layers' gradients without 8-bit steps (banding); at 10 bits the screen frame is RGBA16F, so the layers' precision
reaches the conversion. A 4:4:4 HEVC
file is a master: edit software plays it, Windows' own player often does not. The render loop is not locked to a
rate, so the length follows the window's measured tick rate; `frame_step` 2 doubles it, and a smaller size gives
more (HD is a quarter of UHD's bytes).

| Recording (60 ticks/s) | Per frame | Per second | Seconds in 8 GB | Seconds in 60 GB |
|------------------------|-----------|------------|-----------------|------------------|
| UHD, 4:4:4             | 24.9 MB   | 1.49 GB    | 5.4 s           | 40 s             |
| UHD, 4:2:0             | 12.4 MB   | 746 MB     | 10.7 s          | 80 s             |
| UHD, 4:4:4, 10 bits    | 33.2 MB   | 1.99 GB    | 4.0 s           | 30 s             |
| UHD, 4:2:0, 10 bits    | 24.9 MB   | 1.49 GB    | 5.4 s           | 40 s             |
| UHD, RGBA8 (not used)  | 33.2 MB   | 1.99 GB    | 4.0 s           | 30 s             |
| QHD, 4:4:4             | 11.1 MB   | 664 MB     | 12.0 s          | 90 s             |
| QHD, 4:2:0             | 5.5 MB    | 332 MB     | 24.1 s          | 181 s            |
| HD, 4:4:4              | 6.2 MB    | 373 MB     | 21.4 s          | 161 s            |
| HD, 4:2:0              | 3.1 MB    | 187 MB     | 42.9 s          | 322 s            |

All rows are **(deduction)**: computed from the frame sizes, not measured. The PC has 96 GB of RAM, 74 GB free with
the app off, and a 4090 with 24 GB **(measured 2026-11-19)**.

The store is plain RAM (`ScreenFrameStore`, numpy blocks): a block allocates without locking any page, and its
memory goes back when the store is cleared. It is not pinned: pinning a one-second block (about 750 MB at UHD60)
locks every page and serializes the driver with the GPU, which stalls the render tick once a second
**(seen in low_flow, 2026-11-19)**. A store in video memory gives no more length: the render tick does
the same work either way, the encode needs every frame in RAM, and the fluid and the models use the card's memory.

The store grows while it records, in blocks of one second. The recorder's thread adds the next block when half of
the last one is left, after reading the RAM available to processes (psutil), so what the rest of the app holds is
counted. When the next block would leave less than `ram_reserve` free (default 8 GB), the store grows no more and
the recording stops when it is full. The check runs once per recorded second and Start makes no large allocation.

A run in deep_flow (`-sim`, 20261112-130105_show, UHD portrait at 4:2:0, 60 Hz with v_sync, `ram_reserve` 10 GB,
commit 51145c73, encoded at `-preset p5` without multipass or AQ) **(measured 2026-11-20)**:

| Phase                 | Ticks | Mean (ms) | p95 (ms) | Max (ms) | Over 25 ms | Dropped |
|-----------------------|-------|-----------|----------|----------|------------|---------|
| Idle                  | 178   | 16.75     | 26.4     | 38.9     | 8 %        |         |
| Recording 30 s        | 1821  | 16.69     | 25.8     | 42.8     | 7 %        | 0       |
| Encoding the 30 s     | 2609  | 17.00     | 27.9     | 45.2     | 11 %       |         |
| Recording to reserve  | 6171  | 16.74     | 25.8     | 62.3     | 7 %        | 0       |
| Held, 60 GB in RAM    | 300   | 16.72     | 24.8     | 32.7     | 4 %        |         |

The long ticks come as often idle as while recording and at no once-a-second rhythm. The 30 s take held 1800 frames
at 59.93 fps (worst gap 33.3 ms) and encoded in 20.1 s, 90 frames/s (118.6 MB). The take to the reserve held 6151
frames, 103 s, and stopped with 10.3 GB free; the app ran on while it was held, and `discard` gave the RAM back.

The same run at the default format, 4:4:4, with the encode in [Encode](#encode-on-the-recorders-thread) (commits
7e32c69a and fb119ec4) **(measured 2026-11-20)**:

| Phase                 | Ticks | Mean (ms) | p95 (ms) | Max (ms) | Over 25 ms | Dropped |
|-----------------------|-------|-----------|----------|----------|------------|---------|
| Idle                  | 179   | 16.68     | 25.1     | 37.9     | 6 %        |         |
| Recording 30 s        | 1803  | 16.85     | 26.3     | 43.4     | 7 %        | 0       |

The 30 s take, 1782 frames at 59.34 fps (worst gap 36.1 ms), encoded in 48.7 s: 36.6 frames/s, under the show's
58, so a take takes about 1.6 times its length to save (120.2 MB).

At the heaviest read-back, 4:4:4 at 10 bits (33.2 MB a frame, commit ef3adfa3) **(measured 2026-11-20)**:

| Phase                 | Ticks | Mean (ms) | p95 (ms) | Max (ms) | Over 25 ms | Dropped |
|-----------------------|-------|-----------|----------|----------|------------|---------|
| Idle                  | 174   | 17.15     | 27.4     | 38.4     | 11 %       |         |
| Recording 15 s        | 903   | 17.03     | 26.7     | 48.1     | 7 %        | 0       |

The 15 s take, 882 frames at 58.73 fps, encoded in 24.0 s: 36.8 frames/s, as at 8 bits (HEVC Rext,
`yuv444p10le`, 61.1 MB).

`max_duration` is the estimate (free RAM + the store's own − `ram_reserve`) / (bytes per frame × measured ticks/s /
`frame_step`), refreshed once a second. `duration` 0, or above `max_duration`, records until the memory stops it.

`duration` is counted in time, by the frames' own timestamps: a take ends at the first frame whose time is
`duration` after the first frame's, so it spans `duration` whatever the tick rate does while it records. The
blocks are sized by the rate measured at Start; they are only chunks of memory, and the store grows by as many
as the take needs.

### The hand-over, to the recorder's thread

- `ScreenRecorder.submit_screen_frame` enqueues the frame and returns; a frame no take wants is given back at once.
- The recorder's thread copies each frame into its slot in the store and gives the buffer back. A full queue (three
  frames) drops its oldest frame and gives its buffer back: waiting for the recorder's thread would stall the render
  tick and the show, and a drop needs the thread to fall three ticks behind; no measured take dropped a frame.
  Frames left out, by the queue, by the capture with no buffer free, or made at a size the take is not, are counted
  for the take (`dropped`), shown on the panel and written in the file's comment.
- Frames still on their way when a take is held, or from before a take began, are given back unused.

### Encode, on the recorder's thread

- One ffmpeg process takes the raw planes on stdin (`ClipWriter` with `ClipInput.YUV444P`, `NV12`, `XV30` or
  `P010`, and `master_encode`) through a pipe of 8 MB (`PIPE_SIZE`; Windows' default pipe takes 4 KB a write):
  `hevc_nvenc -preset p7 -tune hq -multipass fullres -rc vbr -cq <quality> -b:v 0 -spatial-aq 1 -temporal-aq 1`,
  tagged BT.709 video range and `hvc1`, so Windows and QuickTime play the 4:2:0 files. The encode runs after a take,
  so its time goes to quality: the slowest preset, a full-resolution second pass, and the bits put where the eye
  sees them. 4:4:4 input is encoded as HEVC 4:4:4 (Range Extensions), P010 as HEVC Main 10. NVENC does not take
  `xv30le`, so ffmpeg unpacks it to `yuv444p16le` on the CPU, which costs only the encode's time. HEVC because fine
  detail at 4K needs fewer bits than h264 for the same quality.
- Each slot of the store is written to the pipe as it is.
- After the last frame the pipe is closed and ffmpeg waited for; the store stays held.
- The thread makes no GL call; it touches only RAM and the pipe. NVENC is its own unit on the card, so the show
  runs on while it encodes.

Files go to the folder the app gives, named `<date-time>_<name>.mp4`; a second part saved within the same second
gets `_2`, and so on.

### The held take

A held take keeps its store until `discard`. The `in_out` timeline (the player's `Widget.timeline`, with its
`playhead`, `scrub` and `seek` fields) spans the take's length by its frames' times; `save` writes the frames whose
times lie between in and out, and the take stays held, so a second part can be saved from it. `discard` frees the
store, or in look back empties it and records on. Start is refused while a take is held. The playhead follows the
dragged marker, or the place on the bar pressed, so it shows where the take is looked at. `scrub` and `seek` are
bound, because the panel writes them back to −1 at once.

A dragged marker shows its frame in the main window through render's preview
([The preview](../render/general/screen_recording/SCREEN_RECORDING.md#the-preview)).

`play` plays the held take over the main window through the same preview: each render tick the recorder moves the
play position by the tick's time, looping between in and out, so the frames show at their recorded times whatever
the tick rate, and `preview_frame` gives the frame there. A dragged marker shows its own frame, and play goes on from
it. Only a held take plays: the switch goes off without one and when a save starts, as the save turns the take to
ENCODING. While it plays the show waits: each app reads `ScreenRecordingCompositor.playing` (the recorder's state,
not the switch, so a refused play holds nothing), takes and drops its camera frames, and runs and draws no layer.
The fluid goes on where it stopped, and the other windows keep the show's last picture. This is not
`freeze_frame`, the apps' switch that holds the camera frame while the fluid runs on with it. In deep_flow (`-sim`,
UHD portrait 4:4:4, 60 Hz with v_sync) **(measured 2026-11-20)**: playing the middle half of a held 3.85 s take for
6 s, the ticks were 16.66 ms on average (p95 19.3 ms, none over 25 ms; idle 16.89 ms, 8 % over 25 ms), the motion and
the fluid did no work, and the preview drew 118 different frames; after play the fluid ran on.

The show's first ticks after play are its ordinary ones. Four plays of 2 s, each followed by 3 s of show, against four
3 s windows of show without play (deep_flow `-sim`, 20261112-130105_show) **(measured 2026-11-20)**:

| Window                | Resume tick (ms) | Slowest tick (ms) | Fluid's update in it (ms) |
|-----------------------|------------------|-------------------|---------------------------|
| 3 s after play        | 16.1–22.1        | 37.6–64.5         | 1.5–35.0                  |
| 3 s of show, no play  |                  | 33.2–35.7         | 13.9–17.5                 |

The fluid's update is the CPU time of its GL calls; a long one waits on the GPU **(deduction)**.

### Look back

With `look_back` on the store records all the time as a ring, the newest frame replacing the oldest. It holds 1.2 ×
`duration` at the rate measured when it is made (`LOOK_BACK_MARGIN`), so ticks faster than that still leave
it the last `duration` seconds; with `duration` 0 it holds what the memory holds. Stop holds the frames whose
times lie in the last `duration` seconds, as it holds a recording; `save` only writes a held take, oldest first (see the held take); a slot being written over at that
moment is left out. The store pauses while the take is held and records on after `discard`; it is blind for that
time, because recording on in a second store would need the memory twice. It holds its memory while the switch is
on; a change of size or duration empties it and makes it again.

### Time

One frame per render tick, or per second tick with `frame_step` 2. Each slot keeps its frame's `timestamp`, the
tick's. The file is written at the rate measured over the take, (frames − 1) / (last − first time), so every
stored frame is in it once and the video lasts as long as the take did; uneven ticks play back evened out. The
mean and the worst gap and the frames dropped go in the log line and the file's comment.

### States

| State     | Means                                                        |
|-----------|--------------------------------------------------------------|
| IDLE      | nothing held; Start, or look back on, begins                 |
| RECORDING | frames are captured and stored each tick                     |
| HELD      | the take waits for in, out, `save` and `discard`             |
| ENCODING  | the thread encodes a part, or frees the store                |
| ERROR     | a failed capture or encode; only Start retries               |

A failure (no ffmpeg, no NVENC, a capture that cannot be made) is logged once, shown as ERROR, and the show
runs on. At exit a recording in progress is written up to its last stored frame, a look-back ring is not written,
and a held take none of whose parts is written yet is written whole; the thread is joined, every frame is given back and the store freed, then the
main thread frees the capture.

### Limits

- 8 bit: video range keeps 219 of the window's 256 levels, and smooth dark smoke can band after compression.
  10 bits keep the gradients, at a third more bytes a frame at 4:4:4 and twice the bytes at 4:2:0
  (`ScreenFormat.frame_bytes`; seconds in [How many seconds fit](#how-many-seconds-fit)).
- Sizes that are not multiples of 16 are padded inside HEVC and cropped on playback; any even size is exact
  in the file.
- NVENC refuses frames 128 wide and less (measured, ffmpeg's `hevc_nvenc` on the 4090).

### Sources

- PyNvVideoCodec would encode straight from memory without ffmpeg. The recorder pipes through ffmpeg instead: the
  encode runs after Stop, ffmpeg is already the encoder here, and it adds no package. It is the swap if the
  encode is too slow.

### Constraints from the alternatives

- The store is RAM, not GL textures or buffers: a GL store would be read back on the main thread while the show
  runs, and the encode thread cannot touch GL.
- The store is not pinned (above).
- The encode runs after the recording, not live: live it would push ~750 MB/s **(deduction)** through a pipe and
  NVENC while the show runs.
- The store holds YUV planes, not RGBA: 4:4:4 is 3 bytes a pixel against 4, and 4:2:0 gives 2.7× the seconds.
- The capture's own constraints (a fragment shader, 10 bits packed, no `x2rgb10le`) are in
  [SCREEN_RECORDING.md](../render/general/screen_recording/SCREEN_RECORDING.md#constraints-from-the-alternatives).
