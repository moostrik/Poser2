# The video module's plan

## A. One ffmpeg system

The package runs ffmpeg two ways, and names the same hardware two ways. What the code shows:

| Way                                  | Where                                                        | Does                                                                  |
|--------------------------------------|--------------------------------------------------------------|-----------------------------------------------------------------------|
| `ffmpeg.py`: `popen` + argument list | `ClipWriter`, `decode_clip`, audio, share, `probe`, trial    | the screen recorder, smile's clips and share, the recording's merge   |
| ffmpeg-python (`import ffmpeg`)      | `StreamWriter`, `StreamReader` (2 pipelines, 3 probes)       | the camera recordings: write, read, frame count                       |

| Names                                        | Values                                            | Hardware decode                     | Used by                    |
|----------------------------------------------|---------------------------------------------------|-------------------------------------|----------------------------|
| `ffmpeg.py` `Encoder`, `ClipCodec`           | AUTO, QSV, NVENC, CPU; H264, HEVC                 | `hwaccel_args(adapter)`: d3d11va    | screen recorder, smile     |
| `codec.py` `CoderType`, `CoderFormat`        | CPU, GPU, iGPU; H264 = '.mp4', H265 = '.hevc'     | `hwaccel(decoder)`: d3d12va, 0 or 1 | recorder, player (presets) |

- `ffmpeg.py` starts a process without a console window and reads and logs its errors. ffmpeg-python runs at
  `-loglevel quiet`, so a failing camera encoder says nothing; it adds nothing `ffmpeg.py` does not do, and its
  `import ffmpeg` resolves to our `ffmpeg.py` in pyright, so both stream files fail to type-check.
- `CoderFormat` mixes a codec with a file suffix; chunks are `.mkv` whatever the codec. The player takes a color-only
  chunk's suffix from the codec (`player.py`), the recorder from the container, so an H.265 recording in the MP4
  container is written as `0_C_000.mp4` and looked for as `.hevc`.
- The player's decoder is chosen with `CoderType`, an encoder's names: what both pick is the chip (CPU, Intel iGPU,
  Nvidia GPU), from which the encoder follows with the codec, and the hardware decode alone.
- `StreamReader` reads its decoder through Windows' default 4 KB pipe; `decode_clip` uses the 8 MB pipe, measured 6
  times faster for a clip. Whether it costs the player time is not measured.
- `ffmpeg.py` holds many subjects: the process, the encoders, the clip writer, audio, share, the probes, the clip
  decode.

The target: one place starts ffmpeg and ffprobe, with one set of names. Each step is one commit and leaves the
repository working; its own tests run with it. The runs in an app (the player's timing, a recording, the live run) and
the depthai 2 venv's tests run once, after the last step, except where a step changes the real-time path (A3).

- A1. `ffmpeg.py` split by subject, a pure move: process, probe, encoder, clip writer, clip decode, audio and share,
  each file named after it; imports through `__init__.py` unchanged. VIDEO.md's file list. Check: guard and video
  tests.
- A2. One probe: any stream's size, fps, exact frame count (stored, else packets counted) and comment. `probe`,
  `read_comment`, `StreamReader.frame_count` and the reader's `_get_video_dimensions` become it; the recording tests'
  helpers, the seek test's packet listing included, start ffprobe through the process file. VIDEO.md. Check: video,
  recording and smile tests.
- A3. The player's decoder: `StreamReader` starts its decoder through the process file on the 8 MB pipe, the same
  seek, hardware decode, stream and pixel format as now. VIDEO.md. Check: tests; one deep_flow `-sim` run on one
  recording before and after: the player's frame interval (mean, p95) and late frames.
- A4. One set of names: one hardware enum (CPU, iGPU, GPU), the chip, used to encode and to decode, with `ClipCodec`
  beside it; the encoder (h264_qsv, hevc_nvenc…) and the hardware decode follow from chip and codec in code. Smile's
  `AUTO` stays an encoder choice on top. `codec.py` goes, and the fixed `DEPTH_ENCODER` comes from the same table.
  The recorder's and the player's settings use them; the player's `video_format` goes and it takes the color file's
  suffix from the container, as the recorder does; `recording_folder.py` finds files by container. The color part
  is written as `.mkv`, as the depth's is: today H.265 writes a raw `.hevc` with no timestamps, which the merge
  copies in without a frame rate **(deduction: the chunk may get ffmpeg's 25 fps)**. One hardware decode, d3d11va or
  d3d12va, whichever decodes a real chunk faster on the iGPU, both timed. Takes over RECORDING_PLAN.md's rename of
  `CoderFormat` and `CoderType` (its line 2). All presets migrated (deep_flow 5, low_flow 1, hd_trio 2, white_space
  2, smile's if the name changes). VIDEO.md, RECORDING.md's stream table. Check: tests, every preset loads, the
  decode timing, an H.265 chunk's merged timestamps.
- A5. One writer: `StreamWriter` keeps its queue, thread and split, and writes through `ClipWriter`, which takes gray,
  depth (P012) and color input, a keyframe interval, no B-frames on request, and HEVC on QSV; errors are logged.
  ffmpeg-python leaves `pyproject.toml`; pyright passes on both stream files. Each writer keeps its own hardware, so
  the camera records on the iGPU while the screen recorder encodes on the GPU. VIDEO.md. Check: the recording round
  trip with depth and a gray stream.
- A6. deep_flow's color in HEVC (CAMERA_PLAN.md A4's writer side, after the camera plan's A1): deep_flow's presets
  from H.264 to HEVC on the iGPU, `MP4` staying H.264. The confidence stream's writer is A5's; the camera plan wires
  it. OAK4.md's recording section. Check: the seek check and the encoder load run again.

After A6, once: deep_flow `-sim` playback of a known recording; the depthai 2 venv's tests (white_space, hd_trio);
one live deep_flow run recording the camera on the iGPU while a screen take saves on NVENC: the chunk whole with every
stream's frames counted, the take 0 dropped, the ticks.

Assumed: hevc_qsv takes nv12, p010 and p012, not gray, so ffmpeg turns the confidence's gray into nv12 on the CPU
**(deduction: cheap at 480 × 300, not measured; A5's round trip shows it works)**.

## The rules it touches

- Remove obsolete pathways instead of leaving parallel ones: A3, A5
- Name a concept after what it is in the domain; no synonyms, rename everywhere: A4
- Keep presets in sync with code changes: A4, A6
- Its files each hold one class or one subject: A1
- Log exceptions with context; never fail silently: A5
- Fix a type error at its cause: A5
- A document describes what is built: every step
