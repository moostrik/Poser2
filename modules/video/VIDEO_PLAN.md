# The video module's plan

## A. One ffmpeg system

The package runs ffmpeg two ways, and names the same hardware two ways. What the code shows:

| Way                                   | Where                                                     | Does                                                                |
|---------------------------------------|-----------------------------------------------------------|---------------------------------------------------------------------|
| `process.py`: `popen` + argument list | `ClipWriter`, `decode_clip`, audio, share, `probe`, trial | the screen recorder, smile's clips and share, the recording's merge |
| ffmpeg-python (`import ffmpeg`)       | `StreamWriter`, `StreamReader` (2 pipelines, 3 probes)    | the camera recordings: write, read, frame count                     |

| Names                                 | Values                                        | Hardware decode                     | Used by                    |
|---------------------------------------|-----------------------------------------------|-------------------------------------|----------------------------|
| `Encoder`, `ClipCodec`                | AUTO, QSV, NVENC, CPU; H264, HEVC             | `hwaccel_args(adapter)`: d3d11va    | screen recorder, smile     |
| `codec.py` `CoderType`, `CoderFormat` | CPU, GPU, iGPU; H264 = '.mp4', H265 = '.hevc' | `hwaccel(decoder)`: d3d12va, 0 or 1 | recorder, player (presets) |

- `process.py` starts a process without a console window and reads and logs its errors. ffmpeg-python runs at
  `-loglevel quiet`, so a failing camera encoder says nothing; it adds nothing `process.py` does not do.
- `CoderFormat` mixes a codec with a file suffix; chunks are `.mkv` whatever the codec. The player takes a color-only
  chunk's suffix from the codec (`player.py`), the recorder from the container, so an H.265 recording in the MP4
  container is written as `0_C_000.mp4` and looked for as `.hevc`.
- The player's decoder is chosen with `CoderType`, an encoder's names: what both pick is the chip (CPU, Intel iGPU,
  Nvidia GPU), from which the encoder follows with the codec, and the hardware decode alone.
- `StreamReader` reads its decoder through Windows' default 4 KB pipe; `decode_clip` uses the 8 MB pipe, measured 6
  times faster for a clip. Whether it costs the player time is not measured.

The target: one place starts ffmpeg and ffprobe, with one set of names. Each step is one commit and leaves the
repository working; its own tests run with it. The runs in an app (the player's timing, a recording, the live run) and
the depthai 2 venv's tests run once, after the last step.

- A5. One writer: `StreamWriter` keeps its queue, thread and split, and writes through `ClipWriter`, which takes gray,
  depth (P012) and color input, a keyframe interval, no B-frames on request, and HEVC on QSV; errors are logged.
  ffmpeg-python leaves `pyproject.toml`; pyright passes on both stream files. Each writer keeps its own hardware, so
  the camera records on the iGPU while the screen recorder encodes on the GPU. VIDEO.md. Check: the recording round
  trip with depth and a gray stream.
- A6. deep_flow's color in HEVC (CAMERA_PLAN.md A4's writer side, after the camera plan's A1): deep_flow's presets
  from H.264 to HEVC on the iGPU, `MP4` staying H.264. The confidence stream's writer is A5's; the camera plan wires
  it. OAK4.md's recording section. Check: the seek check and the encoder load run again.
- A7. The player's pacing: a frame's moment is waited for on `perf_counter`, not with `Event.wait` alone, which on
  Windows waits in ~15.6 ms steps **(deduction)**: in A3's run about a quarter of the frames came ~30 ms after the one
  before, against 20 ms nominal (p95 31.8 ms). One shared helper in `modules/utils` waits until a `perf_counter`
  deadline: a timed wait on a stop event until ~2 ms before it, then `time.sleep` for the rest, returning at once when
  the event is set. `StreamReader` uses it; the module refactor thread reuses it for the other pacing loops. Its unit
  test in `modules/utils/tests`. VIDEO.md. Check: the helper's test; A3's `-sim` run again: the frame interval (mean,
  p95) and the frames over 30 ms. Built after A5 (and A6).

After A7, once: deep_flow `-sim` playback of a known recording; the depthai 2 venv's tests (white_space, hd_trio);
one live deep_flow run recording the camera on the iGPU while a screen take saves on NVENC: the chunk whole with every
stream's frames counted, the take 0 dropped, the ticks.

Assumed: hevc_qsv takes nv12, p010 and p012, not gray, so ffmpeg turns the confidence's gray into nv12 on the CPU
**(deduction: cheap at 480 × 300, not measured; A5's round trip shows it works)**.

## The rules it touches

- Remove obsolete pathways instead of leaving parallel ones: A5
- Read `time.perf_counter()` for every stamp and duration: A7
- Keep presets in sync with code changes: A6
- Log exceptions with context; never fail silently: A5
- Fix a type error at its cause: A5
- A document describes what is built: every step
