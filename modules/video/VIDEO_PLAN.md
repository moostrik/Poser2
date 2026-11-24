# The video module's plan

What is to be built in `modules/video`: ffmpeg, the clips and streams it writes and reads, and the screen recorder
([VIDEO.md](VIDEO.md)).

## A. The reader's own end

- A1. A `StreamReader` that ends its decoder logs nothing of that end. Now `_play_frames` breaks at
  `end_frame`, `_play`'s `finally` closes the pipe and terminates ffmpeg, which is still writing, and `_log_error`
  logs its lines as errors, as it stays quiet only after `stop()` **(seen)**: in deep_flow -sim on
  `20261121-010616_measure_confidence`, out marker at 34.6 s inside chunk 0, every loop logged about 10 ERROR lines
  ("Error submitting a packet to the muxer: Invalid argument", "Terminating thread with return code -22"), and
  none on `20261124-102304_Test`, out marker at its end **(seen, 2026-11-24)**. How: an `Event` set in `_close` itself,
  before `frames_in.close()`, so every path that ends the decoder (`end_frame`, the stream's end, `stop()`, a failed
  play) is covered once; `_log_error` checks that `Event` in place of `stop_event`. Why: an ERROR line is a fault to
  look into; a reader that ends its own decoder is none. Test: `test_stream_reader`, a reader with an `end_frame`
  before the clip's end and no failing callback logs no "Decoder of" line. ffmpeg's write after the close is a race:
  `test_a_frame_callback_that_raises_is_logged_and_the_reader_plays_on` already reads `end_frame` 5 of a longer clip
  and passed on Windows, so the new test may pass there without the fix too **(seen)**.

## The rules it touches

- Log exceptions with context; never fail silently: A1 (quiet only for an end the reader made itself)
