# The relay of SMILE Online

One small server between the phones and SMILE: it serves the phone page (`web/`; the caption's font
`web/Bangers-Regular.ttf` is a copy of SMILE's `data/Bangers-Regular.ttf`, the VPS has only this folder), keeps the waiting
line and passes the frames, the judging and the results (`server.py`, the protocol is in its docstring).
SMILE decides every turn; the relay only keeps the line. Its data is a folder of the visitors' files
(`data/`, git-ignored): the stills (one per turn, served at `/api/grid`; the page shows none), the portraits and the grins (the phones' share clips), and the library the
phones play from: the music tracks (`tracks/`, AAC, about 150 MB for all 75: SMILE uploads a track the first time a
visitor gets it) and the laughter samples (`laughs/`, 8.7 MB, uploaded at SMILE's first connect).
Where it runs and how it is reached, on the tailnet for a test and on the VPS, is in
`data/documents/OnlineHosting.md`.

## Run

* a token: one line of random text in `relay/token.txt` (git-ignored) and the same in SMILE's
  `data/online_token.txt`
* `pip install aiohttp` (in SMILE's venv it is in `requirements.txt`), then from the repo root:
  `python -m relay.server --port 8800` (`-v` for every message); `scripts\start_relay.bat` does this on the PC.
  The log goes to the console and to `data/relay.log`; every phone reports the end of its turn there
  (`phone … timing: result, play, playing, file; first byte, KB, KB per second`: seconds after the turn's end that
  the result came, the video was started, played its first frame, and the file was on the phone; then the
  download itself: when its first byte came, its size, and the KB received at every second of it)
* the page is at `http://<host>:8800/`. A phone only opens its camera over HTTPS, so a TLS proxy stands in
  front (`data/documents/OnlineHosting.md`); without one the relay takes `--cert` and `--key`
* without SMILE: `python -m modules.online.fake` stands in for it (a Haar face detector, a fake
  transformation, the frame comes back as the portrait)
