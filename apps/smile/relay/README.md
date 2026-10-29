# The relay of SMILE Online

One small server between the phones and SMILE: it serves the phone page (`web/`; the caption's font
`web/Bangers-Regular.ttf` is a copy of SMILE's `data/Bangers-Regular.ttf`, the VPS has only this folder), keeps the waiting
line and passes the frames, the judging and the results (`server.py`, the protocol is in its docstring).
SMILE decides every turn; the relay only keeps the line. Its data is a folder of the visitors' files
(`data/`, git-ignored): the portraits and the grins (the phones' share clips, one each per kept visitor, under names
nobody can guess; deleted 5 minutes after the visitor's phone left or 30 minutes after the upload, `KEEP_AFTER_LEAVE`
and `KEEP_AT_MOST`), and the library the
phones play from: the music tracks (`tracks/`, AAC, about 1.5 MB a track: SMILE uploads a track the first time a
visitor gets it) and the laughter samples (`laughs/`, 8.7 MB, uploaded at SMILE's first connect).
Where it runs and how it is reached, on the tailnet for a test and on the VPS, is in
`data/documents/OnlineHosting.md`.

## Run

* a token: one line of random text in `relay/token.txt` (git-ignored) and the same in SMILE's
  `data/online_token.txt` (`python -c "import secrets; print(secrets.token_urlsafe(32))"` makes one); SMILE sends it
  as an `Authorization` header, never in a URL, and the relay keeps no access log, so no log holds it
* `pip install aiohttp` (Poser2's venv has it), then from the Poser2 root:
  `python -m apps.smile.relay.server --port 8800` (`-v` for debug logging); `apps\smile\scripts\start_relay.bat` does this on the PC.
  On the VPS the `relay` folder stands alone and runs as `python -m relay.server` beside it.
  The log goes to the console and to `data/relay.log`; every phone reports the end of its turn there
  (`phone … timing: result, play, playing, file; first byte, KB, KB at every second`: seconds after the turn's end that
  the result came, the video was started, played its first frame, and the file was on the phone; then the
  download itself: when its first byte came, its size, and the KB received at every second of it)
* the page is at `http://<host>:8800/`. A phone only opens its camera over HTTPS, so a TLS proxy stands in
  front (`data/documents/OnlineHosting.md`); without one the relay takes `--cert` and `--key`
* tests: `relay/tests` covers the line, the server and the page's files (`test_web.py`: the stylesheet's structure, every
  id the script and the stylesheet name, no name declared twice in the script: what a browser would refuse silently)
* without SMILE: `python -m apps.smile.online.fake` stands in for it (the SCRFD face detector, or the middle of the
  frame without the model; a fake transformation with SMILE's stages; the frame comes back as the portrait)
