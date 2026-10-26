# The relay of SMILE Online

One small server between the phones and SMILE: it serves the phone page (`web/`), keeps the waiting
line and passes the frames, the judging and the results (`server.py`, the protocol is in its docstring).
SMILE decides every turn; the relay only keeps the line. Its data is a folder of the visitors' files
(`data/`, git-ignored): the stills the grid shows, the portraits and the grins.

## Run

* a token: one line of random text in `relay/token.txt` (git-ignored) and the same in SMILE's
  `data/online_token.txt`
* `pip install aiohttp` (in SMILE's venv it is in `requirements.txt`), then from the repo root:
  `python -m relay.server --port 8800` (`-v` for every message); `scripts\start_relay.bat` does this on the PC
* the page is at `http://<host>:8800/`. A phone only opens its camera over HTTPS: for a test on the
  SMILE PC run `tailscale serve --bg 8800` and open the PC's ts.net address on a phone with Tailscale
  (`tailscale funnel 8800` opens it to the internet, for testers without; switch it off after); on the
  VPS put the relay behind a reverse proxy with a certificate, or give it `--cert` and `--key`
* without SMILE: `python -m modules.online.fake` stands in for it (a Haar face detector, a fake
  transformation, the frame comes back as the portrait)

## On the VPS

* a TransIP VPS V1 is enough; install Python 3.12+, clone the repo (only `relay/` is used), `pip install aiohttp`
* run it as a service (systemd) as a normal user, with `--data` on a disk that is backed up if the
  portraits are to be kept
* SMILE's panel: `online.relay` = `wss://<host>/ws/smile`, the token file, and `online.active` on
