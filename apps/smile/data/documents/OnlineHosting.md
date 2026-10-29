# SMILE Online: hosting the relay

Where the relay of SMILE Online runs and how SMILE and the phones reach it: on the tailnet for the tests
now, on the VPS for the public. The relay, its start, its data folder and the way it is reached are the
same in both; the two differ in the machine, the TLS proxy in front and the name. `relay/README.md` has the
relay itself, `OnlinePhone.md` what happens on a phone.

## Now: the test on the tailnet

1. The token: one line of random text in `relay/token.txt`, the same line in `data/online_token.txt`.
2. The relay: `scripts\start_relay.bat` (port 8800, data in `relay/data`, a window of its own).
3. TLS in front, once: `tailscale serve --bg 8800`. The page is `https://iaai.tail63e3df.ts.net/`;
   `tailscale serve status` shows the proxy, `tailscale serve off` removes it. A phone opens its camera over
   HTTPS only.
4. SMILE, Online tab: `relay` = `wss://iaai.tail63e3df.ts.net/ws/smile`, `active` on. SMILE goes through the
   same name and proxy as the phones, so its websocket and its uploads run as they will on the VPS.
   `ws://127.0.0.1:8800/ws/smile`, the default, reaches the relay past the proxy and tests less.
5. Phones on the tailnet open the page as it is. A phone without Tailscale needs `tailscale funnel 8800`
   (the page open to the internet while it runs; Ctrl-C ends it).
6. The library: the relay fills it from SMILE (see [The library](#the-library)), or it is copied from a
   relay that has it.

## Later: the VPS

1. A VPS (a TransIP VPS V1 is enough) with Python 3.12 or newer; clone the repo (only `relay/` is used),
   `pip install aiohttp`, the token in `relay/token.txt` as above.
2. The relay as a systemd service under a normal user: `python -m relay.server --port 8800` from the repo
   root, data in `relay/data` as on the PC. The backup is a copy of that folder.
3. TLS in front: a DNS name for the VPS and Caddy as reverse proxy from 443 to `127.0.0.1:8800`; Caddy
   fetches and renews the certificate. Without a proxy the relay takes `--cert` and `--key` and the renewal
   is yours.
4. SMILE, Online tab: `relay` = `wss://<name>/ws/smile`, `active` on. SMILE makes outgoing connections only;
   nothing on the installation is opened to the internet.
5. The library: copy `relay/data/tracks` and `relay/data/laughs` from the test relay before the first
   visitor.
6. After the first turns: `relay/data/relay.log`, the `timing` and `frames to the phone` lines, for what the
   venue's uplink does to the times measured on one PC in `OnlinePhone.md`.

## The same and the different

| Part                   | Tailnet test                             | VPS                                   |
|------------------------|------------------------------------------|---------------------------------------|
| the relay              | this PC, `scripts\start_relay.bat`       | the VPS, a systemd service            |
| its start              | `python -m relay.server --port 8800`     | the same                              |
| its data               | `relay/data`                             | the same                              |
| TLS                    | `tailscale serve`, the ts.net certificate | Caddy, a Let's Encrypt certificate   |
| the name               | `iaai.tail63e3df.ts.net`                 | a DNS name of the VPS                 |
| who reaches the page   | the tailnet (`funnel`: anyone)           | anyone                                |
| SMILE's `online.relay` | `wss://iaai.tail63e3df.ts.net/ws/smile`  | `wss://<name>/ws/smile`               |
| the QR code on the walls | `https://iaai.tail63e3df.ts.net/` (`online.page` empty: the relay's host) | `https://<name>/`, or `online.page` |
| SMILE to the relay     | one PC                                   | the venue's uplink                    |

The relay never terminates TLS itself in either case and reads no difference between the two.

## The library

SMILE's tracks (`data/audio/SMILE-BS-Music`, `DB.MUSIC_FOLDER`) and laughter samples (`data/audio/laughter`)
are the phones' too. The relay keeps them in `relay/data/tracks` (as AAC) and `relay/data/laughs`, and tells
SMILE what it has at every connect (`/api/library`). SMILE uploads only what the relay lacks: the laughter
at the first connect (`Online._upload_laughs`), a track the first time a visitor gets it
(`Online._ensure_track`, transcoded by `Link._track_aac`), each once per relay; the relay keeps them across
restarts. A track uploaded during its first turn may reach the phone after the flash and starts that far
in (`offset`). Copying the two folders from a relay that has them, step 5 above, spares a new relay that
first use of every track.

| What                                  | Size                                   |
|---------------------------------------|----------------------------------------|
| the 75 tracks a visitor can get, MP3  | 200 MB (`music_map.json`, not rejected) |
| the same as AAC 128 kbit/s            | about 150 MB, 2 MB a track              |
| in the test relay after 46 turns      | 28 tracks, 55 MB                        |
| the 304 laughter samples              | 8.7 MB                                  |
| one turn's portrait and grin          | about 1.6 MB (152 turns: 246 MB)        |
| the relay's upload limit              | 64 MB (`MAX_UPLOAD`)                    |

A turn's portrait and grin (`KINDS`) are stored under names nobody can guess (the id is the time, which a 30-minute
window would leave 1800 guesses) and deleted `KEEP_AFTER_LEAVE` (5 minutes) after the visitor's phone left or
`KEEP_AT_MOST` (30 minutes) after the upload, whichever comes first (`Delivery`, `Relay.sweep`): the phone has the
video as a copy of its own once the download is through, so nothing on the relay is needed after that, and a
download cut by a dark screen has its minutes to try again. The sweep also takes whatever else in `portraits/`
and `grins/` is older than 30 minutes (a restart between upload and result; an older relay's files), so the
folders hold at most the last half hour of turns.

## Open

- SMILE through the TLS proxy (step 4 of the test) has not run yet; the proxy's limit on an upload's size
  is not checked against the 2 MB tracks and grins.
- The relay on another machine of the tailnet, to put a network between SMILE and the relay in the test.
- The VPS's DNS name, and whether the page gets a name under oostrik.net.
- The times over the venue's uplink.
