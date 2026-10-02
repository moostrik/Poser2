"""The record of a visitor: how its portrait and grin were made, kept inside the saved files.

A JSON text: id, style, description, seed, the grin's peaks and the pipeline settings in preset shape
(settings.pipeline). ClipSaver writes it into the clip (the mp4's comment tag) and the portrait (JPEG
comment segments); this module puts it in a JPEG and reads it back from either.

    python -m modules.save.record <clip or portrait>                  prints the record
    python -m modules.save.record <clip or portrait> --preset <name>  writes its settings to data/settings/<name>.json
    python -m modules.save.record <clip or portrait> --open           opens the record as a file (scripts/clip_info.bat)
"""

import argparse
import json
import os

from modules.settings import presets
from modules.video import ffmpeg

SOI: bytes = b'\xff\xd8'
COM: int = 0xFE
SOS: int = 0xDA
MAX_SEGMENT: int = 65533    # bytes of one segment's payload: its length field counts itself, in 16 bits
JPEG_EXTENSIONS: tuple[str, ...] = ('.jpg', '.jpeg')


def embed_jpeg(jpeg: bytes, text: str) -> bytes:
    """The JPEG with text as comment segments, after the APPn segments (JFIF wants its APP0 first)."""
    if not jpeg.startswith(SOI):
        raise ValueError('not a JPEG')
    at: int = 2
    while at + 4 <= len(jpeg) and jpeg[at] == 0xFF and 0xE0 <= jpeg[at + 1] <= 0xEF:
        at += 2 + int.from_bytes(jpeg[at + 2:at + 4], 'big')
    data: bytes = text.encode('utf-8')
    segments: list[bytes] = []
    for start in range(0, len(data), MAX_SEGMENT):
        part: bytes = data[start:start + MAX_SEGMENT]
        segments.append(bytes((0xFF, COM)) + (len(part) + 2).to_bytes(2, 'big') + part)
    return jpeg[:at] + b''.join(segments) + jpeg[at:]


def read_jpeg(path: str) -> str | None:
    """The text of the JPEG's comment segments, or None when it has none."""
    with open(path, 'rb') as f:
        jpeg: bytes = f.read()
    if not jpeg.startswith(SOI):
        return None
    parts: list[bytes] = []
    at: int = 2
    while at + 4 <= len(jpeg) and jpeg[at] == 0xFF and jpeg[at + 1] != SOS:
        length: int = int.from_bytes(jpeg[at + 2:at + 4], 'big')
        if jpeg[at + 1] == COM:
            parts.append(jpeg[at + 4:at + 2 + length])
        at += 2 + length
    return b''.join(parts).decode('utf-8', errors='replace') if parts else None


def read(path: str) -> dict | None:
    """The record in a clip or a portrait, or None when the file has none."""
    is_jpeg: bool = path.lower().endswith(JPEG_EXTENSIONS)
    text: str | None = read_jpeg(path) if is_jpeg else ffmpeg.read_comment(path)
    if not text:
        return None
    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser(description='The record in a saved clip or portrait: how it was made')
    parser.add_argument('file', help='a clip (output/grins/<style>/<id>.mp4) or its portrait (<id>_portrait.jpg)')
    parser.add_argument('--preset', metavar='NAME', help='write the settings to data/settings/NAME.json')
    parser.add_argument('--open', action='store_true', help='write the record to output/temp/ and open it')
    args = parser.parse_args()

    record: dict | None = read(args.file)
    if record is None:
        raise SystemExit(f'No record in {args.file}')
    text: str = json.dumps(record, indent=2, ensure_ascii=False)

    if args.preset:
        presets.set_app('smile')
        presets.validate_name(args.preset)
        preset_path = presets.path(args.preset)
        preset_path.write_text(json.dumps(record.get('settings', {}), indent=2), encoding='utf-8')
        print(f'Wrote {preset_path}')
    elif args.open:
        id: str = str(record.get('id') or os.path.splitext(os.path.basename(args.file))[0])
        path: str = os.path.join('output', 'temp', f'{id}_record.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        try:
            os.startfile(path)
        except OSError:
            print(f'Wrote {path}')
    else:
        print(text)


if __name__ == '__main__':
    main()
