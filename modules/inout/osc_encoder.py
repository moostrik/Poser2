"""OSC encoder: a message's and a bundle's bytes, as OSC 1.0 has them.

An int is an int32 (`i`), a float a float32 (`f`), a str a string (`s`), and a float array a row of float32s
(`f` per value). A bundle's time tag is "immediately"."""

from typing import Iterable
import struct

import numpy as np

OscArg = int | float | str | np.ndarray
OscMessage = tuple[str, tuple[OscArg, ...]]

_BUNDLE_HEAD: bytes = b'#bundle\x00' + (1).to_bytes(8, 'big')      # the tag and the time tag "immediately"


def _string(value: str) -> bytes:
    """UTF-8, ended by one to four zero bytes to a multiple of four."""
    data: bytes = value.encode('utf-8')
    return data + b'\x00' * (4 - len(data) % 4)


def encode_message(address: str, args: Iterable[OscArg]) -> bytes:
    tags: list[str] = [',']
    data: list[bytes] = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            tags.append('f' * arg.size)
            data.append(arg.astype('>f4').tobytes())
        elif isinstance(arg, (bool, np.bool_)):
            raise TypeError(f'{address}: a bool has no OSC type here; send it as an int')
        elif isinstance(arg, (int, np.integer)):
            tags.append('i')
            data.append(struct.pack('>i', arg))
        elif isinstance(arg, (float, np.floating)):
            tags.append('f')
            data.append(struct.pack('>f', arg))
        elif isinstance(arg, str):
            tags.append('s')
            data.append(_string(arg))
        else:
            raise TypeError(f'{address}: no OSC type for {type(arg).__name__}')
    return _string(address) + _string(''.join(tags)) + b''.join(data)


def encode_bundle(messages: Iterable[bytes]) -> bytes:
    """A bundle of encoded messages."""
    parts: list[bytes] = [_BUNDLE_HEAD]
    for message in messages:
        parts.append(struct.pack('>i', len(message)))
        parts.append(message)
    return b''.join(parts)
