"""ffmpeg and ffprobe as subprocesses, and the pipes their frames go through.

The apps run under pythonw, so every process is started with CREATE_NO_WINDOW
and without inheriting stdin or stderr, which are invalid handles there. stderr is a pipe that is
read when the process ends, so a failure has a message."""

import io
import os
import shutil
import subprocess


CREATE_NO_WINDOW: int = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
FFMPEG: str = 'ffmpeg'
FFPROBE: str = 'ffprobe'


def available() -> bool:
    return shutil.which(FFMPEG) is not None and shutil.which(FFPROBE) is not None


BELOW_NORMAL: int = getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0)


def popen(args: list[str], stdin, stdout, command: str = FFMPEG, background: bool = False) -> subprocess.Popen:
    """Start ffmpeg (or ffprobe) with the arguments after the log options; stderr is always a pipe.
    A background process runs below normal priority."""
    flags: int = CREATE_NO_WINDOW | (BELOW_NORMAL if background else 0)
    return subprocess.Popen([command, '-hide_banner', '-loglevel', 'error', *args],
                            stdin=stdin, stdout=stdout, stderr=subprocess.PIPE,
                            creationflags=flags)


PIPE_SIZE: int = 8 << 20     # the default Windows pipe holds 4 KB: reading a clip through it takes 6 times as long


def pipe() -> tuple[int, int]:
    """A pipe with a large buffer: (read descriptor, write descriptor)."""
    if os.name == 'nt':
        import _winapi
        import msvcrt
        read_handle, write_handle = _winapi.CreatePipe(None, PIPE_SIZE)
        return msvcrt.open_osfhandle(read_handle, os.O_RDONLY), msvcrt.open_osfhandle(write_handle, 0)
    return os.pipe()


def big_pipe() -> tuple[io.FileIO, int]:
    """A pipe with a large buffer: (unbuffered read file, write descriptor to hand to the child)."""
    read_fd, write_fd = pipe()
    return os.fdopen(read_fd, 'rb', buffering=0), write_fd


def error_text(output: bytes | None) -> str:
    return (output or b'').decode(errors='replace').strip()
