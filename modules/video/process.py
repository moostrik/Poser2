"""ffmpeg and ffprobe as subprocesses, and the pipes their frames go through.

The apps run under pythonw, so every process is started with CREATE_NO_WINDOW
and without inheriting stdin or stderr, which are invalid handles there. stderr is a pipe that is
read when the process ends, so a failure has a message."""

import io
import os
import shutil
import subprocess
from threading import Thread
from typing import IO, Callable

import logging
logger = logging.getLogger(__name__)


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


def run(args: list[str], stdin_bytes: bytes | None = None, timeout: float = 60.0, command: str = FFMPEG) -> bytes:
    """Run ffmpeg (or ffprobe) to its end, `stdin_bytes` fed to it; its output. Kills it after `timeout` s. Raises
    RuntimeError with its error text when it fails or is killed, OSError when it cannot start."""
    stdin = subprocess.DEVNULL if stdin_bytes is None else subprocess.PIPE
    process = popen(args, stdin, subprocess.PIPE, command=command)
    try:
        out, err = process.communicate(stdin_bytes, timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise RuntimeError(f'{command} did not finish in {timeout:g} s')
    if process.returncode != 0:
        raise RuntimeError(f'{command} failed: {error_text(err) or f"exit code {process.returncode}"}')
    return out


def read_exactly(file: io.RawIOBase, view: memoryview) -> int:
    """Fill `view` from an unbuffered pipe, which hands over what it holds per read; the bytes filled, short only at
    the pipe's end."""
    got = 0
    while got < len(view):
        n = file.readinto(view[got:])
        if not n:
            break
        got += n
    return got


class ErrorReader:
    """A process's error output, read on its own thread as it comes, so its pipe never fills and stops the process;
    each line goes to `on_line`, or without it is kept for `text`. The pipe is closed at its end, when the process
    ends."""

    def __init__(self, process: subprocess.Popen, on_line: Callable[[bytes], None] | None = None,
                 name: str = 'ffmpeg errors') -> None:
        assert process.stderr is not None
        self._stderr: IO[bytes] = process.stderr
        self._on_line: Callable[[bytes], None] | None = on_line
        self._lines: list[bytes] = []
        self._thread: Thread = Thread(target=self._read, name=name, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        with self._stderr as stderr:
            for line in stderr:
                if self._on_line is not None:
                    try:
                        self._on_line(line)
                    except Exception as e:
                        logger.error('Error line callback failed: %s', e)
                else:
                    self._lines.append(line)

    def text(self, timeout: float = 5.0) -> str:
        """The error output kept, once the pipe reached its end or `timeout` s passed."""
        self._thread.join(timeout)
        return error_text(b''.join(self._lines))
