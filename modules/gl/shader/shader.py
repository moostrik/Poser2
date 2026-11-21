import inspect
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import cast

from OpenGL.GL import GL_ACTIVE_UNIFORMS, GL_COMPILE_STATUS, GL_FRAGMENT_SHADER, GL_LINK_STATUS, GL_VERTEX_SHADER, glAttachShader, glCompileShader, glCreateProgram, glCreateShader, glDeleteProgram, glDeleteShader, glGetActiveUniform, glGetProgramInfoLog, glGetProgramiv, glGetShaderInfoLog, glGetShaderiv, glGetUniformLocation, glLinkProgram, glShaderSource
from watchdog.observers.api import BaseObserver

from .file_watcher import monitor_path

logger = logging.getLogger(__name__)

SHARED_SHADERS: Path = Path(__file__).parent.parent / "shaders"     # the shared includes and `_generic.vert`
SETTLE: float = 0.1                 # s after a file's last event before its programs compile again: a save is whole
READ_RETRIES: int = 30              # frames a program whose file reads empty is tried again before the error
_LOG_LOCATION = re.compile(r'^(\d+)\((\d+)\)')               # NVIDIA's "file(line)" at the start of a log line


def name_error_lines(log: str, files: list[Path]) -> str:
    """A driver's compile log with each "k(line)" location named by its file, k the file's place in `files` (the
    `#line` numbering of `Shader.read_shader_source`); a line without one is kept as it is."""
    def name(match: re.Match[str]) -> str:
        index: int = int(match.group(1))
        return f"{files[index].name}({match.group(2)})" if index < len(files) else match.group(0)
    return '\n'.join(_LOG_LOCATION.sub(name, line) for line in log.splitlines())


def file_stamp(path: Path) -> tuple[int, int] | None:
    """A file's write time (ns) and size; None when it cannot be read."""
    try:
        status = path.stat()
    except OSError:
        return None
    return status.st_mtime_ns, status.st_size


@dataclass
class _Program:
    """A shader class's compiled program, shared by all its objects: compiled at the first `allocate`, freed at the
    last `deallocate`, compiled again when a file it was made from changes."""
    program: int | None = None
    users: int = 0
    # the files it was compiled from, each with its `file_stamp` as it was read
    files: dict[Path, tuple[int, int] | None] = field(default_factory=dict)
    uniforms: dict[str, int] = field(default_factory=dict)     # read after linking; an array element at its use
    marked: bool = False                                # a file changed: compiled again at the next reload point
    marked_at: float = 0.0                              # the last change's time (perf_counter), for SETTLE
    retries: int = 0                                    # frames its files read empty since the change


class Shader():
    """A GL program from GLSL sources named after the module of the class, beside it: `<module>.vert` (else the
    shared `_generic.vert`) and `<module>.frag`. Each object is its own; the compiled program is shared by every
    object of the class."""

    VERTEX_SUFFIX = '.vert'
    FRAGMENT_SUFFIX = '.frag'

    _programs: dict[type, _Program] = {}                # main thread; the watcher thread only marks, under the lock
    _lock = threading.Lock()
    _marked: bool = False                               # any program marked
    _hot_reload_enabled: bool = False
    _directory_observers: dict[Path, BaseObserver] = {}

    def __init__(self) -> None:
        self.allocated: bool = False
        self.shader_name: str = type(self).__name__

    @property
    def shader_program(self) -> int | None:
        """The class's compiled program; None before `allocate` or after a failed first compile."""
        entry: _Program | None = Shader._programs.get(type(self))
        return entry.program if entry is not None else None

    def allocate(self) -> None:
        """Take part in the class's program: compiled at its first user."""
        if self.allocated:
            return
        cls = type(self)
        entry: _Program = Shader._programs.setdefault(cls, _Program())
        entry.users += 1
        if entry.program is None:
            cls._compile_into(entry)
        if Shader._hot_reload_enabled:
            with Shader._lock:
                Shader._watch_files(entry.files)
        self.allocated = True

    def deallocate(self) -> None:
        """Leave the class's program: freed at its last user."""
        if not self.allocated:
            return
        self.allocated = False
        cls = type(self)
        entry: _Program | None = Shader._programs.get(cls)
        if entry is None:
            return
        entry.users -= 1
        if entry.users > 0:
            return
        if entry.program is not None:
            glDeleteProgram(entry.program)
        with Shader._lock:
            del Shader._programs[cls]

    def get_uniform_loc(self, name: str) -> int:
        """A uniform's location in the class's program: read after linking, an array element (`colors[2]`) looked up
        once; a name the program does not have warns once and is -1, which GL ignores."""
        entry: _Program = Shader._programs[type(self)]
        location: int | None = entry.uniforms.get(name)
        if location is None:
            location = int(glGetUniformLocation(entry.program, name))
            entry.uniforms[name] = location
            if location < 0:
                logger.warning("%s: no uniform %s in the program (misspelled, or unused and left out by the compiler)",
                               type(self).__name__, name)
        return location

    # THE SOURCES
    @classmethod
    def _source_dir(cls) -> Path:
        return Path(inspect.getfile(cls)).parent

    @classmethod
    def _source_stem(cls) -> str:
        return cls.__module__.rsplit('.', 1)[-1]

    @classmethod
    def _stages(cls) -> list[tuple[int, Path]]:
        """The program's stages and their files: the class's own `.vert`, else the shared `_generic.vert`, and its
        `.frag`."""
        own_vertex: Path = cls._source_dir() / f"{cls._source_stem()}{cls.VERTEX_SUFFIX}"
        vertex: Path = own_vertex if own_vertex.exists() else SHARED_SHADERS / "_generic.vert"
        fragment: Path = cls._source_dir() / f"{cls._source_stem()}{cls.FRAGMENT_SUFFIX}"
        return [(int(GL_VERTEX_SHADER), vertex), (int(GL_FRAGMENT_SHADER), fragment)]

    # COMPILE
    @classmethod
    def _read_sources(cls, entry: _Program) -> list[tuple[int, Path, str, list[Path]]]:
        """The stages' sources; the files they read, their includes too, become the ones whose change marks it."""
        sources: list[tuple[int, Path, str, list[Path]]] = []
        files: dict[Path, tuple[int, int] | None] = {}
        for stage, path in cls._stages():
            read: list[Path] = []
            sources.append((stage, path, cls.read_shader_source(str(path), read, files), read))
        with Shader._lock:
            entry.files = files
        return sources

    @classmethod
    def _compile_into(cls, entry: _Program, sources: list[tuple[int, Path, str, list[Path]]] | None = None) -> bool:
        """Compile the class's program into `entry`, from `sources` or read now; on failure the program it had stays."""
        if sources is None:
            sources = cls._read_sources(entry)
        program: int | None = cls._compile(sources)
        if program is None:
            return False
        if entry.program is not None:
            glDeleteProgram(entry.program)
        entry.program = program
        entry.uniforms = cls._active_uniforms(program)  # locations may have changed
        return True

    @staticmethod
    def _active_uniforms(program: int) -> dict[str, int]:
        """The program's active uniforms and their locations, an array by its name and by its first element's; a
        uniform in a block has none and is left out."""
        uniforms: dict[str, int] = {}
        for index in range(int(glGetProgramiv(program, GL_ACTIVE_UNIFORMS))):
            raw_name, _size, _type = glGetActiveUniform(program, index)
            name: str = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
            location: int = int(glGetUniformLocation(program, name))
            if location < 0:
                continue
            uniforms[name] = location
            if name.endswith('[0]'):
                uniforms[name[:-3]] = location
        return uniforms

    @classmethod
    def _compile(cls, sources: list[tuple[int, Path, str, list[Path]]]) -> int | None:
        """A linked program from the stages' sources, or None after logging what failed: a compile error by the file
        and line it is in."""
        name: str = cls.__name__
        compiled: list[int] = []
        try:
            for stage, path, source, read in sources:
                if not source:
                    logger.error("%s: no shader source at %s", name, path)
                    return None
                shader: int = cast(int, glCreateShader(stage))       # the stub says None; it is the id
                compiled.append(shader)
                glShaderSource(shader, source)
                glCompileShader(shader)
                if not glGetShaderiv(shader, GL_COMPILE_STATUS):
                    log: str = name_error_lines(glGetShaderInfoLog(shader).decode(), read)
                    logger.error("%s: compile error in %s:\n%s", name, path.name, log)
                    return None
            program: int = cast(int, glCreateProgram())             # the stub says None; it is the id
            for shader in compiled:
                glAttachShader(program, shader)
            glLinkProgram(program)
            if not glGetProgramiv(program, GL_LINK_STATUS):
                logger.error("%s: link error:\n%s", name, glGetProgramInfoLog(program).decode())
                glDeleteProgram(program)
                return None
            return program
        except Exception:
            logger.exception("%s: unexpected error while compiling", name)
            return None
        finally:
            for shader in compiled:
                glDeleteShader(shader)

    # HOT RELOAD
    @classmethod
    def reload_marked(cls) -> None:
        """Compile again every program a changed file marked, once its files have been quiet for SETTLE; on the main
        thread, at the top of each frame. A file that reads empty, a save under way, is tried again next frame."""
        if not Shader._marked:
            return
        now: float = perf_counter()
        with Shader._lock:
            marked: list[tuple[type, _Program]] = [(shader_class, entry) for shader_class, entry in Shader._programs.items()
                                                   if entry.marked and now - entry.marked_at >= SETTLE]
            for _, entry in marked:
                entry.marked = False
            Shader._marked = any(entry.marked for entry in Shader._programs.values())
        for shader_class, entry in marked:
            sources = shader_class._read_sources(entry)
            if any(not source for _, _, source, _ in sources) and entry.retries < READ_RETRIES:
                entry.retries += 1
                with Shader._lock:
                    entry.marked = True
                    Shader._marked = True
                continue
            entry.retries = 0
            compiled: bool = shader_class._compile_into(entry, sources)
            if compiled:
                logger.info("%s reloaded", shader_class.__name__)
            else:
                logger.error("%s reload failed; the previous program stays", shader_class.__name__)

    @staticmethod
    def read_shader_source(filename: str, read: list[Path] | None = None,
                           stamps: dict[Path, tuple[int, int] | None] | None = None,
                           _included: frozenset[str] = frozenset()) -> str:
        """Read shader source code from file; a line `#include "name.glsl"` is replaced by that file, beside
        the including one or else in `modules/gl/shaders` (as `position.glsl`), so passes can share one rule (each
        file included once per shader). `read` gets each file read, the first one first; an included file is framed by
        `#line` directives numbering it by its place there, so a compile error names its file (`name_error_lines`);
        `stamps` gets each file's `file_stamp`, taken before it is read, so a save during the read still counts."""
        if read is None:
            read = []
        index: int = len(read)
        read.append(Path(filename))
        if stamps is not None:
            stamps[Path(filename)] = file_stamp(Path(filename))
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                source = file.read()
        except FileNotFoundError:
            return ''
        if '#include' not in source:
            return source
        folder: Path = Path(filename).parent
        lines: list[str] = []
        for number, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith('#include "') and stripped.endswith('"'):
                name: str = stripped[len('#include "'):-1]
                path = str(folder / name)
                if not Path(path).exists():
                    path = str(SHARED_SHADERS / name)
                if path in _included:
                    lines.append('')                    # the line count stays
                    continue
                child: int = len(read)
                included = Shader.read_shader_source(path, read, stamps, _included | {path})
                if not included:
                    logger.error("%s: include %s not found", filename, name)
                    lines.append('')
                    continue
                lines.extend((f'#line 1 {child}', included.rstrip('\n'), f'#line {number + 1} {index}'))
            else:
                lines.append(line)
        return '\n'.join(lines)

    @classmethod
    def enable_hot_reload(cls) -> None:
        """Watch the folders of every shader, so a changed file marks its programs for the next reload point; a
        program made later is watched when it is made. Calling it again changes nothing."""
        with Shader._lock:
            if Shader._hot_reload_enabled:
                return
            Shader._hot_reload_enabled = True
            for entry in Shader._programs.values():
                Shader._watch_files(entry.files)
            logger.info("Hot reload enabled for %s directories", len(Shader._directory_observers))

    @classmethod
    def disable_hot_reload(cls) -> None:
        """Stop watching; the observer threads end."""
        with Shader._lock:
            Shader._hot_reload_enabled = False
            observers = list(Shader._directory_observers.values())
            Shader._directory_observers.clear()
        for observer in observers:
            observer.stop()
            observer.join(timeout=2.0)
        logger.info("Hot reload disabled")

    @staticmethod
    def _watch_files(files: dict[Path, tuple[int, int] | None]) -> None:
        """Start watching the folders of these files, each once; under the lock."""
        for directory in {path.parent for path in files}:
            if directory not in Shader._directory_observers:
                Shader._directory_observers[directory] = monitor_path(str(directory), Shader._on_file_changed)

    @staticmethod
    def _on_file_changed(filepath: str) -> None:
        """The watcher's thread: mark each program made from the file, unless its write time and size are as when
        that program read it: Windows reports a read as a change too (the read's last access time)."""
        path = Path(filepath)
        stamp: tuple[int, int] | None = file_stamp(path)
        now: float = perf_counter()
        with Shader._lock:
            for entry in Shader._programs.values():
                if path in entry.files and (stamp is None or entry.files[path] != stamp):
                    entry.marked = True
                    entry.marked_at = now
                    Shader._marked = True
