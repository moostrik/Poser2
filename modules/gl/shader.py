import inspect
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from OpenGL.GL import GL_COMPILE_STATUS, GL_FRAGMENT_SHADER, GL_LINK_STATUS, GL_VERTEX_SHADER, glAttachShader, glCompileShader, glCreateProgram, glCreateShader, glDeleteProgram, glDeleteShader, glGetProgramInfoLog, glGetProgramiv, glGetShaderInfoLog, glGetShaderiv, glGetUniformLocation, glLinkProgram, glShaderSource
from watchdog.observers.api import BaseObserver

from .file_watcher import monitor_path

logger = logging.getLogger(__name__)

SHARED_SHADERS: Path = Path(__file__).parent / "shaders"     # the shared includes and `_generic.vert`


@dataclass
class _Program:
    """A shader class's compiled program, shared by all its objects: compiled at the first `allocate`, freed at the
    last `deallocate`, compiled again when a file it was made from changes."""
    program: int | None = None
    users: int = 0
    files: set[Path] = field(default_factory=set)       # the files it was compiled from
    uniforms: dict[str, int] = field(default_factory=dict)
    marked: bool = False                                # a file changed: compiled again at the next reload point


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
                Shader._watch_directory(cls._source_dir())
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
        """A uniform's location in the class's program, looked up once."""
        entry: _Program = Shader._programs[type(self)]
        location: int | None = entry.uniforms.get(name)
        if location is None:
            location = int(glGetUniformLocation(entry.program, name))
            entry.uniforms[name] = location
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
    def _compile_into(cls, entry: _Program) -> bool:
        """Compile the class's program into `entry`; on failure the program it had stays."""
        stages: list[tuple[int, Path]] = cls._stages()
        entry.files = {path for _, path in stages}
        program: int | None = cls._compile(stages)
        if program is None:
            return False
        if entry.program is not None:
            glDeleteProgram(entry.program)
        entry.program = program
        entry.uniforms.clear()                          # locations may have changed
        return True

    @classmethod
    def _compile(cls, stages: list[tuple[int, Path]]) -> int | None:
        """A linked program from the stages' files, or None after logging what failed."""
        name: str = cls.__name__
        compiled: list[int] = []
        try:
            for stage, path in stages:
                source: str = cls.read_shader_source(str(path))
                if not source:
                    logger.error("%s: no shader source at %s", name, path)
                    return None
                shader: int = cast(int, glCreateShader(stage))       # the stub says None; it is the id
                compiled.append(shader)
                glShaderSource(shader, source)
                glCompileShader(shader)
                if not glGetShaderiv(shader, GL_COMPILE_STATUS):
                    logger.error("%s: compile error in %s:\n%s", name, path.name, glGetShaderInfoLog(shader).decode())
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
        """Compile again every program a changed file marked; on the main thread, at the top of each frame."""
        if not Shader._marked:
            return
        with Shader._lock:
            Shader._marked = False
            marked: list[tuple[type, _Program]] = [(shader_class, entry) for shader_class, entry in Shader._programs.items()
                                                   if entry.marked]
            for _, entry in marked:
                entry.marked = False
        for shader_class, entry in marked:
            compiled: bool = shader_class._compile_into(entry)
            if compiled:
                logger.info("%s reloaded", shader_class.__name__)
            else:
                logger.error("%s reload failed; the previous program stays", shader_class.__name__)

    @staticmethod
    def read_shader_source(filename: str, _included: frozenset[str] = frozenset()) -> str:
        """Read shader source code from file; a line `#include "name.glsl"` is replaced by that file, beside
        the including one or else in `modules/gl/shaders` (as `position.glsl`), so passes can share one rule (each
        file included once per shader)."""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                source = file.read()
        except FileNotFoundError:
            return ''
        if '#include' not in source:
            return source
        folder: Path = Path(filename).parent
        lines: list[str] = []
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith('#include "') and stripped.endswith('"'):
                name: str = stripped[len('#include "'):-1]
                path = str(folder / name)
                if not Path(path).exists():
                    path = str(SHARED_SHADERS / name)
                if path in _included:
                    continue
                included = Shader.read_shader_source(path, _included | {path})
                if not included:
                    logger.error("%s: include %s not found", filename, name)
                lines.append(included)
            else:
                lines.append(line)
        return '\n'.join(lines)

    @classmethod
    def enable_hot_reload(cls) -> None:
        """Watch the folders of every shader, so a changed file marks its programs for the next reload point."""
        with Shader._lock:
            Shader._hot_reload_enabled = True
            for shader_class in Shader._programs:
                Shader._watch_directory(shader_class._source_dir())
            if Shader._directory_observers:
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
    def _watch_directory(directory: Path) -> None:
        """Start watching a folder, once; under the lock."""
        if directory not in Shader._directory_observers:
            Shader._directory_observers[directory] = monitor_path(str(directory), Shader._on_file_changed)

    @staticmethod
    def _on_file_changed(filepath: str) -> None:
        """The watcher's thread: mark each program made from the file."""
        path = Path(filepath)
        with Shader._lock:
            for entry in Shader._programs.values():
                if path in entry.files:
                    entry.marked = True
                    Shader._marked = True
