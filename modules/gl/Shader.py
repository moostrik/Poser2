from OpenGL.GL import * # type: ignore
from OpenGL.GL import shaders
import time
import inspect
import threading
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FileModifiedHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback
        self.last_modified: float = time.time()

    def on_modified(self, event):
        if time.time() - self.last_modified < 0.5:
            return
        self.last_modified = time.time()
        if not event.is_directory:
            self.callback(event.src_path)

def monitor_path(path, callback):
    event_handler = FileModifiedHandler(callback)
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=False)
    observer.start()
    return observer

def draw_quad() -> None :
    glBegin(GL_QUADS)
    glTexCoord2f( 0.0,  0.0)
    glVertex2f(  -1.0, -1.0)
    glTexCoord2f( 1.0,  0.0)
    glVertex2f(   1.0, -1.0)
    glTexCoord2f( 1.0,  1.0)
    glVertex2f(   1.0,  1.0)
    glTexCoord2f( 0.0,  1.0)
    glVertex2f(  -1.0,  1.0)
    glEnd()

class Shader():
    # Class-level hot-reload management (one observer per directory)
    _directory_observers = {}  # {Path: Observer}
    _hot_reload_enabled = False
    _monitored_shaders = []
    _observer_lock = threading.Lock()

    # Shader file suffixes
    VERTEX_SUFFIX = '.vert'
    FRAGMENT_SUFFIX = '.frag'

    # Embedded generic shaders as fallbacks
    GENERIC_VERTEX_SHADER = """#version 460 core

in vec2 position;
out vec2 texCoord;

void main() {
    texCoord = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

    GENERIC_FRAGMENT_SHADER = """#version 460 core

uniform sampler2D tex0;
uniform sampler2D tex1;
uniform float fade;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 texel0 = texture(tex0, texCoord);
    vec4 texel1 = texture(tex1, texCoord);
    fragColor = mix(texel0, texel1, fade);
}
"""

    def __init__(self, shader_name: str = '') -> None:
        """Initialize shader and discover shader files."""
        self.allocated: bool = False
        self.shader_name: str = shader_name or self.__class__.__name__
        self.shader_program: shaders.ShaderProgram | None = None
        self.need_reload: bool = False
        self._reload_lock = threading.Lock()

        # Discover shader files now (not in allocate)
        shader_name_normalized = self.shader_name.lower()
        self.shader_dir = self._discover_shader_dir()
        self.vertex_file_path = self._find_shader_file(self.shader_dir, shader_name_normalized, self.VERTEX_SUFFIX)
        self.fragment_file_path = self._find_shader_file(self.shader_dir, shader_name_normalized, self.FRAGMENT_SUFFIX)

    def allocate(self) -> None:
        """Compile shaders and create OpenGL resources. Safe to call multiple times."""
        if self.allocated:
            return

        # Register shader and watch its directory if hot-reload is enabled
        with Shader._observer_lock:
            if self not in Shader._monitored_shaders:
                Shader._monitored_shaders.append(self)

            # Start watching this shader's directory if hot-reload is enabled
            if Shader._hot_reload_enabled and self.shader_dir:
                Shader._watch_directory(self.shader_dir)

        # Now try to compile - sets self.allocated on success
        self.allocated = self._compile_shaders()

    def deallocate(self) -> None:
        """Clean up OpenGL resources and unregister from hot-reload."""
        # Don't guard on allocated - we need to cleanup registration regardless
        self.allocated = False

        # Unregister from shared observer
        with Shader._observer_lock:
            if self in Shader._monitored_shaders:
                Shader._monitored_shaders.remove(self)

        # Cleanup OpenGL resources
        if self.shader_program is not None:
            glDeleteProgram(self.shader_program)
        self.shader_program = None

    def reload(self) -> bool:
        """Reload shader if marked for reload. Returns True if reloaded."""
        if not self.need_reload:
            return False

        with self._reload_lock:
            self.need_reload = False
            self.allocated = self._compile_shaders(verbose=True)
            return True  # We attempted reload

    def _on_file_changed(self, file_path: Path) -> None:
        """Mark shader for reload when its file changes."""
        if (self.vertex_file_path and file_path == self.vertex_file_path or
            self.fragment_file_path and file_path == self.fragment_file_path):
            self.need_reload = True  # Atomic bool, no lock needed for setting

    def _compile_shaders(self, verbose: bool = False) -> bool:
        """Load and compile shader sources into OpenGL program. Returns True if successful."""
        # Clear existing OpenGL resources before recompiling
        if self.shader_program is not None:
            glDeleteProgram(self.shader_program)
            self.shader_program = None

        # Try to load vertex shader from file, fallback to embedded generic
        vertex_source: str = ''
        if self.vertex_file_path:
            vertex_source = self.read_shader_source(str(self.vertex_file_path))
        if not vertex_source:
            vertex_source = self.GENERIC_VERTEX_SHADER
            if verbose:
                logging.info(f"{self.shader_name}: Using generic vertex shader")

        # Try to load fragment shader from file, fallback to embedded generic
        fragment_source: str = ''
        if self.fragment_file_path:
            fragment_source = self.read_shader_source(str(self.fragment_file_path))
        if not fragment_source:
            fragment_source = self.GENERIC_FRAGMENT_SHADER
            if verbose:
                logging.info(f"{self.shader_name}: Using generic fragment shader")

        # Compile shaders
        vertex_shader = None
        fragment_shader = None
        try:
            vertex_shader = shaders.compileShader(vertex_source, shaders.GL_VERTEX_SHADER) # type: ignore
        except shaders.ShaderCompilationError as e:
            logging.error(f"{self.shader_name} VERTEX SHADER ERROR: {e}")
            return False

        try:
            fragment_shader = shaders.compileShader(fragment_source, shaders.GL_FRAGMENT_SHADER) # type: ignore
        except shaders.ShaderCompilationError as e:
            logging.error(f"{self.shader_name} FRAGMENT SHADER ERROR: {e}")
            return False

        if not vertex_shader or not fragment_shader:
            return False

        # Link shader program
        try:
            self.shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
            if verbose:
                logging.info(f"{self.shader_name} loaded successfully")
            return True
        except Exception as e:
            logging.error(f"{self.shader_name} PROGRAM LINKING ERROR: {e}")
            return False


    # STATIC METHODS
    @staticmethod
    def read_shader_source(filename: str) -> str:
        """Read shader source code from file."""
        try:
            with open(filename, 'r') as file:
                return file.read()
        except FileNotFoundError:
            return ''

    @staticmethod
    def _discover_shader_dir() -> Path:
        """Discover shader directory by inspecting the calling class file location."""
        try:
            # Get the file path of the class that instantiated this shader
            frame = inspect.currentframe()
            if frame and frame.f_back and frame.f_back.f_back:
                caller_frame = frame.f_back.f_back
                caller_file = caller_frame.f_globals.get('__file__')
                if caller_file:
                    return Path(caller_file).parent
        except:
            pass

        # Fallback to default shader directory
        return Path('modules/gl/shaders')

    @staticmethod
    def _find_shader_file(shader_dir: Path | None, shader_name_normalized: str, suffix: str) -> Path | None:
        """Find shader file in the same directory as the Python class file."""
        if not shader_dir:
            return None

        shader_path = shader_dir / f"{shader_name_normalized}{suffix}"
        return shader_path if shader_path.exists() else None


    # CLASS METHODS FOR HOT-RELOAD MANAGEMENT
    @classmethod
    def enable_hot_reload(cls) -> None:
        """Enable hot-reload for all shaders. Watches each shader's directory."""
        with cls._observer_lock:
            cls._hot_reload_enabled = True
            # Start watching all unique shader directories
            for shader in cls._monitored_shaders:
                if shader.shader_dir:
                    cls._watch_directory(shader.shader_dir)
            if cls._directory_observers:
                logging.info(f"Hot-reload enabled for {len(cls._directory_observers)} directories")

    @classmethod
    def _watch_directory(cls, shader_dir: Path) -> None:
        """Start watching a directory if not already watched."""
        if shader_dir not in cls._directory_observers:
            observer = monitor_path(str(shader_dir), cls._on_any_file_changed)
            cls._directory_observers[shader_dir] = observer
            logging.info(f"Watching shader directory: {shader_dir}")

    @classmethod
    def disable_hot_reload(cls) -> None:
        """Disable hot-reload and stop all observer threads."""
        with cls._observer_lock:
            cls._hot_reload_enabled = False
            # Stop all directory observers
            for observer in cls._directory_observers.values():
                observer.stop()
                observer.join(timeout=2.0)
            cls._directory_observers.clear()
            logging.info("Hot-reload disabled")

    @classmethod
    def _on_any_file_changed(cls, filepath: str) -> None:
        """Called when any shader file changes - notify relevant shaders."""
        file_path = Path(filepath)
        with cls._observer_lock:
            for shader in cls._monitored_shaders:
                shader._on_file_changed(file_path)