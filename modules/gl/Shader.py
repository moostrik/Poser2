from OpenGL.GL import * # type: ignore
from OpenGL.GL import shaders
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FileModifiedHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback
        self.last_modified: float = time.time()

    def on_modified(self, event):
        if time.time() - self.last_modified < 1.0:
            return
        self.last_modified = time.time()
        if not event.is_directory:
            self.callback(event.src_path)

def monitor_path(path, callback):
    event_handler = FileModifiedHandler(callback)
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=False)
    observer.start()

def read_shader_source(filename) -> str:
    try:
        with open(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return ''

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
    def __init__(self) -> None :
        self.allocated: bool = False
        self.shader_name: str = ''
        self.shader_path: str = 'modules/gl/shaders/'
        self.vertex_suffix: str = '.vert'
        self.fragment_suffix: str = '.frag'
        self.vertex_file_name: str = ''
        self.vertex_generic_file_name: str = f'{self.shader_path}_Generic{self.vertex_suffix}'
        self.fragment_file_name: str = ''
        self.vertex_shader = None
        self.fragment_shader = None
        self.shader_program: shaders.ShaderProgram | None = None
        self.need_reload: bool = False

    def allocate(self, shader_name: str, monitor_file = False) -> None :
        self.shader_name = shader_name
        self.vertex_file_name = f'{self.shader_path}{self.shader_name}{self.vertex_suffix}'
        self.fragment_file_name = f'{self.shader_path}{self.shader_name}{self.fragment_suffix}'

        if monitor_file:
            monitor_path(self.shader_path, self.file_changed)
        self.load()

    def deallocate(self) -> None :
        if not self.allocated:
            return
        self.allocated = False
        if self.shader_program is not None:
            glDeleteProgram(self.shader_program)
        self.shader_program = None
        self.vertex_shader = None
        self.fragment_shader = None

    def file_changed(self, file: str) -> None:
        if (file == self.vertex_file_name or
            file == self.vertex_generic_file_name or
            file == self.fragment_file_name):
            self.need_reload = True

    def load(self, verbose: bool = False) -> None :
        self.unload()

        vertex_source: str = read_shader_source(self.vertex_file_name)
        if not vertex_source: vertex_source = read_shader_source(self.vertex_generic_file_name)
        fragment_source: str = read_shader_source(self.fragment_file_name)

        if vertex_source == '' or fragment_source == '':
            print(f'files for {self.shader_name} not found', self.vertex_file_name)
            self.allocated = False
            return

        vertex_shader = None
        fragment_shader = None
        try:
            vertex_shader = shaders.compileShader(vertex_source, shaders.GL_VERTEX_SHADER) # type: ignore
        except shaders.ShaderCompilationError as e:
            print(f"{self.shader_name} VERTEX SHADER ERROR {e}")
        try:
            fragment_shader = shaders.compileShader(fragment_source, shaders.GL_FRAGMENT_SHADER) # type: ignore
        except shaders.ShaderCompilationError as e:
            print(f"{self.shader_name} FRAGMENT SHADER ERROR {e}")

        if not vertex_shader or not fragment_shader:
            self.allocated = False
            return

        self.vertex_shader = vertex_shader
        self.fragment_shader = fragment_shader

        self.shader_program = shaders.compileProgram(self.vertex_shader, self.fragment_shader)
        self.allocated = True
        if verbose:
            print(self.shader_name, 'loaded')

    def unload(self) -> None :
        if not self.allocated:
            return
        self.allocated = False
        if self.shader_program != None: glDeleteProgram(self.shader_program)
        self.shader_program = None
        self.vertex_shader = None
        self.fragment_shader = None

    def use(self) -> None:
        if self.need_reload :
            self.need_reload = False
            self.load(True)
