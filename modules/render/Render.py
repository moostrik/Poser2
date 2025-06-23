from OpenGL.GL import * # type: ignore
import OpenGL.GLUT as glut
from threading import Lock
import numpy as np
from enum import Enum, IntEnum, auto
import math

from modules.gl.RenderWindow import RenderWindow
from modules.gl.Texture import Texture
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Mesh import Mesh
from modules.gl.Utils import lfo, fit, fill
from modules.gl.Image import Image
from modules.gl.Shader import Shader
from modules.gl.shaders.WS_Angles import WS_Angles
from modules.gl.shaders.WS_Lines import WS_Lines

from modules.cam.depthcam.Definitions import Tracklet, Rect, Point3f, FrameType
from modules.pose.detection.Definitions import Pose, Indices
from modules.person.Person import Person, PersonColor
from modules.person.Definitions import *
from modules.av.Definitions import AvOutput

from modules.Settings import Settings

class ImageType(Enum):
    TOT = 0
    CAM = 1
    SIM = 2
    PSN = 3
    VIS = 4

class SimType(Enum):
    MAP = 0

class VisType(IntEnum):
    LINES = 0
    LIGHTS = 1

Composition_Subdivision = dict[ImageType, dict[int, tuple[int, int, int, int]]]

class Render(RenderWindow):
    def __init__(self, settings: Settings) -> None:

        self.num_cams: int = settings.camera_num
        self.num_sims: int = len(SimType)
        self.num_viss: int = len(VisType)
        self.num_persons: int = settings.pose_num

        self.cam_images: dict[int, Image] = {}
        self.psn_images: dict[int, Image] = {}

        self.cam_fbos: dict[int, Fbo] = {}
        self.sim_fbos: dict[int, Fbo] = {}
        self.vis_fbos: dict[int, Fbo] = {}
        self.psn_fbos: dict[int, Fbo] = {}
        self.pose_meshes: dict[int, Mesh] = {}

        self.all_images: list[Image] = []
        self.all_fbos: list[Fbo | SwapFbo] = []
        self.all_meshes: list[Mesh] = []
        self.all_shaders: list[Shader] = []

        for i in range(self.num_cams):
            self.cam_images[i] = Image()
            self.all_images.append(self.cam_images[i])
            self.cam_fbos[i] = Fbo()
            self.all_fbos.append(self.cam_fbos[i])

        for i in range(self.num_sims):
            self.sim_fbos[i] = Fbo()
            self.all_fbos.append(self.sim_fbos[i])

        for i in range(self.num_viss):
            self.vis_fbos[i] = Fbo()
            self.all_fbos.append(self.vis_fbos[i])

        for i in range(self.num_persons):
            self.psn_images[i] = Image()
            self.all_images.append(self.psn_images[i])

            self.psn_fbos[i] = Fbo()
            self.all_fbos.append(self.psn_fbos[i])

            self.pose_meshes[i] = Mesh()
            self.pose_meshes[i].set_indices(Indices)
            self.all_meshes.append(self.pose_meshes[i])

        self.composition: Composition_Subdivision = self.make_composition_subdivision(settings.render_width, settings.render_height, self.num_cams, self.num_sims, self.num_persons, self.num_viss)
        super().__init__(self.composition[ImageType.TOT][0][2], self.composition[ImageType.TOT][0][3], settings.render_title, settings.render_fullscreen, settings.render_v_sync, settings.render_fps, settings.render_x, settings.render_y)

        self.allocated = False

        self.input_mutex: Lock = Lock()
        self.input_tracklets: dict[int, dict[int, Tracklet]] = {}   # cam_id -> track_id -> Tracklet
        self.input_persons: dict[int, Person | None] = {}           # person_id -> Person
        for i in range(self.num_persons):
            self.input_tracklets[i] = {}
            self.input_persons[i] = None

        self.vis_width: int = settings.light_resolution
        self.vis_height: int = 1
        self.vis_image = Image()
        self.all_images.append(self.vis_image)
        self.vis_line_shader: WS_Lines = WS_Lines()
        self.vis_angle_shader: WS_Angles = WS_Angles()
        self.all_shaders.append(self.vis_line_shader)
        self.all_shaders.append(self.vis_angle_shader)

    def reshape(self, width, height) -> None: # override
        super().reshape(width, height)
        self.composition = self.make_composition_subdivision(width, height, self.num_cams, self.num_sims, self.num_persons, self.num_viss)

        for key in self.cam_fbos.keys():
            x, y, w, h = self.composition[ImageType.CAM][key]
            self.cam_fbos[key].allocate(w, h, GL_RGBA)

        for key in self.sim_fbos.keys():
            x, y, w, h = self.composition[ImageType.SIM][key]
            self.sim_fbos[key].allocate(w, h, GL_RGBA)

        for key in self.psn_fbos.keys():
            x, y, w, h = self.composition[ImageType.PSN][key]
            self.psn_fbos[key].allocate(w, h, GL_RGBA)

    def draw(self) -> None: # override
        if not self.allocated:
            self.reshape(self.window_width, self.window_height)
            for s in self.all_shaders:
                s.allocate(True) # type: ignore
            for fbo in self.vis_fbos.values():
                fbo.allocate(self.vis_width, self.vis_height, GL_RGBA32F)
            self.allocated = True

        self.update_pose_meshes()

        self.draw_cameras()
        self.draw_sims()
        self.draw_lights()
        self.draw_persons()

        self.draw_composition()

    def update_pose_meshes(self) -> None:
        for i in range(self.num_persons):
            person: Person | None = self.get_person(i, clear=False)
            if person is not None:
                poses: list[Pose] | None = person.pose
                if poses is not None and len(poses) > 0:
                    self.pose_meshes[i].set_vertices(poses[0].getVertices())
                    self.pose_meshes[i].set_colors(poses[0].getColors(threshold=0.0))
                    self.pose_meshes[i].update()

    def draw_cameras(self) -> None:
        for i in range(self.num_cams):
            image: Image = self.cam_images[i]
            image.update()
            fbo: Fbo = self.cam_fbos[i]
            tracklets: dict[int, Tracklet] = self.get_tracklets(i)
            persons: list[Person] = self.get_persons_for_cam(i)

            self.setView(fbo.width, fbo.height)
            fbo.begin()

            image.draw(0, 0, fbo.width, fbo.height)
            for tracklet in tracklets.values():
                self.draw_tracklet(tracklet, 0, 0, fbo.width, fbo.height)
            for person in persons:
                if person.active:
                    roi: Rect | None = person.pose_roi
                    mesh: Mesh = self.pose_meshes[person.id]
                    if roi is not None and mesh.isInitialized():
                        x, y, w, h = roi.x, roi.y, roi.width, roi.height
                        x *= fbo.width
                        y *= fbo.height
                        w *= fbo.width
                        h *= fbo.height

                        self.draw_person(person, mesh, x, y, w, h, False, True, False)

            fbo.end()

    def draw_sims(self) -> None:
        for i in range(self.num_sims):
            fbo: Fbo = self.sim_fbos[i]
            self.setView(fbo.width, fbo.height)
            if i == SimType.MAP.value:
                self.draw_map_positions(self.input_persons, self.num_cams, fbo)

    def draw_lights(self) -> None:
        self.vis_image.update()

        for i in range(self.num_viss):
            fbo: Fbo = self.vis_fbos[i]
            self.setView(fbo.width, fbo.height)
            fbo.begin()
            if i == VisType.LINES.value:
                self.draw_light_lines(fbo, self.vis_image, self.vis_line_shader)
            elif i == VisType.LIGHTS.value:
                self.draw_light_angles(fbo, self.vis_image, self.vis_angle_shader)
            fbo.end()

    def draw_persons(self) -> None:
        for i in range(self.num_persons):
            fbo: Fbo = self.psn_fbos[i]
            person: Person | None = self.get_person(i)
            if person is None or person.img is None or not person.active:
                continue

            image: Image = self.psn_images[i]
            image.set_image(person.img)
            image.update()

            self.setView(fbo.width, fbo.height)
            fbo.begin()

            if person is not None and person.active:
                image.draw(0, 0, fbo.width, fbo.height)
                mesh: Mesh = self.pose_meshes[person.id]
                self.draw_person(person, mesh, 0, 0, fbo.width, fbo.height, False, True, True)

            fbo.end()

    def draw_composition(self) -> None:
        self.setView(self.window_width, self.window_height)
        for i in range(self.num_cams):
            x, y, w, h = self.composition[ImageType.CAM][i]
            self.cam_fbos[i].draw(x, y, w, h)

        for i in range(self.num_sims):
            x, y, w, h = self.composition[ImageType.SIM][i]
            self.sim_fbos[i].draw(x, y, w, h)

        for i in range(self.num_persons):
            x, y, w, h = self.composition[ImageType.PSN][i]
            self.psn_fbos[i].draw(x, y, w, h)

        for i in range(self.num_viss):
            x, y, w, h = self.composition[ImageType.VIS][i]
            self.vis_fbos[i].draw(x, y, w, h)

    # SETTERS AND GETTERS
    def set_cam_image(self, cam_id: int, frame_type: FrameType, image: np.ndarray) -> None :
        self.cam_images[cam_id].set_image(image)

    def get_tracklets(self, cam_id: int, clear: bool = False) -> dict[int, Tracklet]:
        with self.input_mutex:
            ret_person: dict[int, Tracklet] =  self.input_tracklets[cam_id].copy()
            if clear:
                self.input_tracklets[cam_id].clear()
            return ret_person
    def add_tracklet(self, cam_id: int, tracklet: Tracklet) -> None :
        with self.input_mutex:
            self.input_tracklets[cam_id][tracklet.id] = tracklet

    def get_person(self, id: int, clear: bool = False) -> Person | None:
        with self.input_mutex:
            ret_person: Person | None = self.input_persons[id]
            if clear:
                self.input_persons[id] = None
            return ret_person
    def get_persons_for_cam(self, cam_id: int) -> list[Person]:
        with self.input_mutex:
            persons: list[Person] = []
            for person in self.input_persons.values():
                if person is not None and person.cam_id == cam_id:
                    persons.append(person)
            return persons
    def add_person(self, person: Person) -> None:
        with self.input_mutex:
            self.input_persons[person.id] = person

    def set_av(self, value: AvOutput) -> None:
        self.vis_image.set_image(value.img)
        self.av_angle = value.angle

    # STATIC METHODS
    @staticmethod
    def draw_tracklet(tracklet: Tracklet, x: float, y: float, w: float, h: float) -> None:
        if tracklet.status == Tracklet.TrackingStatus.REMOVED:
            return

        x = x + tracklet.roi.x * w
        y = y + tracklet.roi.y * h
        w = tracklet.roi.width * w
        h = tracklet.roi.height * h

        r: float = 1.0
        g: float = 1.0
        b: float = 1.0
        a: float = min(tracklet.age / 100.0, 0.33)
        if tracklet.status == Tracklet.TrackingStatus.NEW:
            r, g, b, a = (1.0, 1.0, 1.0, 1.0)
        if tracklet.status == Tracklet.TrackingStatus.TRACKED:
            r, g, b, a = (0.0, 1.0, 0.0, a)
        if tracklet.status == Tracklet.TrackingStatus.LOST:
            r, g, b, a = (1.0, 0.0, 0.0, a)
        if tracklet.status == Tracklet.TrackingStatus.REMOVED:
            r, g, b, a = (1.0, 0.0, 0.0, 1.0)

        glColor4f(r, g, b, a)   # Set color
        glBegin(GL_QUADS)       # Start drawing a quad
        glVertex2f(x, y)        # Bottom left
        glVertex2f(x, y + h)    # Bottom right
        glVertex2f(x + w, y + h)# Top right
        glVertex2f(x + w, y)    # Top left
        glEnd()                 # End drawing
        glColor4f(1.0, 1.0, 1.0, 1.0)  # Reset color

        string: str
        x += 9
        y += 14
        string = f'ID: {tracklet.id}'
        RenderWindow.draw_string(x, y, string)
        y += 14
        string = f'Age: {tracklet.age}'
        RenderWindow.draw_string(x, y, string)

        glFlush()               # Render now

    @staticmethod
    def draw_raw_positions(persons: dict[int, Person | None], num_cams: int, fbo: Fbo) -> None:
        fbo.begin()
        glClearColor(0.1, 0.0, 0.0, 1.0)  # Set background color to black
        glClear(GL_COLOR_BUFFER_BIT)       # Actually clear the buffer!

        for person in persons.values():
            if person is None or not person.active:
                continue

            id: int = person.cam_id
            w: float = person.tracklet.roi.width * fbo.width / num_cams
            h: float = person.tracklet.roi.height * fbo.height
            x: float = person.tracklet.roi.x * fbo.width / num_cams + (id * fbo.width / num_cams)
            y: float = person.tracklet.roi.y * fbo.height
            # y: float = (fbo.height - h) * 0.5
            color = PersonColor(person.id, aplha=0.5)

            glColor4f(*color)  # Reset color
            glBegin(GL_QUADS)       # Start drawing a quad
            glVertex2f(x, y)        # Bottom left
            glVertex2f(x, y + h)    # Bottom right
            glVertex2f(x + w, y + h)# Top right
            glVertex2f(x + w, y)    # Top left
            glEnd()                 # End drawing
            glColor4f(1.0, 1.0, 1.0, 1.0)  # Reset color

        fbo.end()
        glFlush()
        return

    @staticmethod
    def draw_map_positions(persons: dict[int, Person | None], num_cams: int, fbo: Fbo) -> None:
        fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)  # Set background color to black
        glClear(GL_COLOR_BUFFER_BIT)       # Actually clear the buffer!

        for person in persons.values():
            if person is None or not person.active:
                continue

            w: float = person.tracklet.roi.width * fbo.width / num_cams
            h: float = person.tracklet.roi.height * fbo.height
            x: float = person.world_angle / 360.0 * fbo.width
            y: float = person.tracklet.roi.y * fbo.height
            color: list[float] = PersonColor(person.id, aplha=0.9)
            if person.overlap == True:
                color[3] = 0.3

            glColor4f(*color)  # Reset color
            glBegin(GL_QUADS)       # Start drawing a quad
            glVertex2f(x, y)        # Bottom left
            glVertex2f(x, y + h)    # Bottom right
            glVertex2f(x + w, y + h)# Top right
            glVertex2f(x + w, y)    # Top left
            glEnd()                 # End drawing
            glColor4f(1.0, 1.0, 1.0, 1.0)  # Reset color

            string: str
            x += 9
            y += 14
            string = f'A: {person.world_angle:.1f}'
            RenderWindow.draw_string(x, y, string)
            y += 14
            string = f'L: {person.local_angle:.1f}'
            RenderWindow.draw_string(x, y, string)

        fbo.end()
        glFlush()

    @staticmethod
    def draw_light_lines(fbo: Fbo, img: Image, shader: WS_Lines) -> None:
        shader.use(fbo.fbo_id, img.tex_id)
        glFlush()

    @staticmethod
    def draw_light_angles(fbo: Fbo, img: Image, shader: WS_Angles) -> None:
        shader.use(fbo.fbo_id, img.tex_id, img.width)
        glFlush()

    @staticmethod
    def draw_person(person: Person, pose: Mesh, x: float, y: float, w: float, h: float, draw_box = False, draw_pose = False, draw_text = False) -> None:
        if draw_box:
            r: float = 0.0
            g: float = 0.0
            b: float = 0.0
            a: float = 0.2

            glColor4f(r, g, b, a)
            glBegin(GL_QUADS)
            glVertex2f(x, y)        # Bottom left
            glVertex2f(x, y + h)    # Bottom right
            glVertex2f(x + w, y + h)# Top right
            glVertex2f(x + w, y)    # Top left
            glEnd()                 # End drawing
            glColor4f(1.0, 1.0, 1.0, 1.0)  # Reset color

        if draw_pose and pose.isInitialized():
            pose.draw(x, y, w, h)

        if draw_text:
            string: str = f'ID: {person.id} Cam: {person.cam_id} Age: {person.last_time - person.start_time:.0f}'
            x += 9
            y += 15
            RenderWindow.draw_string(x, y, string)

        glFlush()               # Render now

    @staticmethod
    def make_composition_subdivision(dst_width: int, dst_height: int,
                                     num_cams: int, num_sims: int, num_persons: int, num_viss: int,
                                     cam_aspect_ratio: float = 16.0 / 9.0,
                                     sim_aspect_ratio: float = 10.0,
                                     psn_aspect_ratio: float = 1.0,
                                     vis_aspect_ratio: float = 20.0) -> Composition_Subdivision:

        ret: Composition_Subdivision = {}
        ret[ImageType.TOT] = {}
        ret[ImageType.CAM] = {}
        ret[ImageType.SIM] = {}
        ret[ImageType.PSN] = {}
        ret[ImageType.VIS] = {}

        cams_per_row: int = 4
        cam_rows: int = math.ceil(num_cams / cams_per_row)
        cam_columns: int = 0 if cam_rows == 0 else math.ceil(num_cams / cam_rows)

        dst_aspect_ratio: float = dst_width / dst_height
        cam_grid_aspect_ratio: float = 100.0 if cam_rows == 0 else cam_aspect_ratio * cam_columns / cam_rows
        sim_grid_aspect_ratio: float = sim_aspect_ratio / num_sims
        vis_grid_aspect_ratio: float = vis_aspect_ratio / num_viss
        psn_grid_aspect_ratio: float = psn_aspect_ratio * num_persons
        tot_aspect_ratio: float = 1.0 / (1.0 / cam_grid_aspect_ratio + 1.0 / sim_grid_aspect_ratio + 1.0 / vis_grid_aspect_ratio + 1.0 / psn_grid_aspect_ratio)

        fit_width: float
        fit_height: float
        if tot_aspect_ratio > dst_aspect_ratio:
            fit_width = dst_width
            fit_height = dst_width / tot_aspect_ratio
        else:
            fit_width = dst_height * tot_aspect_ratio
            fit_height = dst_height
        fit_x: float = (dst_width - fit_width) / 2.0
        fit_y: float = (dst_height - fit_height) / 2.0

        ret[ImageType.TOT][0] = (0, 0, int(fit_width), int(fit_height))

        cam_width: float = fit_width if cam_columns == 0 else fit_width / cam_columns
        cam_height: float = cam_width / cam_aspect_ratio
        sim_height: float = fit_width / sim_aspect_ratio
        psn_width: float =  fit_width / num_persons
        psn_height: float = psn_width / psn_aspect_ratio
        vis_height: float = fit_width / vis_aspect_ratio
        y_start: float = fit_y

        for i in range(num_cams):
            cam_x: float = (i % cam_columns) * cam_width + fit_x
            cam_y: float = (i // cam_columns) * cam_height + fit_y
            ret[ImageType.CAM][i] = (int(cam_x), int(cam_y), int(cam_width), int(cam_height))

        y_start += cam_height * cam_rows
        for i in range(num_sims):
            sim_y: float = y_start + i * sim_height
            ret[ImageType.SIM][i] = (int(fit_x), int(sim_y), int(fit_width), int(sim_height))

        y_start += sim_height * num_sims
        for i in range(num_persons):
            psn_x: float = i * psn_width + fit_x
            psn_y: float = y_start
            ret[ImageType.PSN][i] = (int(psn_x), int(psn_y), int(psn_width), int(psn_height))

        y_start += psn_height
        for i in range(num_viss):
            sim_y: float = y_start + i * vis_height
            ret[ImageType.VIS][i] = (int(fit_x), int(sim_y), int(fit_width), int(vis_height))

        # Fill the last Vis till the bottom of the window
        if num_viss > 0:
            last_Vis = ret[ImageType.VIS][num_viss - 1]
            if last_Vis[1] + last_Vis[3] < dst_height:
                ret[ImageType.VIS][num_viss - 1] = (last_Vis[0], last_Vis[1], last_Vis[2], dst_height - last_Vis[1])

        return ret