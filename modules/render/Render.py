# Standard library imports
import math
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Type, Any, Dict, Tuple
from threading import Lock

# Third-party imports
from OpenGL.GL import * # type: ignore
import OpenGL.GLUT as glut

# Local application imports
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Image import Image
from modules.gl.Mesh import Mesh
from modules.gl.RenderWindow import RenderWindow
from modules.gl.Shader import Shader
from modules.gl.Texture import Texture
from modules.gl.Utils import lfo, fit, fill

from modules.av.Definitions import AvOutput
from modules.cam.depthcam.Definitions import Tracklet, Rect, Point3f, FrameType
from modules.person.Person import Person, PersonColor, TrackingStatus
from modules.pose.PoseCorrelation import PoseCorrelationWindow, PoseCorrelationBatch
from modules.pose.PoseDefinitions import Pose, PoseEdgeIndices
from modules.pose.PoseWindowBuffer import PoseWindowData
from modules.Settings import Settings

from modules.utils.HotReloadMethods import HotReloadMethods

# Shaders
from modules.gl.shaders.WS_Angles import WS_Angles
from modules.gl.shaders.WS_Lines import WS_Lines

class ImageType(Enum):
    TOT = 0
    CAM = 1
    SIM = 2
    PSN = 3
    VIS = 4

class DataVisType(Enum):
    TRACKING = 0
    R_PAIRS = 1

class LightVisType(Enum):
    LINES = 0
    LIGHTS = 1

Composition_Subdivision = dict[ImageType, dict[int, tuple[int, int, int, int]]]


class Render(RenderWindow):
    def __init__(self, settings: Settings) -> None:

        self.num_cams: int = settings.camera_num
        self.num_sims: int = len(DataVisType)
        self.num_viss: int = len(LightVisType)
        self.num_persons: int = settings.pose_num

        self.cam_images: dict[int, Image] = {}
        self.psn_images: dict[int, Image] = {}

        self.cam_fbos: dict[int, Fbo] = {}
        self.sim_fbos: dict[int, Fbo] = {}
        self.vis_fbos: dict[int, Fbo] = {}
        self.psn_fbos: dict[int, Fbo] = {}

        self.R_meshes: dict[Tuple[int, int], Mesh] = {}  # pair_id -> Mesh
        self.pose_meshes: dict[int, Mesh] = {}
        self.angle_meshes: dict[int, Mesh] = {}

        self.all_images: list[Image] = []
        self.all_fbos: list[Fbo | SwapFbo] = []
        # self.all_meshes: list[Mesh] = []
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
            self.pose_meshes[i].set_indices(PoseEdgeIndices)

            self.angle_meshes[i] = Mesh()
            # self.all_meshes.append(self.angle_meshes[i])

        self.composition: Composition_Subdivision = self.make_composition_subdivision(settings.render_width, settings.render_height, self.num_cams, self.num_sims, self.num_persons, self.num_viss)
        super().__init__(self.composition[ImageType.TOT][0][2], self.composition[ImageType.TOT][0][3], settings.render_title, settings.render_fullscreen, settings.render_v_sync, settings.render_fps, settings.render_x, settings.render_y)

        self.allocated = False

        self.input_mutex: Lock = Lock()
        self.input_tracklets: dict[int, dict[int, Tracklet]] = {}   # cam_id -> track_id -> Tracklet
        self.input_persons: dict[int, Optional[Person]] = {}           # person_id -> Person
        self.input_angle_windows: dict[int, Optional[np.ndarray]] = {}          # person_id -> angles
        for i in range(self.num_persons):
            self.input_tracklets[i] = {}
            self.input_persons[i] = None
            self.input_angle_windows[i] = None

        self.vis_width: int = settings.light_resolution
        self.vis_height: int = 1
        self.vis_image = Image()
        self.all_images.append(self.vis_image)
        self.vis_line_shader: WS_Lines = WS_Lines()
        self.vis_angle_shader: WS_Angles = WS_Angles()
        self.all_shaders.append(self.vis_line_shader)
        self.all_shaders.append(self.vis_angle_shader)

        self.analysis_images: dict[int, Image] = {}  # person_id -> Image
        for i in range(self.num_persons):
            self.analysis_images[i] = Image()
            self.all_images.append(self.analysis_images[i])
        # self.analysis_shader: WS_Angles = WS_Angles()
        # self.all_shaders.append(self.analysis_shader)

        self.input_R_windows:dict[Tuple[int, int], np.ndarray] = {}
        self.input_R_window_consumed: bool = True

        self.hot_reloader = HotReloadMethods(self.__class__, True, False)

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

        try:
            self.update_R_meshes()

            self.draw_cameras()
            self.draw_sims()
            self.draw_lights()

            self.update_pose_meshes()
            self.update_angle_meshes()
            self.draw_persons()

            self.draw_composition()
        except Exception as e:
            print(f"Error in draw_persons: {e}")

    def update_R_meshes(self) -> None:
        R_windows: Optional[dict[Tuple[int, int], np.ndarray]] = self.get_correlation_windows()

        if R_windows is None:
            return

        self.R_meshes.clear()

        for pair_id, data in R_windows.items():
            if data is None or len(data) < 2:
                continue  # Need at least 2 points to draw a line

            mesh = Mesh()
            num_points = len(data)

            # X: normalized time (0 to 1), Y: similarity value (assume in [0, 1])
            x = np.linspace(0, 1, num_points, dtype=np.float32)
            y = np.clip(data.astype(np.float32), 0.0, 1.0)
            vertices = np.stack([x, y, np.zeros_like(x)], axis=1)  # shape (num_points, 3)

            # Indices for line strip
            indices = np.arange(num_points - 1, dtype=np.uint32)
            indices = np.stack([indices, indices + 1], axis=1).flatten()

            # Colors (e.g., white, or color by pair_id)
            colors = np.ones((num_points, 4), dtype=np.float32)

            mesh.set_vertices(vertices)
            mesh.set_indices(indices)
            mesh.set_colors(colors)
            mesh.update()

            self.R_meshes[pair_id] = mesh

    def update_pose_meshes(self) -> None:
        for i in range(self.num_persons):
            person: Person | None = self.get_person(i, clear=False)
            if person is not None:
                pose: Pose | None = person.pose
                if pose is not None:
                    self.pose_meshes[i].set_vertices(pose.getVertices())
                    self.pose_meshes[i].set_colors(pose.getColors(threshold=0.0))
                    self.pose_meshes[i].update()

    def update_angle_meshes(self) -> None:
        for i in range(self.num_persons):
            data: Optional[np.ndarray] = self.get_pose_window(i, clear=True)
            if data is None:
                continue

            # Only use the first 4 joints
            data = data[:, :4, :]

            mesh: Mesh = self.angle_meshes[i]
            num_frames, num_joints, _ = data.shape
            if num_frames < 2 or num_joints < 1:
                continue

            # Prepare confidences and angles
            confidences: np.ndarray = np.clip(data[..., 1], 0, 1)
            angles_raw: np.ndarray = data[..., 0]
            angles_norm: np.ndarray = np.clip(np.abs(angles_raw) / np.pi, 0, 1)

            joint_height: float = 1.0 / (num_joints + 2)

            # INDICES
            base = np.arange(num_joints) * num_frames
            frame_idx: np.ndarray = np.arange(num_frames - 1)
            start = base[:, None] + frame_idx
            end = start + 1
            indices: np.ndarray = np.stack([start, end], axis=-1).reshape(-1, 2).astype(np.uint32).flatten()
            mesh.set_indices(indices)

            # VERTICES
            frame_grid, joint_grid = np.meshgrid(np.arange(num_frames), np.arange(num_joints), indexing='ij')
            x = frame_grid / (num_frames - 1)
            y = (joint_grid + 1.0) * joint_height + angles_norm * joint_height
            vertices: np.ndarray = np.zeros((num_frames * num_joints, 3), dtype=np.float32)
            vertices[:, 0] = x.T.flatten()
            vertices[:, 1] = y.T.flatten()
            mesh.set_vertices(vertices)

            # COLORS
            even_mask: np.ndarray = (np.arange(num_joints) % 2 == 0)
            even_mask: np.ndarray = np.repeat(even_mask, num_frames)
            odd_mask = ~even_mask

            angle_mask: np.ndarray = (angles_raw > 0).T.flatten()

            colors: np.ndarray = np.ones((num_joints * num_frames, 4), dtype=np.float32)
            # Even joints
            colors[even_mask & angle_mask, :3] = [1.0, 1.0, 0.0]  # Yellow
            colors[even_mask & ~angle_mask, :3] = [1.0, 0.0, 0.0] # Red
            # Odd joints
            colors[odd_mask & angle_mask, :3] = [0.0, 0.7, 1.0]   # Blue
            colors[odd_mask & ~angle_mask, :3] = [0.0, 1.0, 0.0]  # Green

            # Alpha from confidences

            conf_flat: np.ndarray = confidences.T.flatten()
            colors[:, 3] = np.where(conf_flat > 0.0, 1.0, 0.0)
            # colors[:, 3] = confidences.T.flatten()
            mesh.set_colors(colors)

            mesh.update()

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
                if person.status == TrackingStatus.REMOVED or person.status == TrackingStatus.LOST:
                    continue
                roi: Rect | None = person.pose_roi
                mesh: Mesh = self.pose_meshes[person.id]
                if roi is not None and mesh.isInitialized():
                    x, y, w, h = roi.x, roi.y, roi.width, roi.height
                    x *= fbo.width
                    y *= fbo.height
                    w *= fbo.width
                    h *= fbo.height
                    self.draw_person(person, mesh, x, y, w, h, True, True, False)

            fbo.end()

    def draw_sims(self) -> None:
        for i in range(self.num_sims):
            fbo: Fbo = self.sim_fbos[i]
            self.setView(fbo.width, fbo.height)
            if i == DataVisType.TRACKING.value:
                self.draw_map_positions(self.input_persons, self.num_cams, fbo)
            elif i == DataVisType.R_PAIRS.value:
                self.draw_R_pairs(self.R_meshes, fbo)

    def draw_lights(self) -> None:
        self.vis_image.update()

        for i in range(self.num_viss):
            fbo: Fbo = self.vis_fbos[i]
            self.setView(fbo.width, fbo.height)
            fbo.begin()
            if i == LightVisType.LINES.value:
                self.draw_light_lines(fbo, self.vis_image, self.vis_line_shader)
            elif i == LightVisType.LIGHTS.value:
                self.draw_light_angles(fbo, self.vis_image, self.vis_angle_shader)
            fbo.end()

    def draw_persons(self) -> None:
        for i in range(self.num_persons):
            fbo: Fbo = self.psn_fbos[i]
            person: Person | None = self.get_person(i)
            if person is None:
                continue

            image: Image = self.psn_images[i]
            if person.img is not None:
                image.set_image(person.img)
                image.update()

            analysis_image: Image = self.analysis_images[i]
            analysis_image.update()

            self.setView(fbo.width, fbo.height)
            fbo.begin()

            if person.status == TrackingStatus.REMOVED or person.status == TrackingStatus.LOST:
                glColor4f(1.0, 1.0, 1.0, 0.25)   # Set color
                image.draw(0, 0, fbo.width, fbo.height)
                glColor4f(1.0, 1.0, 1.0, 1.0)   # Set color
            else:
                image.draw(0, 0, fbo.width, fbo.height)
            # analysis_image.draw(0, 0, fbo.width, fbo.height)
            mesh: Mesh = self.pose_meshes[person.id]
            self.draw_person(person, mesh, 0, 0, fbo.width, fbo.height, False, True, True)
            angle_mesh = self.angle_meshes[person.id]
            if angle_mesh.isInitialized():
                angle_mesh.draw(0, 0, fbo.width, fbo.height)

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
    def set_tracklet(self, cam_id: int, tracklet: Tracklet) -> None :
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
    def set_person(self, person: Person) -> None:
        with self.input_mutex:
            self.input_persons[person.id] = person

    def get_pose_window(self, id: int, clear = False) -> Optional[np.ndarray]:
        with self.input_mutex:
            ret_window: Optional[np.ndarray] = self.input_angle_windows[id]
            if clear:
                self.input_angle_windows[id] = None
            return ret_window
    def set_pose_window(self, data: PoseWindowData) -> None:
        angles_np: np.ndarray = np.nan_to_num(data.angles.to_numpy(), nan=0.0)
        conf_np: np.ndarray = data.confidences.to_numpy()
        if angles_np.shape[0] != conf_np.shape[0] or angles_np.shape[1] != conf_np.shape[1]:
            print(f"Angles shape {angles_np.shape} does not match confidences shape {conf_np.shape}")
            return
        mesh_data: np.ndarray = np.stack([angles_np, conf_np], axis=-1)

        with self.input_mutex:
            self.input_angle_windows[data.window_id] = mesh_data

    def get_correlation_windows(self) -> Optional[dict[Tuple[int, int], np.ndarray]]:
        with self.input_mutex:
            if self.input_R_window_consumed:
                return None
            self.input_R_window_consumed = True
            return self.input_R_windows.copy()
    def set_correlation_window(self, window: PoseCorrelationWindow) -> None:
        top_pairs: list[Tuple[int, int]] = window.get_top_pairs(n=3, time_window=0.5)
        with self.input_mutex:
            self.input_R_window_consumed = False
            self.input_R_windows.clear()
            for pair in top_pairs:
                data: Optional[np.ndarray] = window.get_metric_window(pair, metric_name='similarity')
                if data is not None:
                    self.input_R_windows[pair] = data

    def set_av(self, value: AvOutput) -> None:
        self.vis_image.set_image(value.img)

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
            if person is None or person.status == TrackingStatus.REMOVED:
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
            if person is None:
                continue
            if person.status != TrackingStatus.TRACKED and person.status != TrackingStatus.NEW:
                continue

            w: float = person.tracklet.roi.width * fbo.width / num_cams
            h: float = person.tracklet.roi.height * fbo.height
            x: float = person.world_angle / 360.0 * fbo.width
            y: float = person.tracklet.roi.y * fbo.height
            color: list[float] = PersonColor(person.id, aplha=0.9)
            if person.overlap == True:
                color[3] = 0.3
            if person.status == TrackingStatus.NEW:
                color = [1.0, 1.0, 1.0, 1.0]

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
    def draw_R_pairs(R_meshes: dict[Tuple[int, int], Mesh], fbo: Fbo) -> None:
        fbo.begin()
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        num_pairs = len(R_meshes)
        if num_pairs == 0:
            fbo.end()
            glFlush()
            return

        draw_width: int = fbo.width - 80
        slice_height = fbo.height // num_pairs

        for idx, (pair_id, mesh) in enumerate(R_meshes.items()):
            if not mesh.isInitialized():
                continue

            # Color based on first half of pair_id
            color_seed = pair_id[0] % 6
            color_map = [
                (1.0, 0.2, 0.2, 0.5),  # red
                (0.2, 1.0, 0.2, 1.0),  # green
                (0.2, 0.2, 1.0, 1.0),  # blue
                (1.0, 1.0, 0.2, 1.0),  # yellow
                (1.0, 0.2, 1.0, 1.0),  # magenta
                (0.2, 1.0, 1.0, 1.0),  # cyan
            ]
            glColor4f(*color_map[color_seed])
                # Draw background rect for this slice
            y_offset = idx * slice_height
            glBegin(GL_QUADS)
            glVertex2f(0, y_offset)
            glVertex2f(draw_width, y_offset)
            glVertex2f(draw_width, y_offset + slice_height)
            glVertex2f(0, y_offset + slice_height)
            glEnd()

            glColor4f(1.0, 1.0, 1.0, 1.0)  # Reset color
            # Compute vertical offset for this slice
            y_offset = idx * slice_height
            mesh.draw(0, y_offset + slice_height, draw_width, -slice_height)

            # Draw pair label at the end of the line
            vertices: Optional[np.ndarray] = mesh.vertices
            if vertices is not None and len(vertices) > 0:
                last_vertex = vertices[-1]
                x = int(last_vertex[0] * draw_width + 10)
                y = int(y_offset + (1.0 - last_vertex[1]) * slice_height + 7)
                text = f"{pair_id[0]}-{pair_id[1]}: {last_vertex[1]:.2f}"
                RenderWindow.draw_string(x, y, text)

        glColor4f(1.0, 1.0, 1.0, 1.0)  # Reset color
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
    def draw_person(person: Person, pose_mesh: Mesh, x: float, y: float, w: float, h: float, draw_box = False, draw_pose = False, draw_text = False) -> None:
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

        if draw_pose and pose_mesh.isInitialized():
            pose_mesh.draw(x, y, w, h)

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