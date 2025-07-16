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
from modules.cam.depthcam.Definitions import Tracklet as CamTracklet, Rect as CamRect, FrameType
from modules.tracker.Tracklet import Tracklet, TrackletIdColor, TrackingStatus, Rect
from modules.correlation.PairCorrelationStream import PairCorrelationStreamData
from modules.pose.PoseDefinitions import Pose, PosePoints, PoseEdgeIndices
from modules.pose.PoseStream import PoseStreamData
from modules.Settings import Settings

from modules.render.RenderCompositionSubdivision import make_subdivision, SubdivisionType
from modules.render.RenderDataManager import RenderDataManager

from modules.utils.HotReloadMethods import HotReloadMethods

# Shaders
from modules.gl.shaders.WS_Angles import WS_Angles
from modules.gl.shaders.WS_Lines import WS_Lines
from modules.gl.shaders.WS_PoseStream import WS_PoseStream

from modules.gl.shaders.RStream import RStream


class DataVisType(Enum):
    TRACKING = 0
    R_PAIRS = 1

class LightVisType(Enum):
    LINES = 0
    LIGHTS = 1

Composition_Subdivision = dict[SubdivisionType, dict[int, tuple[int, int, int, int]]]


class Render(RenderWindow):
    def __init__(self, settings: Settings) -> None:

        self.data: RenderDataManager = RenderDataManager()

        self.max_players: int = settings.max_players
        self.num_cams: int = settings.camera_num
        self.num_sims: int = len(DataVisType)
        self.num_viss: int = len(LightVisType)
        self.vis_width: int = settings.light_resolution
        self.r_streams: int = 3

        # images
        self.avi_image = Image()
        self.cam_images: dict[int, Image] = {}
        self.pse_images: dict[int, Image] = {}
        self.a_s_images: dict[int, Image] = {}
        self.r_s_images: dict[int, Image] = {}
        for i in range(self.num_cams):
            self.cam_images[i] = Image()
        for i in range(self.max_players):
            self.pse_images[i] = Image()
            self.a_s_images[i] = Image()
        for i in range(self.r_streams):
            self.r_s_images[i] = Image()

        # fbos
        self.cam_fbos: dict[int, Fbo] = {}
        self.sim_fbos: dict[int, Fbo] = {}
        self.vis_fbos: dict[int, Fbo] = {}
        self.pse_fbos: dict[int, Fbo] = {}
        for i in range(self.num_cams):
            self.cam_fbos[i] = Fbo()
        for i in range(self.num_sims):
            self.sim_fbos[i] = Fbo()
        for i in range(self.num_viss):
            self.vis_fbos[i] = Fbo()
        for i in range(self.max_players):
            self.pse_fbos[i] = Fbo()

        # meshes
        self.pose_meshes: dict[int, Mesh] = {}
        self.angle_meshes: dict[int, Mesh] = {}
        self.R_meshes: dict[Tuple[int, int], Mesh] = {}
        for i in range(self.max_players):
            self.pose_meshes[i] = Mesh()
            self.pose_meshes[i].set_indices(PoseEdgeIndices)
            self.angle_meshes[i] = Mesh()

        # shaders
        self.vis_line_shader: WS_Lines = WS_Lines()
        self.vis_angle_shader: WS_Angles = WS_Angles()
        self.pose_stream_shader: WS_PoseStream = WS_PoseStream()
        self.r_stream_shader = RStream()
        self.all_shaders: list[Shader] = [self.vis_line_shader, self.vis_angle_shader, self.pose_stream_shader, self.r_stream_shader]

        # composition
        self.composition: Composition_Subdivision = make_subdivision(settings.render_width, settings.render_height, self.num_cams, self.num_sims, self.max_players, self.num_viss)
        super().__init__(self.composition[SubdivisionType.TOT][0][2], self.composition[SubdivisionType.TOT][0][3], settings.render_title, settings.render_fullscreen, settings.render_v_sync, settings.render_fps, settings.render_x, settings.render_y)

        self.allocated = False

        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def reshape(self, width, height) -> None: # override
        super().reshape(width, height)
        self.composition = make_subdivision(width, height, self.num_cams, self.num_sims, self.max_players, self.num_viss)

        for key in self.cam_fbos.keys():
            x, y, w, h = self.composition[SubdivisionType.CAM][key]
            self.cam_fbos[key].allocate(w, h, GL_RGBA)

        for key in self.sim_fbos.keys():
            x, y, w, h = self.composition[SubdivisionType.SIM][key]
            self.sim_fbos[key].allocate(w, h, GL_RGBA)

        for key in self.pse_fbos.keys():
            x, y, w, h = self.composition[SubdivisionType.PSN][key]
            self.pse_fbos[key].allocate(w, h, GL_RGBA)

    def draw(self) -> None: # override
        if not self.allocated:
            self.reshape(self.window_width, self.window_height)
            for s in self.all_shaders:
                s.allocate(True) # type: ignore
            for fbo in self.vis_fbos.values():
                fbo.allocate(self.vis_width, 1, GL_RGBA32F)
            self.allocated = True

        try:
            self.update_pose_meshes()
            self.update_angle_meshes()
            # self.update_R_meshes()

            self.draw_cameras()
            self.draw_sims()
            self.draw_poses()
            self.draw_lights()

            self.draw_composition()
        except Exception as e:
            print(f"Error in draw: {e}")

    def update_pose_meshes(self) -> None:
        for i in range(self.max_players):
            pose: Pose | None = self.data.get_pose(i, True)
            if pose is not None:
                points: PosePoints | None = pose.points
                if points is not None:
                    self.pose_meshes[i].set_vertices(points.getVertices())
                    self.pose_meshes[i].set_colors(points.getColors(threshold=0.0))
                    self.pose_meshes[i].update()

    def update_angle_meshes(self) -> None:
        for i in range(self.max_players):
            pose_stream: PoseStreamData | None = self.data.get_pose_stream(i, False)
            if pose_stream is None:
                continue

            angles_np: np.ndarray = np.nan_to_num(pose_stream.angles.to_numpy(), nan=0.0)
            conf_np: np.ndarray = pose_stream.confidences.to_numpy()
            if angles_np.shape[0] != conf_np.shape[0] or angles_np.shape[1] != conf_np.shape[1]:
                print(f"Angles shape {angles_np.shape} does not match confidences shape {conf_np.shape}")
                continue
            mesh_data: np.ndarray = np.stack([angles_np, conf_np], axis=-1)


            # Only use the first 4 joints
            data: np.ndarray = mesh_data[:, :4, :]

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
            colors[:, 3] = conf_flat #np.where(conf_flat > 0.0, 1.0, 0.0)
            # colors[:, 3] = confidences.T.flatten()
            mesh.set_colors(colors)

            mesh.update()

    def update_R_meshes(self) -> None:
        r_streams: PairCorrelationStreamData | None = self.data.get_correlation_streams()

        if r_streams is None:
            return

        return
        self.R_meshes.clear()

        for pair_r in r_streams.values():
            pair_id: Tuple[int, int] = pair_r[0]
            data: Optional[np.ndarray] = pair_r[1]

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


    def draw_cameras(self) -> None:
        for i in range(self.num_cams):
            frame: np.ndarray | None = self.data.get_cam_image(i)
            image: Image = self.cam_images[i]
            if frame is not None:
                image.set_image(frame)
                image.update()
            fbo: Fbo = self.cam_fbos[i]
            depth_tracklets: list[CamTracklet] | None = self.data.get_depth_tracklets(i, False)
            poses: list[Pose] = self.data.get_poses_for_cam(i)

            Render.draw_camera(fbo, image, depth_tracklets, poses, self.pose_meshes)

    def draw_sims(self) -> None:
        for i in range(self.num_sims):
            fbo: Fbo = self.sim_fbos[i]
            self.setView(fbo.width, fbo.height)
            if i == DataVisType.TRACKING.value:
                self.draw_map_positions(self.data.get_tracklets(), self.num_cams, fbo)
            # elif i == DataVisType.R_PAIRS.value:
            #     R_windows = self.get_correlation_windows()
            #     if R_windows:
            #         self.draw_R_pairs_shader(R_windows, fbo)

    def draw_poses(self) -> None:
        for i in range(self.max_players):
            fbo: Fbo = self.pse_fbos[i]
            pose: Pose | None = self.data.get_pose(i, False)
            if pose is None:
                continue #??
            pose_image: Image = self.pse_images[i]
            pose_image_np: np.ndarray | None = pose.image
            if pose_image_np is not None:
                pose_image.set_image(pose_image_np)
                pose_image.update()
            pose_mesh: Mesh = self.pose_meshes[pose.id]
            pose_stream: PoseStreamData | None = self.data.get_pose_stream(i)
            a_s_image: Image = self.a_s_images[i]
            if pose_stream is not None:
                a_s_image_np: np.ndarray = self.pose_stream_to_image(pose_stream)
                a_s_image.set_image(a_s_image_np)
                a_s_image.update()

            angle_mesh: Mesh = self.angle_meshes[pose.id]
            # i =  pose.id

            Render.draw_pose(fbo, pose_image, pose, pose_mesh, a_s_image, angle_mesh, self.pose_stream_shader)

    def draw_lights(self) -> None:
        light_image: AvOutput | None = self.data.get_light_image()
        if light_image is None:
            return

        self.avi_image.set_image(light_image.img)
        self.avi_image.update()

        for i in range(self.num_viss):
            fbo: Fbo = self.vis_fbos[i]
            self.setView(fbo.width, fbo.height)
            fbo.begin()
            if i == LightVisType.LINES.value:
                self.vis_line_shader.use(fbo.fbo_id, self.avi_image.tex_id)
            elif i == LightVisType.LIGHTS.value:
                self.vis_angle_shader.use(fbo.fbo_id, self.avi_image.tex_id, light_image.resolution)
            glFlush()
            fbo.end()

    def draw_composition(self) -> None:
        self.setView(self.window_width, self.window_height)
        for i in range(self.num_cams):
            x, y, w, h = self.composition[SubdivisionType.CAM][i]
            self.cam_fbos[i].draw(x, y, w, h)

        for i in range(self.num_sims):
            x, y, w, h = self.composition[SubdivisionType.SIM][i]
            self.sim_fbos[i].draw(x, y, w, h)

        for i in range(self.max_players):
            x, y, w, h = self.composition[SubdivisionType.PSN][i]
            self.pse_fbos[i].draw(x, y, w, h)

        for i in range(self.num_viss):
            x, y, w, h = self.composition[SubdivisionType.VIS][i]
            self.vis_fbos[i].draw(x, y, w, h)

    # STATIC METHODS
    @staticmethod
    def draw_camera(fbo: Fbo, image: Image, depth_tracklets: list[CamTracklet], poses: list[Pose], pose_meshes: dict[int, Mesh]) -> None:
        RenderWindow.setView(fbo.width, fbo.height)
        fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        image.draw(0, 0, fbo.width, fbo.height)

        for depth_tracklet in depth_tracklets:
            Render.draw_depth_tracklet(depth_tracklet, 0, 0, fbo.width, fbo.height)

        for pose in poses:
            tracklet: Tracklet | None = pose.tracklet
            if tracklet is None or tracklet.is_removed or tracklet.is_lost:
                continue
            roi: Rect | None = pose.crop_rect
            mesh: Mesh = pose_meshes[pose.id]
            if roi is None or not mesh.isInitialized():
                continue
            x, y, w, h = roi.x, roi.y, roi.width, roi.height
            x *= fbo.width
            y *= fbo.height
            w *= fbo.width
            h *= fbo.height
            Render.draw_tracklet(tracklet, mesh, x, y, w, h, True, True, False)

        glFlush()  # Render now
        fbo.end()

    @staticmethod
    def draw_depth_tracklet(tracklet: CamTracklet, x: float, y: float, w: float, h: float) -> None:
        if tracklet.status == CamTracklet.TrackingStatus.REMOVED:
            return

        x = x + tracklet.roi.x * w
        y = y + tracklet.roi.y * h
        w = tracklet.roi.width * w
        h = tracklet.roi.height * h

        r: float = 1.0
        g: float = 1.0
        b: float = 1.0
        a: float = min(tracklet.age / 100.0, 0.33)
        if tracklet.status == CamTracklet.TrackingStatus.NEW:
            r, g, b, a = (1.0, 1.0, 1.0, 1.0)
        if tracklet.status == CamTracklet.TrackingStatus.TRACKED:
            r, g, b, a = (0.0, 1.0, 0.0, a)
        if tracklet.status == CamTracklet.TrackingStatus.LOST:
            r, g, b, a = (1.0, 0.0, 0.0, a)
        if tracklet.status == CamTracklet.TrackingStatus.REMOVED:
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

        # glFlush()               # Render now

    @staticmethod
    def draw_tracklet(tracklet: Tracklet, pose_mesh: Mesh, x: float, y: float, w: float, h: float, draw_box = False, draw_pose = False, draw_text = False) -> None:
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
            string: str = f'ID: {tracklet.id} Cam: {tracklet.cam_id} Age: {tracklet.age_in_seconds:.2f}'
            x += 9
            y += 15
            RenderWindow.draw_string(x, y, string)

    @staticmethod
    def draw_pose(fbo: Fbo, pose_image: Image, pose: Pose, pose_mesh: Mesh, angle_image: Image, angle_mesh: Mesh, shader: WS_PoseStream) -> None:
        RenderWindow.setView(fbo.width, fbo.height)

        if pose.is_final:
            fbo.begin()
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            fbo.end()
            return

        fbo.begin()
        pose_image.draw(0, 0, fbo.width, fbo.height)
        tracklet: Tracklet | None = pose.tracklet
        if tracklet is not None:
            draw_box: bool = tracklet.is_lost
            Render.draw_tracklet(tracklet, pose_mesh, 0, 0, fbo.width, fbo.height, draw_box, True, True)
        if angle_mesh.isInitialized():
            angle_mesh.draw(0, 0, fbo.width, fbo.height)
        glFlush()
        fbo.end()
        shader.use(fbo.fbo_id, angle_image.tex_id)

    @staticmethod
    def draw_map_positions(tracklets: dict[int, Tracklet], num_cams: int, fbo: Fbo) -> None:
        fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)  # Set background color to black
        glClear(GL_COLOR_BUFFER_BIT)       # Actually clear the buffer!

        for tracklet in tracklets.values():
            if tracklet is None:
                continue
            if tracklet.status != TrackingStatus.TRACKED and tracklet.status != TrackingStatus.NEW:
                continue

            world_angle: float = getattr(tracklet.tracker_info, "world_angle", 0.0)
            local_angle: float = getattr(tracklet.tracker_info, "local_angle", 0.0)
            overlap: bool = getattr(tracklet.tracker_info, "overlap", False)

            w: float = tracklet.roi.width * fbo.width / num_cams
            h: float = tracklet.roi.height * fbo.height
            x: float = world_angle / 360.0 * fbo.width

            y: float = tracklet.roi.y * fbo.height
            color: list[float] = TrackletIdColor(tracklet.id, aplha=0.9)
            if overlap == True:
                color[3] = 0.3
            if tracklet.status == TrackingStatus.NEW:
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
            string = f'A: {world_angle:.1f}'
            RenderWindow.draw_string(x, y, string)
            y += 14
            string = f'L: {local_angle:.1f}'
            RenderWindow.draw_string(x, y, string)

        fbo.end()
        glFlush()

    @staticmethod
    def pose_stream_to_image(pose_stream: PoseStreamData) -> np.ndarray:
        angles_raw: np.ndarray = np.nan_to_num(pose_stream.angles.to_numpy(), nan=0.0).astype(np.float32)
        angles_norm: np.ndarray = np.clip(np.abs(angles_raw) / np.pi, 0, 1)
        sign_channel: np.ndarray = (angles_raw > 0).astype(np.float32)
        confidences: np.ndarray = np.clip(pose_stream.confidences.to_numpy().astype(np.float32), 0, 1)
        # width, height = angles_norm.shape
        # img: np.ndarray = np.ones((height, width, 4), dtype=np.float32)
        # img[..., 2] = angles_norm.T   r
        # img[..., 1] = sign_channel.T  g
        # img[..., 0] = confidences.T   b
        channels: np.ndarray = np.stack([confidences, sign_channel, angles_norm], axis=-1)  # shape: (width, height, 3)
        return channels.transpose(1, 0, 2)

    @staticmethod
    def draw_R_pairs(R_meshes: dict[Tuple[int, int], Mesh], fbo: Fbo) -> None:
        fbo.begin()
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        num_pairs = len(R_meshes)
        if num_pairs == 0:
            fbo.end()
            return

        draw_width: int = fbo.width - 80
        slice_height = fbo.height // num_pairs

        # Pre-calculate color map
        color_map = [
            (1.0, 0.2, 0.2, 0.5),  # red
            (0.2, 1.0, 0.2, 1.0),  # green
            (0.2, 0.2, 1.0, 1.0),  # blue
            (1.0, 1.0, 0.2, 1.0),  # yellow
            (1.0, 0.2, 1.0, 1.0),  # magenta
            (0.2, 1.0, 1.0, 1.0),  # cyan
        ]

        # Batch background rectangles
        glBegin(GL_QUADS)
        for idx, (pair_id, mesh) in enumerate(R_meshes.items()):
            if not mesh.isInitialized():
                continue

            color_seed = pair_id[0] % 6
            glColor4f(*color_map[color_seed])

            y_offset = idx * slice_height
            glVertex2f(0, y_offset)
            glVertex2f(draw_width, y_offset)
            glVertex2f(draw_width, y_offset + slice_height)
            glVertex2f(0, y_offset + slice_height)
        glEnd()

        # Draw meshes and text separately to minimize state changes
        glColor4f(1.0, 1.0, 1.0, 1.0)
        text_data = []  # Store text data for batch rendering

        for idx, (pair_id, mesh) in enumerate(R_meshes.items()):
            if not mesh.isInitialized():
                continue

            y_offset = idx * slice_height
            mesh.draw(0, y_offset + slice_height, draw_width, -slice_height)

            # Store text data instead of rendering immediately
            vertices = mesh.vertices
            if vertices is not None and len(vertices) > 0:
                last_vertex = vertices[-1]
                x = int(last_vertex[0] * draw_width + 10)
                y = int(y_offset + (1.0 - last_vertex[1]) * slice_height + 7)
                text = f"{pair_id[0]}-{pair_id[1]}: {last_vertex[1]:.2f}"
                text_data.append((x, y, text))

        # Batch render all text
        for x, y, text in text_data:
            RenderWindow.draw_string(x, y, text)

        fbo.end()

    def draw_R_pairs_shader(self, R_windows: dict[Tuple[int, int], np.ndarray], fbo: Fbo) -> None:
        if not R_windows:
            return

        # Color map for different pairs
        color_map = [
            (1.0, 0.2, 0.2),  # red
            (0.2, 1.0, 0.2),  # green
            (0.2, 0.2, 1.0),  # blue
            (1.0, 1.0, 0.2),  # yellow
            (1.0, 0.2, 1.0),  # magenta
            (0.2, 1.0, 1.0),  # cyan
        ]

        fbo.begin()
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # glEnable(GL_BLEND)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        num_pairs = len(R_windows)
        for idx, (pair_id, data) in enumerate(R_windows.items()):
            if data is None or len(data) < 2:
                continue

            color = color_map[pair_id[0] % len(color_map)]

            self.r_stream_shader.use(
                fbo_id=fbo.fbo_id,
                correlation_data=data,
                pair_index=idx,
                total_pairs=num_pairs,
                line_color=color,
                line_width=10.0,
                viewport_width=float(fbo.width),
                viewport_height=float(fbo.height)
            )

        # glDisable(GL_BLEND)
        # glEnable(GL_BLEND)
        fbo.end()
