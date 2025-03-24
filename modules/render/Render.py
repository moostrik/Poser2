from OpenGL.GL import * # type: ignore
import OpenGL.GLUT as glut
from threading import Lock
import numpy as np
from enum import Enum
import math

from modules.gl.RenderWindow import RenderWindow
from modules.gl.Texture import Texture
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Mesh import Mesh
from modules.gl.Utils import lfo, fit, fill
from modules.gl.Image import Image

from depthai import Rect, Tracklet
from modules.pose.PoseDefinitions import Pose, Indices
from modules.person.Person import Person


class ImageType(Enum):
    CAM = 0
    PERSON = 1

Composition_Subdivision = dict[ImageType, dict[int, tuple[int, int, int, int]]]

class Render(RenderWindow):
    def __init__(self, num_cams: int, num_persons: int, window_width: int, window_height: int, name: str, fullscreen: bool = False, v_sync = False) -> None:
        super().__init__(window_width, window_height, name, fullscreen, v_sync)

        self.num_cams: int = num_cams
        self.num_persons: int = num_persons

        self.cam_images: dict[int, Image] = {}
        self.psn_images: dict[int, Image] = {}

        self.cam_fbos: dict[int, Fbo] = {}
        self.psn_fbos: dict[int, Fbo] = {}
        self.pose_meshes: dict[int, Mesh] = {}

        self.all_images: list[Image] = []
        self.all_fbos: list[Fbo | SwapFbo] = []
        self.all_meshes: list[Mesh] = []

        for i in range(self.num_cams):
            self.cam_images[i] = Image()
            self.all_images.append(self.cam_images[i])
            self.cam_fbos[i] = Fbo()
            self.all_fbos.append(self.cam_fbos[i])

        for i in range(self.num_persons):
            self.psn_images[i] = Image()
            self.all_images.append(self.psn_images[i])

            self.psn_fbos[i] = Fbo()
            self.all_fbos.append(self.psn_fbos[i])

            self.pose_meshes[i] = Mesh()
            self.pose_meshes[i].set_indices(Indices)
            self.all_meshes.append(self.pose_meshes[i])

        self.composition: Composition_Subdivision = self.make_composition_subdivision(window_width, window_height, num_cams, num_persons)
        self.allocated = False

        self.input_mutex: Lock = Lock()
        self.input_tracklets: dict[int, dict[int, Tracklet]] = {}   # cam_id -> track_id -> Tracklet
        self.input_persons: dict[int, Person | None] = {}           # person_id -> Person
        for i in range(num_persons):
            self.input_tracklets[i] = {}
            self.input_persons[i] = None

    def reshape(self, width, height) -> None: # override
        super().reshape(width, height)
        self.composition = self.make_composition_subdivision(width, height, self.num_cams, self.num_persons)

        for key in self.cam_fbos.keys():
            x, y, w, h = self.composition[ImageType.CAM][key]
            self.cam_fbos[key].allocate(w, h, GL_RGBA)

        for key in self.psn_fbos.keys():
            x, y, w, h = self.composition[ImageType.PERSON][key]
            self.psn_fbos[key].allocate(w, h, GL_RGBA)

    def draw(self) -> None: # override
        if not self.allocated:
            self.reshape(self.window_width, self.window_height)
            self.allocated = True

        self.update_pose_meshes()

        self.draw_cameras()
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
                    roi: Rect | None = person.pose_rect
                    mesh: Mesh = self.pose_meshes[person.id]
                    if roi is not None and mesh.isInitialized():
                        x, y, w, h = roi.x, roi.y, roi.width, roi.height
                        x *= fbo.width
                        y *= fbo.height
                        w *= fbo.width
                        h *= fbo.height

                        self.draw_person(person, mesh, x, y, w, h, False, True, False)

            fbo.end()

    def draw_persons(self) -> None:
        for i in range(self.num_persons):
            image: Image = self.psn_images[i]
            image.update()
            fbo: Fbo = self.psn_fbos[i]
            person: Person | None = self.get_person(i)

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

        for i in range(self.num_persons):
            x, y, w, h = self.composition[ImageType.PERSON][i]
            self.psn_fbos[i].draw(x, y, w, h)

    # SETTERS AND GETTERS
    def set_cam_image(self, cam_id: int, image: np.ndarray) -> None :
        self.cam_images[cam_id].set_image(image)

    def set_psn_image(self, psn_id: int, image: np.ndarray) -> None :
        self.psn_images[psn_id].set_image(image)

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

        image: np.ndarray | None = person.pose_image
        if image is not None:
            self.set_psn_image(person.id, image)

        # poses: list[Pose] | None = person.pose
        # if poses is not None and len(poses) > 0:
        #     self.set_vertices(person.id, poses[0].getVertices())
        #     self.set_colors(person.id, poses[0].getColors())

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

        string: str = f'ID: {tracklet.id} Age: {tracklet.age} C: {tracklet.srcImgDetection.confidence:.2f}'
        if tracklet.spatialCoordinates.z > 0:
            string += f' Z: {tracklet.spatialCoordinates.z:.0f}'
        x += 9
        y += 15
        glRasterPos2f(x, y)     # Set position
        for character in string:
            glut.glutBitmapCharacter(glut.GLUT_BITMAP_9_BY_15, ord(character)) # type: ignore
        glRasterPos2f(0, 0)     # Reset position

        glFlush()               # Render now

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
            glRasterPos2f(x, y)     # Set position
            for character in string:
                glut.glutBitmapCharacter(glut.GLUT_BITMAP_9_BY_15, ord(character)) # type: ignore
            glRasterPos2f(0, 0)     # Reset position

        glFlush()               # Render now

    @staticmethod
    def make_composition_subdivision(dst_width: int, dst_height: int,
                                     num_cams: int, num_persons: int,
                                     cam_aspect_ratio: float = 16.0 / 9.0,
                                     psn_aspect_ratio: float = 1.0) -> Composition_Subdivision:

        ret: Composition_Subdivision = {}
        ret[ImageType.CAM] = {}
        ret[ImageType.PERSON] = {}

        cam_rows: int = math.ceil(num_cams / 2)
        cam_colums: int = math.ceil(num_cams / cam_rows)

        dst_aspect_ratio: float = dst_width / dst_height
        cam_grid_aspect_ratio: float = cam_aspect_ratio * cam_colums / cam_rows
        psn_grid_aspect_ratio: float = psn_aspect_ratio * num_persons
        tot_aspect_ratio: float = 1.0 / (1.0 / cam_grid_aspect_ratio + 1.0 / psn_grid_aspect_ratio)

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

        cam_width: float =  fit_width / cam_colums
        cam_height: float = cam_width / cam_aspect_ratio
        psn_width: float =  fit_width / num_persons
        psn_height: float = psn_width / psn_aspect_ratio

        for i in range(num_cams):
            cam_x: float = (i % cam_colums) * cam_width + fit_x
            cam_y: float = (i // cam_colums) * cam_height + fit_y
            ret[ImageType.CAM][i] = (int(cam_x), int(cam_y), int(cam_width), int(cam_height))

        for i in range(num_persons):
            psn_x: float = i * psn_width + fit_x
            psn_y: float = fit_height - psn_height + fit_y
            ret[ImageType.PERSON][i] = (int(psn_x), int(psn_y), int(psn_width), int(psn_height))

        return ret