from modules.Settings import Settings
from modules.cam.DepthCam import DepthCam, DepthSimulator
from modules.cam.recorder.SyncRecorderGui import SyncRecorderGui as Recorder
from modules.cam.depthplayer.SyncPlayerGui import SyncPlayerGui as Player
from modules.render.Render import Render
from modules.gui.PyReallySimpleGui import Gui
from modules.person.Manager import Manager as Detector


class Main():
    def __init__(self, settings: Settings) -> None:
        self.gui = Gui('DepthPose', settings.file_path, 'default')
        self.render = Render(settings.num_cams, settings.num_players, 1280, 720 + 256, 'Depth Pose', fullscreen=False, v_sync=True)
        # self.render = Render(4, settings.num_players, 1280, 720 + 256, 'Depth Pose', fullscreen=False, v_sync=True)

        self.recorder: Recorder | None = None
        self.player: Player | None = None

        self.cameras: list[DepthCam | DepthSimulator] = []
        if settings.simulation or settings.passthrough:
            self.player = Player(self.gui, settings)
            for cam_id in settings.camera_list:
                self.cameras.append(DepthSimulator(self.gui, self.player, cam_id, settings))
        else:
            self.recorder = Recorder(self.gui, settings)
            for cam_id in settings.camera_list:
                camera = DepthCam(self.gui, cam_id, settings)
                self.cameras.append(camera)

        self.detector = Detector(settings)
        self.running: bool = False

    def start(self) -> None:
        self.render.exit_callback = self.stop
        self.render.addKeyboardCallback(self.render_keyboard_callback)
        self.render.start()

        for camera in self.cameras:
            camera.add_preview_callback(self.render.set_cam_image)
            camera.add_frame_callback(self.detector.set_image)
            if self.recorder:
                camera.add_sync_callback(self.recorder.add_synced_frames)
            camera.add_tracker_callback(self.detector.add_tracklet)
            camera.add_tracker_callback(self.render.add_tracklet)
            camera.start()

        self.detector.addCallback(self.render.add_person)
        self.detector.start()

        self.gui.exit_callback = self.stop

        for camera in self.cameras:
            self.gui.addFrame([camera.gui.get_gui_color_frame(), camera.gui.get_gui_depth_frame()])
        if self.player:
            self.gui.addFrame([self.player.get_gui_frame()])
        if self.recorder:
            self.gui.addFrame([self.recorder.get_gui_frame()])
        self.gui.start()
        self.gui.bringToFront()

        for camera in self.cameras:
            camera.gui.gui_check()

        if self.player:
            self.player.gui_check()
            self.player.start()
        if self.recorder:
            self.recorder.gui_check()
            self.recorder.start() # start after gui to prevent record at startup

        self.running = True

    def stop(self) -> None:

        if self.player:
            # print('stop and join player')
            self.player.stop()
            self.player.join()

        # print('stop cameras')
        for camera in self.cameras:
            camera.stop()

        # print('stop detector')
        self.detector.stop()
        if self.recorder:
            # print('stop recorder')
            self.recorder.stop()
            self.recorder.join()

        # print ('join detector')
        self.detector.join()
        # print ('join cameras')
        for camera in self.cameras:
            camera.join()

        # print ('stop gui')
        self.gui.stop()
        # self.gui.join() # does not work as stop can be called from gui's own thread

        # print ('stop render')
        self.render.exit_callback = None
        self.render.stop()
        # self.render.join() # does not work as stop can be called from render's own thread

        self.running = False

    def isRunning(self) -> bool :
        return self.running

    def render_keyboard_callback(self, key, x, y) -> None:
        if not  self.isRunning(): return
        if key == b'g' or key == b'G':
            if not self.gui or not self.gui.isRunning(): return
            self.gui.bringToFront()
