"""Deep Flow — single-camera 3D volumetric fluid installation."""

from typing import Optional
from functools import partial

from modules.utils.Broadcast import Broadcast
from modules.oak import Camera, Simulator, Player, Recorder, Sync
from modules.settings import presets, NiceServer
from modules.inout import OscSound
from modules.tracker import OnePerCamTracker
from modules.pose import batch, nodes, trackers, window
from modules.pose.features import configure_features

from .render_board import RenderBoard
from .settings import Settings, Stage
from .render import DeepFlowRender

APP_NAME = 'deep_flow'


class DeepFlowMain:
    def __init__(self, simulation: bool = False) -> None:

        self.is_running: bool = False
        self.is_finished: bool = False

        # SETTINGS
        presets.set_app(APP_NAME)
        self.settings = Settings()
        preset_file = presets.startup_path()
        if not presets.load(self.settings, preset_file):
            raise FileNotFoundError(f"No preset found for '{APP_NAME}' at {preset_file}")
        self.settings.camera.sim_enabled = simulation
        self.settings.initialize()
        self.settings_server = NiceServer(self.settings, self.settings.server, on_exit=self.stop)

        num_players: int = self.settings.num_players
        p = self.settings.pose

        # BLACKBOARD
        self.board = RenderBoard()
        self.sound_osc = OscSound(self.settings.inout.osc_sound)

        # CAMERA
        configure_features(num_players)

        self.cameras: list[Camera | Simulator] = []
        self.recorder: Optional[Recorder] = None
        self.player: Optional[Player] = None
        if self.settings.camera.sim_enabled:
            self.player = Player(self.settings.camera.simulator)
            for i in range(num_players):
                self.cameras.append(Simulator(self.player, self.settings.camera.cameras[i], self.settings.camera.simulator))
        else:
            self.recorder = Recorder(self.settings.camera.recorder)
            for i in range(num_players):
                self.cameras.append(Camera(self.settings.camera.cameras[i]))
        self.frame_sync_bang = Sync(self.settings.camera.frame_sync, False, 'frame_sync')
        self.tracker = OnePerCamTracker(self.settings.tt.tracker, num_players)
        self.tracklet_sync_bang = Sync(self.settings.camera.tracklet_sync, False, 'tracklet_sync')
        self.image_crop_processor = batch.ImageCropProcessor(p.image_crop)

        for camera in self.cameras:
            if self.recorder:
                camera.add_sync_callback(self.recorder.set_synced_frames)
            camera.add_frame_callback(self.image_crop_processor.set_image)
            camera.add_frame_callback(self.frame_sync_bang.submit_frame)
            camera.add_tracker_callback(self.tracker.add_cam_tracklets)
            camera.add_tracker_callback(self.board.set_depth_tracklets)
            camera.add_tracker_callback(self.tracklet_sync_bang.submit_frame)

        # DETECTION
        self.poses_from_tracklets = batch.PosesFromTracklets(num_players)
        self.point_extractor = batch.PointBatchExtractor(p.detection)
        self.mask_extractor  = batch.MaskBatchExtractor(p.segmentation)
        self.flow_extractor  = batch.FlowBatchExtractor(p.flow)

        self.bbox_filters = trackers.FilterTracker({
            i: trackers.FilterPipeline([
                nodes.BBoxEuroSmoother(p.bbox.smoother),
                nodes.BBoxPredictor(p.bbox.prediction),
            ])
            for i in range(num_players)
        })

        self.tracker.add_tracklet_callback(self.poses_from_tracklets.set_tracklets)
        self.tracklet_sync_bang.add_callback(self.tracker.notify_update)
        self.frame_sync_bang.add_callback(self.poses_from_tracklets.generate)

        self.poses_from_tracklets.add_frames_callback(self.bbox_filters.process)
        self.bbox_filters.add_frames_callback(self.image_crop_processor.process)
        self.image_crop_processor.add_image_callback(self.point_extractor.process)
        self.image_crop_processor.add_image_callback(self.mask_extractor.process)
        self.image_crop_processor.add_image_callback(self.flow_extractor.process)
        self.mask_extractor.add_image_callback(lambda _f, gpu: self.board.set_images(gpu))

        # STAGE WINDOW TRACKERS & BROADCASTS
        self.window_trackers: dict[Stage, window.WindowTracker] = {}
        self.stages: dict[Stage, Broadcast] = {}
        for stage in Stage:
            wt = window.WindowTracker(num_players, getattr(p, f'window_{stage.name.lower()}'))
            wt.add_windows_callback(partial(self.board.set_windows, stage))
            self.window_trackers[stage] = wt
            self.stages[stage] = Broadcast([
                partial(self.board.set_frames, stage),
                partial(self.sound_osc.set_frames, stage),
                wt.process,
            ])

        # STAGE RAW
        self.point_extractor.add_frames_callback(self.stages[Stage.RAW])

        # STAGE CLEAN
        self.filters_clean = trackers.FilterTracker({
            i: trackers.FilterPipeline([
                nodes.PointDualConfFilter(p.point.confidence_filter),
                nodes.AngleExtractor(p.angle_extractor),
                nodes.AngleVelExtractor(p.velocity.extractor),
            ])
            for i in range(num_players)
        })
        self.stages[Stage.RAW].add_callback(self.filters_clean.process)
        self.filters_clean.add_frames_callback(self.stages[Stage.CLEAN])

        # STAGE SMOOTH
        self.filters_smooth = trackers.FilterTracker({
            i: trackers.FilterPipeline([
                nodes.PointEuroSmoother(p.point.smoother),
                nodes.AngleExtractor(p.angle_extractor),
                nodes.AngleVelExtractor(p.velocity.extractor),
                nodes.AngleVelEuroSmoother(p.velocity.smoother),
                nodes.AngleEuroSmoother(p.angle.smoother),
                nodes.AngleMotionExtractor(p.motion.extractor),
                nodes.AngleMotionMovingAverageSmoother(p.motion.moving_average),
                nodes.AngleSymExtractor(),
                nodes.MotionTimeExtractor(),
                nodes.AgeExtractor(),
            ])
            for i in range(num_players)
        })
        self.stages[Stage.CLEAN].add_callback(self.filters_smooth.process)
        self.filters_smooth.add_frames_callback(self.stages[Stage.SMOOTH])

        # STAGE PREDICT
        self.filters_predict = trackers.FilterTracker({
            i: trackers.FilterPipeline([
                nodes.PointPredictor(p.point.prediction),
                nodes.AnglePredictor(p.angle.prediction),
                nodes.AngleVelPredictor(p.velocity.prediction),
                nodes.AngleStickyFiller(p.angle.sticky),
            ])
            for i in range(num_players)
        })
        self.stages[Stage.SMOOTH].add_callback(self.filters_predict.process)
        self.filters_predict.add_frames_callback(self.stages[Stage.PREDICT])

        # STAGE LERP
        self.motion_gate_applicator = nodes.MotionGateApplicator(p.motion_gate)

        self.interpolators_lerp = trackers.InterpolatorTracker({
            i: trackers.InterpolatorPipeline([
                nodes.BBoxChaseInterpolator(p.bbox.interpolator),
                nodes.PointChaseInterpolator(p.point.interpolator),
                nodes.AngleChaseInterpolator(p.angle.interpolator),
                nodes.AngleVelChaseInterpolator(p.velocity.interpolator),
            ])
            for i in range(num_players)
        })
        self.filters_lerp = trackers.FilterTracker({
            i: trackers.FilterPipeline([
                nodes.AngleSymExtractor(),
                nodes.MotionTimeExtractor(),
                nodes.AgeExtractor(),
                nodes.AngleVelStickyFiller(p.velocity.sticky),
                nodes.AngleVelEuroSmoother(p.velocity.smoother),
                nodes.AngleMotionExtractor(p.motion.extractor),
                nodes.AngleMotionMovingAverageSmoother(p.motion.moving_average),
            ])
            for i in range(num_players)
        })
        self.gate_lerp = trackers.FilterTracker({
            i: trackers.FilterPipeline([self.motion_gate_applicator])
            for i in range(num_players)
        })
        self.stages[Stage.PREDICT].add_callback(self.interpolators_lerp.set)
        self.interpolators_lerp.add_frames_callback(self.filters_lerp.process)
        self.filters_lerp.add_frames_callback(self.motion_gate_applicator.set)
        self.filters_lerp.add_frames_callback(self.gate_lerp.process)
        self.gate_lerp.add_frames_callback(self.stages[Stage.LERP])

        # RENDER
        self.render = DeepFlowRender(self.board, self.settings.render, num_cams=len(self.cameras), num_players=num_players)
        self.render.add_exit_callback(self.stop)

        # IN/OUT
        self.render.add_update_callback(self.interpolators_lerp.update)

    def start(self) -> None:
        self.settings_server.start()

        for camera in self.cameras:
            camera.start()

        self.tracker.start()
        self.point_extractor.start()
        self.mask_extractor.start()
        self.flow_extractor.start()

        self.sound_osc.start()

        if self.player:
            self.player.start()
        if self.recorder:
            self.recorder.start()

        self.is_running = True
        self.render.start()

    def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False

        self.render.stop()

        if self.player:
            self.player.stop()
        for camera in self.cameras:
            camera.stop()
        if self.recorder:
            self.recorder.stop()

        self.tracker.stop()
        self.sound_osc.stop()

        self.point_extractor.stop()
        self.mask_extractor.stop()
        self.flow_extractor.stop()

        self.settings_server.stop()

        for camera in self.cameras:
            camera.join(timeout=10)

        self.is_finished = True
