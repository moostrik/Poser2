# Standard library imports
import logging
from dataclasses import dataclass, replace
from itertools import combinations
from queue import Empty, Queue
from threading import Lock, Thread, Event
from time import sleep, time

# Local application imports
from modules.oak import DepthTracklet
from modules.settings import BaseSettings, Field
from .. import (
    BaseTracker, TrackerType, TrackerAnnotation,
    Tracklet, TrackingStatus, TrackletDict, TrackletDictCallback,
)
from .panoramic_tracklet_manager import PanoramicTrackletManager
from .panoramic_geometry import PanoramicGeometry, DistortAlgorithm

logger = logging.getLogger(__name__)


class PanoramicTrackerSettings(BaseSettings):
    """Configuration for PanoramicTracker."""
    fov: Field[float] = Field(110.0, min=90.0, max=130.0, step=0.5)
    tracklet_min_age: Field[int] = Field(5, min=0, max=9, step=1)
    tracklet_min_height: Field[float] = Field(0.25, min=0.0, max=1.0, step=0.05)
    timeout: Field[float] = Field(2.0, min=1.0, max=5.0, step=0.1)
    cam_360_edge_threshold: Field[float] = Field(0.5, min=0.0, max=0.6, step=0.1)
    cam_360_overlap_expansion: Field[float] = Field(0.3, min=0.0, max=1.0, step=0.1)
    cam_360_hysteresis_factor: Field[float] = Field(0.9)
    distortion: Field[DistortAlgorithm] = Field(
        DistortAlgorithm.NONE, description="Lens distortion correction algorithm"
    )
    distortion_k1: Field[float] = Field(
        0.0, min=-2.0, max=2.0, step=0.01, description="Distortion coefficient k1"
    )
    distortion_k2: Field[float] = Field(
        0.0, min=-2.0, max=2.0, step=0.01, description="Distortion coefficient k2"
    )


@dataclass(frozen=True)
class PanoramicAnnotation(TrackerAnnotation):
    local_angle: float
    world_angle: float
    overlap: bool


@dataclass(frozen=True)
class PanoramicOverlapInfo:
    keep_id: int
    remove_id: int
    distance: float
    reason: str


class PanoramicTracker(Thread, BaseTracker):
    def __init__(self, config: PanoramicTrackerSettings, num_players: int, num_cameras: int) -> None:
        super().__init__()

        self._running: bool = False
        self._update_event: Event = Event()

        self._max_players: int = num_players

        self._input_queue: Queue[Tracklet] = Queue()

        self.tracklet_manager: PanoramicTrackletManager = PanoramicTrackletManager(self._max_players)

        self.config: PanoramicTrackerSettings = config
        self.geometry: PanoramicGeometry = PanoramicGeometry(num_cameras, config.fov, 90.0)

        # Wire fov changes to geometry
        PanoramicTrackerSettings.fov.bind(config, lambda v: self.geometry.set_fov(v))
        PanoramicTrackerSettings.distortion.bind(config,    lambda v: self.geometry.set_algorithm(v))
        PanoramicTrackerSettings.distortion_k1.bind(config, lambda v: self.geometry.set_k1(v))
        PanoramicTrackerSettings.distortion_k2.bind(config, lambda v: self.geometry.set_k2(v))

        self._callback_lock = Lock()
        self._tracklet_callbacks: set[TrackletDictCallback] = set()

    @property
    def tracker_type(self) -> TrackerType:
        return TrackerType.PANORAMIC

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        super().start()

    def stop(self) -> None:
        self._running = False

        with self._callback_lock:
            self._tracklet_callbacks.clear()

        self.join(timeout=1.0)  # Wait for the thread to finish

    def notify_update(self) -> None:
        if self._running:
            self._update_event.set()

    def run(self) -> None:
        while self._running:
            self._update_event.wait(timeout=0.1)
            self._update_event.clear()

            try:
                while True:
                    try:
                        tracklet: Tracklet = self._input_queue.get(block=False)
                        self._add_tracklet(tracklet)
                    except Empty:
                        break

                self._update_and_notify()
            except Exception:
                logger.exception("PanoramicTracker error")

    def _add_tracklet(self, new_tracklet: Tracklet) -> None:
        # If tracklet is lost or removed, lose it — let timeout + merge logic handle actual removal
        if new_tracklet.is_lost or new_tracklet.is_removed:
            existing_tracklet_id: int | None = self.tracklet_manager.get_id_by_cam_and_external_id(new_tracklet.cam_id, new_tracklet.external_id)
            if existing_tracklet_id is not None:
                self.tracklet_manager.lose_tracklet(existing_tracklet_id)
            return

        # filter out tracklets that are too young or too small
        if new_tracklet.external_age_in_frames <= self.config.tracklet_min_age:
            return
        if new_tracklet.roi.height < self.config.tracklet_min_height:
            return

        # Construct PanoramicTrackerInfo with local and world angles
        local_angle, world_angle, _overlap = self.geometry.get_angles_and_overlap(new_tracklet.roi, new_tracklet.cam_id, self.config.cam_360_edge_threshold)
        new_tracklet = replace(new_tracklet, annotation=PanoramicAnnotation(local_angle, world_angle, _overlap))

        # Filter out tracklets that are too close to the edge of a camera's field of view
        if self.geometry.angle_in_edge(local_angle, self.config.cam_360_edge_threshold):
            return

        # Check if the new tracklet already exists in the tracker and replace
        existing_tracklet_id: int | None = self.tracklet_manager.get_id_by_cam_and_external_id(new_tracklet.cam_id, new_tracklet.external_id)
        if existing_tracklet_id is not None:
            self.tracklet_manager.replace_tracklet(existing_tracklet_id, new_tracklet)

        # If it doesn't exist, add it as a new tracklet
        elif new_tracklet.is_active:
            self.tracklet_manager.add_tracklet(new_tracklet)

    def _update_and_notify(self) -> None:
        # expire timed-out tracklets
        for tracklet in self.tracklet_manager.all_tracklets():
            if tracklet.is_expired(self.config.timeout):
                self.tracklet_manager.retire_tracklet(tracklet.id)

        # merge overlapping tracklets
        overlaps: set[PanoramicOverlapInfo] = self._find_overlapping_tracklets(self.tracklet_manager.all_tracklets())
        for overlap in overlaps:
            self.tracklet_manager.merge_tracklets(overlap.keep_id, overlap.remove_id)

        # notify all active tracklets + any REMOVED ones (so downstream clears their slots)
        callback_tracklets: TrackletDict = {}
        for tracklet in self.tracklet_manager.all_tracklets():
            callback_tracklets[tracklet.id] = tracklet
        self._notify_callback(callback_tracklets)

        # remove REMOVED tracklets after notification
        for tracklet in self.tracklet_manager.all_tracklets():
            if tracklet.status == TrackingStatus.REMOVED:
                self.tracklet_manager.remove_tracklet(tracklet.id)

    def _find_overlapping_tracklets(self, tracklets: list[Tracklet]) -> set[PanoramicOverlapInfo]:

        for i, tracklet in enumerate(tracklets):
            # Reconstruct PanoramicTrackerInfo with updated overlap field
            if isinstance(tracklet.annotation, PanoramicAnnotation):
                updated_overlap = self.geometry.angle_in_overlap(tracklet.annotation.local_angle, self.config.cam_360_overlap_expansion)

                tracklets[i] = replace(
                    tracklet,
                    annotation=PanoramicAnnotation(
                        local_angle=tracklet.annotation.local_angle,
                        world_angle=tracklet.annotation.world_angle,
                        overlap=updated_overlap
                    )
                )

        overlaps: list[PanoramicOverlapInfo] = []

        for p_a, p_b in combinations(tracklets, 2):
            if not isinstance(p_a.annotation, PanoramicAnnotation) or not isinstance(p_b.annotation, PanoramicAnnotation):
                continue
            if not p_a.annotation.overlap or not p_b.annotation.overlap:
                continue
            if p_a.cam_id == p_b.cam_id:
                continue

            angle_diff: float = self.geometry.angle_diff(p_a.annotation.world_angle, p_b.annotation.world_angle)
            if angle_diff > self.geometry.fov_overlap * (1.0 + self.config.cam_360_overlap_expansion):
                continue

            # look at the hight of the trackets for extra filtering
            height_diff: float = abs(p_a.roi.height - p_b.roi.height)
            if height_diff > 0.1:
                continue

            if not p_a.is_active and p_b.is_active:
                overlaps.append(PanoramicOverlapInfo(p_b.id, p_a.id, angle_diff, f'{p_a.id} inactive'))
                continue

            elif not p_b.is_active and p_a.is_active:
                overlaps.append(PanoramicOverlapInfo(p_a.id, p_b.id, angle_diff, f'{p_b.id} inactive'))
                continue

            if not p_a.is_active and not p_b.is_active:
                continue

            if p_a.age_in_seconds < p_b.age_in_seconds:
                newest, oldest = p_a, p_b
                ann_newest, ann_oldest = p_a.annotation, p_b.annotation
            else:
                newest, oldest = p_b, p_a
                ann_newest, ann_oldest = p_b.annotation, p_a.annotation
            edge_newest: float = self.geometry.angle_from_edge(ann_newest.local_angle)
            edge_oldest: float = self.geometry.angle_from_edge(ann_oldest.local_angle)
            # print(f"Comparing newest {newest.id} (edge {edge_newest}) to oldest {oldest.id} (edge {edge_oldest})")
            if edge_newest >= edge_oldest / self.config.cam_360_hysteresis_factor:
                overlaps.append(PanoramicOverlapInfo(newest.id, oldest.id, angle_diff, f'{edge_newest} >= {edge_oldest}'))
            else:
                overlaps.append(PanoramicOverlapInfo(oldest.id, newest.id, angle_diff, f'{edge_newest} < {edge_oldest}'))

        overlap_sets: set[PanoramicOverlapInfo] = set(overlaps)
        return overlap_sets

    # CALLBACKS
    def _notify_callback(self, tracklets: TrackletDict) -> None:
        with self._callback_lock:
            for c in self._tracklet_callbacks:
                c(tracklets)

    def add_tracklet_callback(self, callback: TrackletDictCallback) -> None:
        with self._callback_lock:
            self._tracklet_callbacks.add(callback)

    def submit_cam_tracklets(self, cam_id: int, cam_tracklets: list[DepthTracklet]) -> None:
        for t in cam_tracklets:
            tracklet: Tracklet | None = Tracklet.from_depthcam(cam_id, t)
            if tracklet is None:
                logger.warning(f"PanoramicTracker: Invalid tracklet from camera {cam_id}, skipping.")
                continue
            self._input_queue.put(tracklet)
