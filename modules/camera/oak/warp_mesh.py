"""The OAK-D's warp meshes as its Warp node takes them: the geometry's mesh points as depthai points (CAMERA.md, the
OAK-D)."""

import logging

import depthai as dai

from ..geometry import MountConfig, WARP_MESH, warp_mesh_points, keystone_mesh_points, lens_reach_report

logger = logging.getLogger(__name__)

KEYSTONE_MESH_W: int = 2
KEYSTONE_MESH_H: int = 64
LENS_REACH_WARNING: float = 3.0     # px black at an edge over which the lens reach warns


def find_warp(
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    mode_width: int,
    mount: MountConfig,
    mesh_w: int = WARP_MESH,
    mesh_h: int = WARP_MESH,
) -> list[dai.Point2f]:
    """`warp_mesh_points` as the Warp node's depthai points."""
    return _points(warp_mesh_points(
        src_size, out_size, mode_width, mount.fov_h, mount.tilt,
        mount.flip_h, mount.flip_v, mesh_w, mesh_h, mount.lens_fov, mount.lens_center))


def clip_warp_mesh(
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    mode_width: int,
    mount: MountConfig,
) -> tuple[list[dai.Point2f], int, int]:
    """The warp mesh for a raw clip (`clip_warp`): the reprojection and the tilt without the flips and the crop the
    clip already carries; with its mesh size."""
    clip_mount = MountConfig(flip_h=False, flip_v=False,
                             tilt=-mount.tilt if mount.flip_v else mount.tilt,
                             keystone=0.0,           # a recording is already post-keystone
                             fov_h=mount.fov_h,
                             lens_fov=mount.lens_fov,
                             lens_center_x=-mount.lens_center_x if mount.flip_h else mount.lens_center_x,
                             lens_center_y=-mount.lens_center_y if mount.flip_v else mount.lens_center_y)
    mesh: list[dai.Point2f] = find_warp(src_size, out_size, mode_width, clip_mount)
    return mesh, WARP_MESH, WARP_MESH


def build_warp_mesh(
    src_size: tuple[int, int],
    out_size: tuple[int, int],
    mode_width: int,
    mount: MountConfig,
) -> tuple[list[dai.Point2f], int, int]:
    """The mesh of the one correction a camera uses, (mesh, mesh_w, mesh_h) for `setWarpMesh`: the keystone
    only for a non-zero `keystone` alone, else the warp; both set warns and takes the tilt."""
    if mount.tilt != 0.0 and mount.keystone != 0.0:
        logger.warning("tilt (%.1f) and keystone (%.2f) are both set; they are exclusive — using tilt",
                       mount.tilt, mount.keystone)
    if mount.tilt != 0.0 or mount.keystone == 0.0:
        _log_lens_reach(src_size, mode_width, mount)
        return find_warp(src_size, out_size, mode_width, mount), WARP_MESH, WARP_MESH

    src_w, src_h = src_size
    out_w, out_h = out_size
    if out_w == src_w:                                          # wide: output is the whole frame
        source_rect: tuple[float, float, float, float] = (0.0, 0.0, float(src_w), float(src_h))
        offset: float = src_w * 0.5 * mount.keystone
    else:                                                       # square: cut from the center
        side: int = out_h - 1                                   # one row less: no yellow line at the bottom
        source_rect = ((src_w - side) / 2, (src_h - side) / 2, float(side), float(side))
        offset = src_h * 0.5 * mount.keystone
    mesh = _points(keystone_mesh_points(source_rect, offset, mount.flip_h, mount.flip_v, KEYSTONE_MESH_W, KEYSTONE_MESH_H))
    return mesh, KEYSTONE_MESH_W, KEYSTONE_MESH_H


def _log_lens_reach(src_size: tuple[int, int], mode_width: int, mount: MountConfig) -> None:
    """The lens reach's report, a warning when the black edge is wider than `LENS_REACH_WARNING`."""
    report = lens_reach_report(src_size, mode_width, mount)
    if report is not None:
        message, pixels = report
        (logger.warning if pixels > LENS_REACH_WARNING else logger.info)(message)


def _points(points: list[tuple[float, float]]) -> list[dai.Point2f]:
    return [dai.Point2f(x, y) for x, y in points]
