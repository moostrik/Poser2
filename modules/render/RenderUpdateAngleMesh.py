import numpy as np

from modules.gl.Mesh import Mesh
from modules.pose.PoseStream import PoseStreamData

def update_angle_mesh(pose_stream: PoseStreamData | None, angle_mesh: Mesh) -> None:

    if pose_stream is None:
        return

    angles_np: np.ndarray = np.nan_to_num(pose_stream.angles.to_numpy(), nan=0.0)
    conf_np: np.ndarray = pose_stream.confidences.to_numpy()
    if angles_np.shape[0] != conf_np.shape[0] or angles_np.shape[1] != conf_np.shape[1]:
        print(f"Angles shape {angles_np.shape} does not match confidences shape {conf_np.shape}")
        return

    mesh_data: np.ndarray = np.stack([angles_np, conf_np], axis=-1)
    capacity: int = pose_stream.capacity

    if mesh_data.shape[0] < capacity:
        # Pad the mesh data to match the capacity
        padding: np.ndarray = np.zeros((capacity - mesh_data.shape[0], mesh_data.shape[1], mesh_data.shape[2]), dtype=mesh_data.dtype)
        mesh_data = np.concatenate([padding, mesh_data], axis=0)

    # Only use the first 4 joints
    data: np.ndarray = mesh_data[:, :4, :]
    num_frames, num_joints, _ = data.shape

    if num_frames < 2 or num_joints < 1:
        return

    # Prepare confidences and angles
    confidences: np.ndarray = np.clip(data[..., 1], 0, 1)
    confidences = np.where(confidences > 0, 0.7, 0.0).astype(np.float32)
    angles_raw: np.ndarray = data[..., 0]
    angles_norm: np.ndarray = np.clip(np.abs(angles_raw) / np.pi, 0, 1)
    joint_height: float = 1.0 / (num_joints)

    # INDICES
    base = np.arange(num_joints) * num_frames
    frame_idx: np.ndarray = np.arange(num_frames - 1)
    start = base[:, None] + frame_idx
    end = start + 1
    indices: np.ndarray = np.stack([start, end], axis=-1).reshape(-1, 2).astype(np.uint32).flatten()
    angle_mesh.set_indices(indices)

    # VERTICES
    frame_grid, joint_grid = np.meshgrid(np.arange(num_frames), np.arange(num_joints), indexing='ij')
    x = frame_grid / (num_frames - 1)
    y = (joint_grid) * joint_height + angles_norm * joint_height - 0.05
    vertices: np.ndarray = np.zeros((num_frames * num_joints, 3), dtype=np.float32)
    vertices[:, 0] = x.T.flatten()
    vertices[:, 1] = y.T.flatten()
    angle_mesh.set_vertices(vertices)

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
    colors[:, 3] = conf_flat
    angle_mesh.set_colors(colors)
    angle_mesh.update()
