from modules.board import HasCameraImages

from ..board import BoardImageLayer


def camera_image_layer(cam_id: int, board: HasCameraImages) -> BoardImageLayer:
    """The camera's image on the board, as a layer, with its capture time."""
    def read():
        image = board.get_camera_image(cam_id)
        return (image.image, image.timestamp) if image else None
    return BoardImageLayer(read)
