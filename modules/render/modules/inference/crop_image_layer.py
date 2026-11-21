from modules.board import HasCropImages

from ..board import BoardImageLayer


def crop_image_layer(track_id: int, board: HasCropImages) -> BoardImageLayer:
    """The player's pose crop on the board, the pose model's input, as a layer."""
    def read():
        image = board.get_crop_image(track_id)
        return (image.crop, None) if image else None
    return BoardImageLayer(read)
