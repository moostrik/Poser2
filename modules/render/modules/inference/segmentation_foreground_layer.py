from OpenGL.GL import GL_CLAMP_TO_BORDER

from modules.board import HasSegmentationImages

from ..board import BoardImageLayer


def segmentation_foreground_layer(track_id: int, board: HasSegmentationImages) -> BoardImageLayer:
    """The player's segmented foreground on the board, as a layer, transparent outside it, with the capture time of the
    frame it was made from."""
    def read():
        image = board.get_segmentation_image(track_id)
        return (image.foreground, image.timestamp) if image else None
    return BoardImageLayer(read, GL_CLAMP_TO_BORDER, (0.0, 0.0, 0.0, 0.0))
