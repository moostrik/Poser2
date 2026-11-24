"""`fit` and `fill`: where a source of one shape lands, unstretched and centered, in a destination of another."""


def fit(src_width: int | float, src_height: int | float, dst_width: int | float, dst_height: int | float) -> tuple[float, float, float, float]:
    """The rectangle (x, y, width, height) that fits a source of that shape inside the destination, centered."""
    return _place(src_width, src_height, dst_width, dst_height, cover=False)


def fill(src_width: int | float, src_height: int | float, dst_width: int | float, dst_height: int | float) -> tuple[float, float, float, float]:
    """The rectangle (x, y, width, height) that fills the destination with a source of that shape, centered."""
    return _place(src_width, src_height, dst_width, dst_height, cover=True)


def _place(src_width: int | float, src_height: int | float, dst_width: int | float, dst_height: int | float,
           cover: bool) -> tuple[float, float, float, float]:
    """The source scaled to the destination's height or width, centered: the one that covers it when `cover`, else
    the one that fits inside it."""
    src_ratio: float = float(src_width) / float(src_height)
    dst_ratio: float = float(dst_width) / float(dst_height)
    height_bound: bool = dst_ratio < src_ratio if cover else dst_ratio > src_ratio

    width: float
    height: float
    if height_bound:
        height = dst_height
        width = height * src_ratio
    else:
        width = dst_width
        height = width / src_ratio

    return ((dst_width - width) / 2.0, (dst_height - height) / 2.0, width, height)
