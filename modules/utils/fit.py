def fit(src_width: int | float, src_height: int | float, dst_width: int | float, dst_height: int | float) -> tuple[float, float, float, float]:
    """The rectangle (x, y, width, height) that fits a source of that shape inside the destination, centered."""
    src_ratio: float = float(src_width) / float(src_height)
    dst_ratio: float = float(dst_width) / float(dst_height)

    x: float
    y: float
    width: float
    height: float

    if dst_ratio > src_ratio:
        height = dst_height
        width = height * src_ratio
    else:
        width = dst_width
        height = width / src_ratio

    x = (dst_width - width) / 2.0
    y = (dst_height - height) / 2.0

    return (x, y, width, height)


def fill(src_width: int | float, src_height: int | float, dst_width: int | float, dst_height: int | float) -> tuple[float, float, float, float]:
    """The rectangle (x, y, width, height) that fills the destination with a source of that shape, centered."""
    src_ratio: float = float(src_width) / float(src_height)
    dst_ratio: float = float(dst_width) / float(dst_height)

    x: float
    y: float
    width: float
    height: float

    if dst_ratio < src_ratio:
        height = dst_height
        width = height * src_ratio
    else:
        width = dst_width
        height = width / src_ratio

    x = (dst_width - width) / 2.0
    y = (dst_height - height) / 2.0

    return (x, y, width, height)
