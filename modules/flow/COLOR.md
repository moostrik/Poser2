# The color

The density's color: the stage between the view and the motion that makes what the dye looks like, from the
view's image and mask ([VIEW.md](VIEW.md)). The view sees the player, the color makes the dye's look, the bridge
carries it beside the amount ([BRIDGE_3D.md](BRIDGE_3D.md#the-mix)). **(deduction)** marks what is inferred.

The stage is `ColorLayer` (settings `color`, `modules/render/modules/flow/color_layer.py`); its work is
`DensityColor` (`modules/flow/color`), its shaders `Grade`, `Sharpen`, `LumaMatte`, `Erode`, `Blur`, `Feather` and
`MaskMultiply` (`modules/gl`), the white balance's matrix `white_balance.py`. The 2D bridge's tint is apart
(`TintColor`).

## The stage

It runs once per capture after the view, at the view's size, and is made again when the view was made again or one
of its settings changes, so also while the app holds a frame. The motion reads the view's own image and mask, not
the color: a grade would change the gradients the Lucas–Kanade solve sees **(deduction)**.

| Step              | Does                                                                                                  |
|-------------------|-------------------------------------------------------------------------------------------------------|
| white balance     | the light the image is corrected for (`white_balance`, K), the grade's first step                     |
| grade             | `exposure`, `offset`, `gamma`, `contrast`, `saturation` (`Grade`)                                     |
| sharpen           | `sharpen` (`Sharpen`)                                                                                 |
| luma matte        | its alpha a matte from its brightness (`luma_matte`, `LumaMatte`)                                     |
| the mask's copy   | eroded by `erode` and feathered inward over `feather` view pixels (`Erode`, `Blur`, `Feather`)        |
| premultiply       | RGB and alpha by the copy (`use_mask`, `MaskMultiply`)                                                |

Its output, `color`, is the bridge's image: the density's color, its alpha the coverage. It works on the straight
image and premultiplies it by the mask last, RGB and alpha alike, as compositing does and as the render blends
(Porter and Duff 1984; Brinkmann): grading or sharpening a cut image changes its half-masked edge apart from the
rest and pulls the cut's black into the edge. With `enabled` off the color is the image as it is, matted and masked
as set.

The white balance is the grade's first step, before the exposure, as a camera's: `white_balance` is the light the
image is corrected for, in kelvin, lower cooler and higher warmer, 6500 K unchanged. The color is decoded to linear
light, its white adapted from that light's (a blackbody's, on the Planckian locus by Kim et al. 2002) to the 6500 K
blackbody's by Bradford's chromatic adaptation, its brightness kept, and encoded again (`Grade`); the matrix is made
on the CPU when the setting changes (`white_balance.py`). It colors only the density: the camera's own `balance` and
`auto_balance` change the whole image, which the pose sees too, and with `auto_balance` on the camera's colors move
under it. Named white balance, as temperature is the fluid's.

The luma matte, after the grading and the sharpening and before the premultiply, is a matte from the color's
brightness, its brightest channel (HSV's value), as video compositing mattes a picture by its brightness: darkness
becomes transparency. The alpha becomes 1 − `luma_matte` + `luma_matte` × that brightness, the RGB as it is
(`LumaMatte`). The premultiply then multiplies both by the mask's copy, so the stored alpha is the feather times the
matte. At 1, the default, a half-dark red is full red at half the alpha and black brings in nothing, so dark
clothes and shadows leave no dark patches in the fluid **(site fact)**; at 0 the alpha is the feather's alone. With
`use_mask` off the color has no edge.

The premultiply takes a copy of the mask, worked as compositing works a matte where the picture is made, so the
mask the motion and the depth use stays as it is:
- `erode` moves the copy's edge inward by whole pixels, one 3 × 3 minimum each (`Erode`), dropping the outermost
  pixels, whose color the camera mixed with the background's;
- `feather` then fades it in inward over its width: the copy blurred by a Gaussian of half the width is 0.5 on its
  edge, and 2 × it − 1, clamped (`Feather`), is 0 on and outside the edge and rises to 1 inside, so nothing reaches
  past the eroded edge.

The color is resampled twice, straightened in the uploader and cut to the view, a little softer than one resample
**(deduction)**, for keeping the camera's work and the flow's apart.

## Checked

Moved from the view into its own stage with the same shaders: on the show recording (`20261112-130105_show`, 40
captures, deep_flow's preset), the last capture's color is the same as the view's was, the largest difference 0 of
255 (`apps/deep_flow/scripts/view_check.py --compare`, 18 November 2026).
