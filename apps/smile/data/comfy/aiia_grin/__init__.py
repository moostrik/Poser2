import contextlib
import io
import math
import sys

import cv2
import numpy as np
import torch
from PIL import Image

import comfy.utils
import nodes

# face parameter, default target value
TARGETS = [
    ("smile", 1.8),
    ("aaa", 50.0),
    ("eee", 15.0),
    ("eyebrow", 8.0),
    ("blink", -9.0),
    ("wink", 0.0),
    ("woo", 0.0),
    ("pupil_x", 0.0),
    ("pupil_y", 0.0),
]

# head rotation, default random range (min, max) in degrees
ROTATIONS = [
    ("rotate_pitch", -15.0, 15.0),
    ("rotate_yaw", -15.0, 15.0),
    ("rotate_roll", -15.0, 15.0),
]

EASES = {
    "smoothstep": lambda t: t * t * (3 - 2 * t),
    "linear": lambda t: t,
    "ease_in": lambda t: t ** 3,
    "ease_out": lambda t: 1 - (1 - t) ** 3,
    "sine_in_out": lambda t: 0.5 - 0.5 * math.cos(math.pi * t),
}


class GrinFrames:
    """Animate a portrait as a loop: from neutral up to an extreme expression with a random head
    turn and back to neutral, each parameter peaking at its own moment (random, or given as
    <name>_peak), one frame at a time.

    Uses Advanced Live Portrait's Expression Editor for the face math. Each frame is sent as a
    progress preview the moment it is rendered; the output is all frames as one batch."""

    def __init__(self):
        self.editor = None
        self.pasted = None      # (the editor's prepared source, the same with our paste transform and mask)

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "frames": ("INT", {"default": 40, "min": 1, "max": 1000}),
            "ease": (list(EASES),),
            "crop_factor": ("FLOAT", {"default": 2.0, "min": 1.5, "max": 3.0, "step": 0.1}),
            "midpoint": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "Moment of the peak as a fraction of the loop."}),
            "midpoint_deviation": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01,
                                             "tooltip": "Each parameter peaks at a random moment within midpoint +- this."}),
            "stitching": ("BOOLEAN", {"default": True,
                                      "tooltip": "LivePortrait stitching. It fits a pasted face to the original image, "
                                                 "but when the face region covers the whole frame it only shifts the head."}),
        }
        for name, value in TARGETS:
            required[name] = ("FLOAT", {"default": value, "min": -1000.0, "max": 1000.0, "step": 0.1})
        for name, lo, hi in ROTATIONS:
            required[f"{name}_min"] = ("FLOAT", {"default": lo, "min": -1000.0, "max": 1000.0, "step": 0.1})
            required[f"{name}_max"] = ("FLOAT", {"default": hi, "min": -1000.0, "max": 1000.0, "step": 0.1})
        optional = {
            "still": ("BOOLEAN", {"default": False,
                                  "tooltip": "Only the peak pose: one frame with every parameter at its target."}),
        }
        for name in [name for name, _ in TARGETS] + [name for name, _, _ in ROTATIONS]:
            optional[f"{name}_peak"] = ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01,
                                                  "tooltip": "Moment this parameter peaks, as a fraction of the loop. "
                                                             "-1: random within midpoint +- deviation."})
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "render"
    CATEGORY = "AIIA"

    def render(self, image, seed, frames, ease, crop_factor, midpoint, midpoint_deviation, stitching=True, still=False, **params):
        if self.editor is None:
            self.editor = nodes.NODE_CLASS_MAPPINGS["ExpressionEditor"]()
        fixed = {"src_ratio": 1.0, "sample_ratio": 1.0, "crop_factor": crop_factor}

        targets = {name: params[name] for name, _ in TARGETS}
        rng = np.random.default_rng(seed)
        for name, _, _ in ROTATIONS:
            targets[name] = float(rng.uniform(params[f"{name}_min"], params[f"{name}_max"]))
        peaks = {}
        for name in targets:
            # always drawn, so a given moment doesn't shift the draws of the others
            moment = midpoint + rng.uniform(-midpoint_deviation, midpoint_deviation)
            given = params.get(f"{name}_peak", -1.0)
            peaks[name] = float(np.clip(given if given >= 0 else moment, 0.01, 0.99))

        # prepare the face once; the editor caches it while the portrait stays the same
        neutral = {name: 0.0 for name in targets}
        # the pack prints a debug line on every call; anything else it says still shows
        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            link = self.editor.run(**neutral, **fixed, src_image=image)["result"][1]
        for line in said.getvalue().splitlines():
            if line.strip() and line.strip() != "Prepare source...":
                print(line)
        psi = self.paste_setup(link[0], crop_factor)
        curve = EASES[ease]

        def envelope(t, peak):
            # up from neutral to the peak, then back down; t runs 0..1 over one loop
            return curve(t / peak) if t <= peak else curve((1 - t) / (1 - peak))

        # a still is the peak pose: one frame with every parameter at its target at the same moment
        count = 1 if still else frames
        pbar = comfy.utils.ProgressBar(count)
        out = []
        for i in range(count):
            t = i / frames  # the next loop starts at neutral again, so no doubled frame at the seam
            values = targets if still else {name: envelope(t, peaks[name]) * target for name, target in targets.items()}
            frame = self.render_frame(psi, values, stitching)
            out.append(frame)
            preview = Image.fromarray(np.clip(frame[0].cpu().numpy() * 255, 0, 255).astype(np.uint8))
            pbar.update_absolute(i + 1, count, ("JPEG", preview, None))
        return (torch.cat(out, dim=0),)

    def paste_setup(self, psi, crop_factor):
        """The prepared source with the transform that puts the 512 px render back into the portrait and the
        paste mask, both made from the detector's crop square. The pack makes its mask from that square clipped
        to the image and, when it was clipped, its transform from the unclipped square, so a face near an edge
        pastes with the mask off by the clipped margin. Done here, from the same detector, so the pack runs as
        published. Cached while the editor keeps the same prepared source."""
        if self.pasted is not None and self.pasted[0] is psi:
            return self.pasted[1]
        alp = sys.modules[type(self.editor).__module__]
        engine = alp.g_engine
        crop = engine.detect_face(psi.src_rgb, crop_factor)
        crop_trans_m = alp.create_transform_matrix(crop[0], crop[1], (crop[2] - crop[0]) / 512., (crop[3] - crop[1]) / 512.)
        mask = cv2.warpAffine(engine.GetMaskImg(), crop_trans_m, alp.get_rgb_size(psi.src_rgb), cv2.INTER_LINEAR)
        ours = alp.PreparedSrcImg(psi.src_rgb, crop_trans_m, psi.x_s_info, psi.f_s_user, psi.x_s_user, mask.astype(np.float32) / 255.)
        self.pasted = (psi, ours)
        return ours

    def render_frame(self, psi, v, stitching):
        """One frame, the same steps as Advanced Live Portrait's ExpressionEditor.run (src_ratio 1,
        no sample image), but with stitching optional."""
        alp = sys.modules[type(self.editor).__module__]
        engine = alp.g_engine
        pipeline = engine.get_pipeline()

        s = psi.x_s_info
        s_exp = s["exp"] * 1.0
        s_exp[0, 5] = s["exp"][0, 5]
        s_exp += s["kp"]

        es = alp.ExpressionSet()
        es.r = engine.calc_fe(es.e, v["blink"], v["eyebrow"], v["wink"], v["pupil_x"], v["pupil_y"], v["aaa"], v["eee"],
                              v["woo"], v["smile"], v["rotate_pitch"], -v["rotate_yaw"], v["rotate_roll"])
        rotation = alp.get_rotation_matrix(s["pitch"] + es.r[0], s["yaw"] + es.r[1], s["roll"] + es.r[2])
        x_d = (s["scale"] * (1 + es.s)) * ((s_exp + es.e) @ rotation) + s["t"]
        if stitching:
            x_d = pipeline.stitching(psi.x_s_user, x_d)

        crop_out = pipeline.parse_output(pipeline.warp_decode(psi.f_s_user, psi.x_s_user, x_d)["out"])[0]
        # the 512 px render is stretched about 2x over the portrait: Lanczos keeps its edges crisper than linear
        full = cv2.warpAffine(crop_out, psi.crop_trans_m, alp.get_rgb_size(psi.src_rgb), cv2.INTER_LANCZOS4)
        frame = np.clip(psi.mask_ori * full + (1 - psi.mask_ori) * psi.src_rgb, 0, 255).astype(np.uint8)
        return torch.from_numpy(frame.astype(np.float32) / 255.0).unsqueeze(0)


from .full_preview import KSamplerFullPreview

NODE_CLASS_MAPPINGS = {"GrinFrames": GrinFrames, "KSamplerFullPreview": KSamplerFullPreview}
NODE_DISPLAY_NAME_MAPPINGS = {"GrinFrames": "Grin Frames", "KSamplerFullPreview": "KSampler (full preview)"}
