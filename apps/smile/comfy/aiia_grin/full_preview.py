import contextlib
import logging

import numpy as np
from PIL import Image

import comfy.sample
import comfy.utils
import nodes


class _NothingUnloaded(logging.Filter):
    def filter(self, record):
        return record.getMessage() != "0 models unloaded."


@contextlib.contextmanager
def _quiet_console():
    """The VAE decode of every step asks for free VRAM on a card ComfyUI keeps full on purpose, and
    ComfyUI answers each time that it could unload nothing: a line per step through the progress
    bar. Kept off the console while this node samples; a log file still gets it."""
    handlers = [h for h in logging.getLogger().handlers
                if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
    quiet = _NothingUnloaded()
    for handler in handlers:
        handler.addFilter(quiet)
    try:
        yield
    finally:
        for handler in handlers:
            handler.removeFilter(quiet)


class KSamplerFullPreview:
    """KSampler that previews every step with the real VAE (like A1111's "Full" live preview).

    Each step's denoised prediction is decoded with the checkpoint's VAE, so the previews have the
    quality of the final image and the last one is the final image. The sampling itself is the
    same as the KSampler node (same noise, same seed, same result)."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = nodes.KSampler.INPUT_TYPES()
        inputs["required"]["vae"] = ("VAE",)
        return inputs

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "AIIA"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, vae, denoise=1.0):
        latent = latent_image
        samples = comfy.sample.fix_empty_latent_channels(model, latent["samples"], latent.get("downscale_ratio_spacial", None), latent.get("downscale_ratio_temporal", None))
        noise = comfy.sample.prepare_noise(samples, seed, latent.get("batch_index", None))
        noise_mask = latent.get("noise_mask", None)

        pbar = comfy.utils.ProgressBar(steps)

        def callback(step, x0, x, total_steps):
            if x0.is_nested:
                x0 = x0.tensors[0]
            images = vae.decode(model.model.process_latent_out(x0[:1]))
            image = images.reshape(-1, *images.shape[-3:])[0]
            preview = Image.fromarray(np.clip(image.cpu().numpy() * 255, 0, 255).astype(np.uint8))
            pbar.update_absolute(step + 1, total_steps, ("JPEG", preview, None))

        with _quiet_console():
            result = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, samples,
                                         denoise=denoise, noise_mask=noise_mask, callback=callback,
                                         disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = result
        return (out,)
