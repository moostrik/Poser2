import cv2
import numpy as np
from PIL import Image
import requests
import io
import base64
import os
import time
from datetime import datetime


webUiUrl = 'http://127.0.0.1:7860'

def check_sd_running() -> bool:
    try:
        response = requests.get(url=f'{webUiUrl}/app_id')
    except :
        return False
    else:
        print('SD Webui is running with app id', response.json()['app_id'])
        return True

def check_sd_api_acces() -> bool:
    response: requests.Response = requests.get(url=f'{webUiUrl}/sdapi/v1/sd-models')
    if 'detail' in response.json() and response.json()['detail'] == 'Not Found':
        print('SD Webui is not running with api enabled')
        return False
    if not response.json():
        print('No response from SD Webui Api')
        return False
    return True

def set_settings() -> None:
    options: dict = {}
    options['export_for_4chan'] = False
    options['auto_launch_browser'] = "Disable"
    options['img2img_fix_steps'] = True
    options['show_progress_every_n_steps'] = 1
    options['show_progress_type'] = 'Full' #'TAESD'
    options['live_preview_content'] = 'Combined'
    options['save_init_img'] = True
    options['interrogate_keep_models_in_memory'] = True
    set_options(options)

def Check_control() -> bool:
    unit_count = 4
    try:
        response: requests.Response = requests.get(url=f'{webUiUrl}/controlnet/settings')
    except:
        print('failed to get control net settings')
        return False
    if not 'control_net_unit_count' in response.json():
        print('WARNING control net Model cache size not set')
        return False
    if response.json()['control_net_unit_count'] is not unit_count:
        print('WARNING control net Model cache size not set to', unit_count, response.json()['control_net_unit_count'] )
        return False
    return True


def start_sd(webuiPath: str) -> bool:
    if not check_sd_running():
        print('no image generator running')
        return False
    time.sleep(0.33)
    if not check_sd_api_acces():
        return False
    set_settings()
    # Check_control()
    return True

def interrupt_sd() -> None:
    requests.post(url=f'{webUiUrl}/sdapi/v1/interrupt')


def b64_img(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format='PNG')
    img_base64: str = 'data:image/png;base64,' + str(base64.b64encode(buffered.getvalue()), 'utf-8')
    return img_base64

def raw_b64_img(image: Image.Image) -> str:
    # XXX controlnet only accepts RAW base64 without headers
    buffered = io.BytesIO()
    image.save(buffered, format='PNG')
    img_base64: str = str(base64.b64encode(buffered.getvalue()), 'utf-8')
    return img_base64

def cv2_img(b64_img: str, rgb2bgr: bool = False) -> cv2.Mat:
    image = np.array(Image.open(io.BytesIO(base64.b64decode(b64_img)))) # type: ignore
    if rgb2bgr:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # type: ignore
    return image # type: ignore

def interrogate(image: str, useBooru: bool = False) -> str:
    model: str = 'deepdanbooru' if (useBooru) else 'clip'
    payload: dict[str, str] = {'image': image, 'model': model }
    response: requests.Response = requests.post(url=f'{webUiUrl}/sdapi/v1/interrogate', json=payload)
    return response.json()['caption']

def get_models() -> list[str]:
    response: requests.Response = requests.get(url=f'{webUiUrl}/sdapi/v1/sd-models')
    L: list[str] = []
    for r in response.json():
        L.append(r['title'])
    return L

def get_samplers() -> list[str]:
    response: requests.Response = requests.get(url=f'{webUiUrl}/sdapi/v1/samplers')
    L: list[str] = []
    for r in response.json():
        L.append(r['name'])
    return L

def get_schedulers() -> list[str]:
    response: requests.Response = requests.get(url=f'{webUiUrl}/sdapi/v1/schedulers')
    L: list[str] = ['unknown']
    # for r in response.json():
    #     L.append(r['name'])
    return L

def get_upscalers() -> list[str]:
    # response: requests.Response = requests.get(url=f'{webUiUrl}/sdapi/v1/upscalers')
    # L: list[str] = []
    # for r in response.json():
    #     L.append(r['name'])

    L: list[str] = ['None', 'ESRGAN_4x']
    return L

def get_styles() -> list[str]:
    response: requests.Response = requests.get(url=f'{webUiUrl}/sdapi/v1/prompt-styles')
    styleNames: list[str] = ['']
    styles: dict[str, dict[str, str]] = {'':{'prompt':'', 'negative_prompt':''}}

    for r in response.json():
        name: str = r['name']
        prompt: str = r['prompt']
        negative_prompt = r['negative_prompt']
        styleNames.append(r['name'])
        styles[name] = {'prompt':prompt, 'negative_prompt':negative_prompt}
    return styleNames

def set_options(options) -> None:
    try:
        response: requests.Response = requests.post(url=f'{webUiUrl}/sdapi/v1/options', json=options)
    except:
        print('failed to set options', options)

# download models from https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main
# more info https://github.com/lllyasviel/ControlNet-v1-1-nightly

def get_control_models() -> list:
    response: requests.Response = requests.get(url=f'{webUiUrl}/controlnet/model_list')
    print('MODEL LIST', response.json())
    L: list[str] = response.json()['model_list']
    return ['none'] + L

def get_control_modules() -> list:
    response: requests.Response = requests.get(url=f'{webUiUrl}/controlnet/module_list')
    L: list[str] = response.json()['module_list']
    L = [L[0]] + sorted(L[1:])
    return L

class control_net_settings:
    def __init__(self) -> None:
        self.enabled: bool  = True
        self.detect_images: list[str] = []
        self.image: str = ''
        self.mask: str = 'none'
        self.model: str = 'none'
        self.module: str = 'none'
        self.thr_min: float = 0.0
        self.thr_max: float = 1.0
        self.weight: float = 1.0
        self.resize_mode: str = 'Crop and Resize'
        self.low_vram: bool = False
        self.processor_res: int = -1
        self.guide_start: float = 0.0
        self.guide_end: float = 1.0
        self.control_mode: str = 'Balanced'
        self.pixel_perfect: bool = True
        self.apply_depth_mask: bool = False

def control_net_detect(settings: control_net_settings) -> list[str] :
    if settings.module == 'canny':
        settings.thr_min *= 255
        settings.thr_max *= 255

    payload: dict = {
        "controlnet_module": settings.module,
        "controlnet_input_images": settings.detect_images,
        "controlnet_processor_res": settings.processor_res,
        "controlnet_threshold_a": settings.thr_min,
        "controlnet_threshold_b": settings.thr_max
    }
    response: requests.Response = requests.post(url=f'{webUiUrl}/controlnet/detect', json=payload)
    result: dict = response.json()
    if 'images' in result.keys():
        return result['images']
    return ['']

def control_net_dict(settings: control_net_settings) -> dict :
   return { 'image': settings.image,
            'enabled': settings.enabled,
            # 'mask': settings.mask,
            'module': settings.module,
            'model': settings.model,
            'weight': settings.weight,
            'resize_mode': settings.resize_mode,
            'low_vram': settings.low_vram,
            'processor_res': settings.processor_res,
            'threshold_a': settings.thr_min,
            'threshold_b': settings.thr_max,
            'guidance_start': settings.guide_start,
            'guidance_end': settings.guide_end,
            'control_mode' : settings.control_mode,
            'pixel_perfect' : settings.pixel_perfect
    }

def get_depth_img(image: str, width: int) -> str:
    depth_settings = control_net_settings()
    depth_settings.detect_images = [image]
    depth_settings.processor_res = width
    depth_settings.pixel_perfect = True
    depth_settings.module = 'depth'
    control_net_image: str = control_net_detect(depth_settings)[0]
    return control_net_image
    return np.array(Image.open(io.BytesIO(base64.b64decode(control_net_image)))) # type: ignore

def get_instant_id_info(image: str, width: int) -> dict | None:
    settings = control_net_settings()
    settings.detect_images = [image]
    settings.processor_res = width
    settings.pixel_perfect = True
    settings.module = 'instant_id_face_info'

    payload: dict = {
        "controlnet_module": settings.module,
        "controlnet_input_images": settings.detect_images,
        "controlnet_processor_res": settings.processor_res,
        "controlnet_threshold_a": settings.thr_min,
        "controlnet_threshold_b": settings.thr_max
    }
    response: requests.Response = requests.post(url=f'{webUiUrl}/controlnet/detect', json=payload)
    result: dict = response.json()

    info: dict = {}
    if 'error' in result.keys():
        return None
    return result['face_info'][0]

class img2img_settings:
    def __init__(self) -> None:
        self.prompt: str = ''
        self.negative_prompt: str = ''
        self.styles: list[str] = []
        self.seed: int = -1
        self.subseed: int = -1
        self.subseed_strength: float = 0.0
        self.seed_resize_from_h: int = -1
        self.seed_resize_from_w: int = -1
        self.sampler_name: str = 'DPM++ SDE'
        self.scheduler: str = 'Karras'
        self.batch_size: int = 1
        self.n_iter: int = 1
        self.steps: int = 20
        self.cfg_scale: float = 7.0
        self.width: int = 512
        self.height: int = 512
        self.restore_faces: bool = False
        self.tiling: bool = False
        self.do_not_save_samples: bool = True
        self.do_not_save_grid: bool = True
        self.eta: float = 1.0
        self.denoising_strength: float = 0.75
        self.s_min_uncond: float = 0.0
        self.s_churn: float = 0.0
        self.s_tmax: float = 0.0
        self.s_tmin: float = 0.0
        self.s_noise: float = 1.0
        self.override_settings: dict = {}
        self.override_settings_restore_afterwards: bool = False
        self.refiner_checkpoint: str = ''
        self.refiner_switch_at: int = 0
        self.disable_extra_networks: bool = False
        self.firstpass_image: str = ''
        self.comments: dict = {}
        self.init_images: list[str] = []
        self.resize_mode: int = 0
        self.image_cfg_scale: float = 0
        self.mask: str = ''
        self.mask_blur_x: int = 4
        self.mask_blur_y: int = 4
        self.mask_blur: int = 0
        self.mask_round: bool = True
        self.inpainting_fill: int = 0
        self.inpaint_full_res: bool = True
        self.inpaint_full_res_padding : int = 0
        self.inpainting_mask_invert: int = 0
        self.initial_noise_multiplier: float = 1.0
        self.latent_mask: str = ''
        self.force_task_id: str = ''
        # self.sampler_index: str = 'Euler'
        self.include_init_images: bool = False
        self.script_name: str = ''
        self.script_args: list[str] =[]
        self.send_images: bool = True
        self.save_images: bool = True
        self.alwayson_scripts: dict = {}
        self.infotext: str = ''

def img2img(settings: img2img_settings) -> list[str] | None:
    payload: dict = {
        'prompt':                   settings.prompt,
        'negative_prompt':          settings.negative_prompt,
        'styles':                   settings.styles,
        'seed':                     settings.seed,
        'subseed':                  settings.subseed,
        'subseed_strength':         settings.subseed_strength,
        'seed_resize_from_h':       settings.seed_resize_from_h,
        'seed_resize_from_w':       settings.seed_resize_from_w,
        'sampler_name':             settings.sampler_name,
        'scheduler':                settings.scheduler,
        'batch_size':               settings.batch_size,
        'n_iter':                   settings.n_iter,
        'steps':                    settings.steps,
        'cfg_scale':                settings.cfg_scale,
        'width':                    settings.width,
        'height':                   settings.height,
        'restore_faces':            settings.restore_faces,
        'tiling':                   settings.tiling,
        'do_not_save_samples':      settings.do_not_save_samples,
        'do_not_save_grid':         settings.do_not_save_grid,
        'eta':                      settings.eta,
        'denoising_strength':       settings.denoising_strength,
        's_min_uncond':             settings.s_min_uncond,
        's_churn':                  settings.s_churn,
        's_tmax':                   settings.s_tmax,
        's_tmin':                   settings.s_tmin,
        's_noise':                  settings.s_noise,
        'override_settings':        settings.override_settings,
        'override_settings_restore_afterwards': settings.override_settings_restore_afterwards,
        'refiner_checkpoint':       settings.refiner_checkpoint,
        'refiner_switch_at':        settings.refiner_switch_at,
        'disable_extra_networks':   settings.disable_extra_networks,
        'firstpass_image':          settings.firstpass_image,
        'comments':                 settings.comments,
        'init_images':              settings.init_images,
        'resize_mode':              settings.resize_mode,
        'image_cfg_scale':          settings.image_cfg_scale,
        'mask':                     settings.mask,
        'mask_blur_x':              settings.mask_blur_x,
        'mask_blur_y':              settings.mask_blur_y,
        'mask_blur':                settings.mask_blur,
        'mask_round':               settings.mask_round,
        'inpainting_fill':          settings.inpainting_fill,
        'inpaint_full_res':         settings.inpaint_full_res,
        'inpaint_full_res_padding': settings.inpaint_full_res_padding,
        'inpainting_mask_invert':   settings.inpainting_mask_invert,
        'initial_noise_multiplier': settings.initial_noise_multiplier,
        'layent_mask':              settings.latent_mask,
        'force_task_id':            settings.force_task_id,
        # 'sampler_index':            settings.sampler_index,
        'include_init_images':      settings.include_init_images,
        'script_name':              settings.script_name,
        'script_args':              settings.script_args,
        'send_images':              settings.send_images,
        'save_images':              settings.save_images,
        'alwayson_scripts':         settings.alwayson_scripts,
        'infotext':                 settings.infotext
    }
    response: requests.Response = requests.post(url=f'{webUiUrl}/sdapi/v1/img2img', json=payload)
    result: dict = response.json()
    if 'images' in result.keys():
        return result['images']
    return None

class txt2img_settings:
    def __init__(self) -> None:
        self.prompt: str = ''
        self.negative_prompt: str = ''
        self.styles: list[str] = []
        self.seed: int = -1
        self.subseed: int = -1
        self.subseed_strength: float = 0.0
        self.seed_resize_from_h: int = -1
        self.seed_resize_from_w: int = -1
        self.sampler_name: str = 'DPM++ SDE'
        self.scheduler: str = 'Karras'
        self.batch_size: int = 1
        self.n_iter: int = 1
        self.steps: int = 20
        self.cfg_scale: float = 7.0
        self.width: int = 512
        self.height: int = 512
        self.restore_faces: bool = False
        self.tiling: bool = False
        self.do_not_save_samples: bool = True
        self.do_not_save_grid: bool = True
        self.eta: float = 1.0
        self.denoising_strength: float = 0.75
        self.s_min_uncond: float = 0.0
        self.s_churn: float = 0.0
        self.s_tmax: float = 0.0
        self.s_tmin: float = 0.0
        self.s_noise: float = 1.0
        self.override_settings: dict = {}
        self.override_settings_restore_afterwards: bool = False
        self.refiner_checkpoint: str = ''
        self.refiner_switch_at: int = 0
        self.disable_extra_networks: bool = False
        self.firstpass_image: str = ''
        self.comments: dict = {}
        self.enable_hr: bool = False
        self.firstphase_width: float = 0.0
        self.firstphase_height: float = 0.0
        self.hr_scale: float = 0
        self.hr_upscaler: str = 'None'
        self.hr_second_pass_steps: int = 0
        self.hr_resize_x: int = 0
        self.hr_resize_y: int = 0
        self.hr_sampler_name: str = ''
        self.hr_checkpoint_name: str = ''
        self.hr_sampler_name: str = ''
        self.hr_scheduler: str = ''
        self.hr_prompt: str = ''
        self.hr_negative_prompt: str = ''
        self.force_task_id: str = ''
        # self.sampler_index: str = 'Euler'
        self.script_name: str = ''
        self.script_args: list[str] =[]
        self.send_images: bool = True
        self.save_images: bool = True
        self.alwayson_scripts: dict = {}
        self.infotext: str = ''

def txt2img(settings: txt2img_settings) -> list[str] | None:
    payload: dict = {
        'prompt':               settings.prompt,
        'negative_prompt':      settings.negative_prompt,
        'styles':               settings.styles,
        'seed':                 settings.seed,
        'subseed':              settings.subseed,
        'subseed_strength':     settings.subseed_strength,
        'seed_resize_from_h':   settings.seed_resize_from_h,
        'seed_resize_from_w':   settings.seed_resize_from_w,
        'sampler_name':         settings.sampler_name,
        'scheduler':            settings.scheduler,
        'batch_size':           settings.batch_size,
        'n_iter':               settings.n_iter,
        'steps':                settings.steps,
        'cfg_scale':            settings.cfg_scale,
        'width':                settings.width,
        'height':               settings.height,
        'restore_faces':        settings.restore_faces,
        'tiling':               settings.tiling,
        'do_not_save_samples':  settings.do_not_save_samples,
        'do_not_save_grid':     settings.do_not_save_grid,
        'eta':                  settings.eta,
        'denoising_strength':   settings.denoising_strength,
        's_min_uncond':         settings.s_min_uncond,
        's_churn':              settings.s_churn,
        's_tmax':               settings.s_tmax,
        's_tmin':               settings.s_tmin,
        's_noise':              settings.s_noise,
        'override_settings':    settings.override_settings,
        'override_settings_restore_afterwards': settings.override_settings_restore_afterwards,
        'refiner_checkpoint':   settings.refiner_checkpoint,
        'refiner_switch_at':    settings.refiner_switch_at,
        'disable_extra_networks': settings.disable_extra_networks,
        'firstpass_image':      settings.firstpass_image,
        'comments':             settings.comments,
        'enable_hr':            settings.enable_hr,
        'firstphase_width':     settings.firstphase_width,
        'firstphase_height':    settings.firstphase_height,
        'hr_scale':             settings.hr_scale,
        'hr_upscaler':          settings.hr_upscaler,
        'hr_second_pass_steps': settings.hr_second_pass_steps,
        'hr_resize_x':          settings.hr_resize_x,
        'hr_resize_y':          settings.hr_resize_y,
        'hr_checkpoint_name':   settings.hr_checkpoint_name,
        'hr_sampler_name':      settings.hr_sampler_name,
        'hr_scheduler':         settings.hr_scheduler,
        'hr_prompt':            settings.hr_prompt,
        'hr_negative_prompt':   settings.hr_negative_prompt,
        'force_task_id':        settings.force_task_id,
        'script_args':          settings.script_args,
        # 'sampler_index':        settings.sampler_index,
        'script_name':          settings.script_name,
        'script_args':          settings.script_args,
        'send_images':          settings.send_images,
        'save_images':          settings.save_images,
        'alwayson_scripts':     settings.alwayson_scripts,
        'infotext':             settings.infotext
    }
    response: requests.Response = requests.post(url=f'{webUiUrl}/sdapi/v1/txt2img', json=payload)
    result: dict = response.json()
    if 'images' in result.keys():
        return result['images']
    return None

def calc_hr_scale(width: int, height: int) -> float:
    res: float = float(min(width, height))
    return res / 512.0

def save_image(image: cv2.Mat, path:str, is_input: bool):
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    in_out: str = '_in' if is_input else '_sd'
    output_filename: str = f"{path}{timestamp}{in_out}.jpg"
    cv2.imwrite(output_filename, image)
