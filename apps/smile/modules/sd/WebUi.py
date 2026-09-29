import cv2
import numpy as np
from PIL import Image
from threading import Thread, Lock
from enum import Enum
import time
from copy import deepcopy
from typing import Callable
import random

from modules.sd.ApiConnect import *
from modules.sd.Progress import Progress, ProgressMessage
from modules.sd.Prompter import Prompter

class WebUiState(Enum):
            NONE        = 0
            IDLE        = 1
            CAM_DEPTH   = 2
            PROMPT      = 3
            CN_DEPTH    = 4
            CN_POSE     = 5
            CN_DETAIL   = 6
            DIFFUSION   = 7
            DIFF_LOOP   = 8
            DIF_DEPTH   = 9

class WebUiMessage():
    def __init__(self, state: WebUiState, progress: float = 0, image: np.ndarray | None = None, prompt: str = '', negative: str = '') -> None:
        self.state: WebUiState = state
        self.progress: float = progress
        self.image: np.ndarray | None = image
        self.prompt: str = prompt
        self.negative: str = negative

MessageCallback= Callable[[WebUiMessage], None]

class WebUi(Thread):
    _message_callbacks: list[MessageCallback]

    def __init__(self, webuiPath: str) -> None:
        super().__init__()

        self._input_mutex: Lock = Lock()
        self._thread_running: bool = False
        self._message_callbacks = []

        # IMAGE INPUTS
        self._image: np.ndarray | None = None
        self._image_consumed: bool = True
        self._depth_mage: np.ndarray | None = None
        self._portrait_mode: bool = False

        # PROMPT
        self.prompter: Prompter = Prompter()

        self.repeat_prompt_iter: int = 1
        self.repeat_prompt_char: str = '@'
        self.repeat_prompt_begin: float = 0.5
        self.repeat_prompt_end: float = 1.5

        # SD
        self.sd_enabled: bool = True
        self._sd_txt2img_settings = txt2img_settings()
        self._sd_img2img_settings = img2img_settings()
        self._sd_sampler: str = ''
        self.sd_models: list[str] = []
        self.sd_samplers: list[str] = []
        self.sd_schedulers: list[str] = []
        self.sd_upscalers: list[str] = []
        self.sub_seed_strength: float = 0.0

        # CONTROL NET
        self._control_net_settings: list[control_net_settings] = []
        self.control_net_amount: int = 3
        for i in range(self.control_net_amount):
            self._control_net_settings.append(control_net_settings())
        self.controlnetModels: list[str] = []
        self.controlnetModules: list[str] = []

        self.is_diffusing: bool = False

        # CONNECT TO API
        self.sd_api_connected = False
        if not start_sd(webuiPath): return
        self.sd_api_connected = True
        self.sd_models = get_models()
        self.sd_samplers = get_samplers()
        self.sd_schedulers = get_schedulers()
        self.sd_upscalers = get_upscalers()
        self._sd_sampler: str = self.sd_samplers[0]

        self.controlnetModels = get_control_models()
        self.controlnetModules = get_control_modules()

    def start(self) -> None:
        self._thread_running = True
        super().start()

    def stop(self) -> None:
        self.interrupt()

        self.diffusing = False
        if (self._thread_running):
            self._thread_running = False
            self.join()

    def run(self) -> None:
        while self._thread_running:
            if not self.sd_api_connected:
                time.sleep(0.01)
                continue
            in_image_bgr: np.ndarray | None = self.get_image()
            if in_image_bgr is not None:
                # self.set_image(None)

                # INPUT
                in_image: np.ndarray = cv2.cvtColor(in_image_bgr, cv2.COLOR_BGR2RGB)
                in_string: str = b64_img(Image.fromarray(in_image))

                # SETTINGS
                sd_settings: img2img_settings | txt2img_settings = self.get_sd_img2img_settings()
                if sd_settings.denoising_strength == 1.0:
                    sd_settings = self.get_sd_txt2img_settings()
                else:
                    sd_settings.init_images = [in_string]
                control_res: int = min(sd_settings.width, sd_settings.height)

                #DEPTH IMAGE
                depth_string: str = get_depth_img(in_string, control_res)
                depth_image: np.ndarray = cv2_img(depth_string)
                self.message_callback(WebUiMessage(WebUiState.CAM_DEPTH, image = depth_image))


                result = get_instant_id_info(in_string, control_res)
                if result is None:
                    print ('Instant ID not found')
                    self.message_callback(WebUiMessage(WebUiState.IDLE))
                else:

                    #PROMPT
                    self.prompter.update(in_string)
                    sd_settings.prompt = self.prompter.get_pos_prompt()
                    sd_settings.negative_prompt = self.prompter.get_neg_prompt()
                    self.message_callback(WebUiMessage(WebUiState.PROMPT, prompt=sd_settings.prompt, negative=sd_settings.negative_prompt))

                    #SEED
                    sd_settings.seed = random.randint(1, 2147483647)
                    sd_settings.subseed =random.randint(1, 2147483647)
                    sd_settings.subseed_strength = self.sub_seed_strength

                    #SCRIPTS
                    alwayson_scripts: dict = {'ControlNet' : {'args': []}}

                    #CONTROL NET
                    for index in range(self.control_net_amount):
                        cn_settings: control_net_settings = self.get_cn_settings(index)
                        if cn_settings.enabled and self._thread_running:
                            # cn_settings.processor_res =  int(control_res / 8) * 4
                            cn_settings.image = in_string
                            alwayson_scripts['ControlNet']['args'].append(control_net_dict(cn_settings))

                    sd_settings.alwayson_scripts = alwayson_scripts

                    #I2I
                    if self.sd_enabled and self._thread_running:
                        self.is_diffusing = True

                        ps: list[str] =  self.prompter.get_rep_input().split('@')
                        iterations: int = len(ps)
                        step: float = 0
                        if iterations > 2:
                            rng: float = self.repeat_prompt_end - self.repeat_prompt_begin
                            step = rng / (iterations - 1)
                        else:
                            iterations = 1

                        sd_imgs: list[str] | None = None
                        for i in range(iterations):
                            r_c = str(self.repeat_prompt_begin + step * i)
                            sd_settings.prompt = self.prompter.get_pos_prompt().replace(self.repeat_prompt_char, ps[i])
                            cnl: int = len(sd_settings.alwayson_scripts['ControlNet']['args'])
                            sd_settings.alwayson_scripts['ControlNet']['args'][cnl-1]['weight'] = self.repeat_prompt_begin + step * i
                            sd_settings.subseed = random.randint(1, 2147483647)

                            sd_settings.subseed_strength = self.sub_seed_strength
                            if type(sd_settings) == img2img_settings:
                                sd_imgs = img2img(sd_settings)
                            elif type(sd_settings) == txt2img_settings:
                                sd_imgs = txt2img(sd_settings)
                            if sd_imgs != None and self.is_diffusing:
                                self.message_callback(WebUiMessage(WebUiState.DIFF_LOOP, float(i) / float(iterations), cv2.cvtColor(cv2_img(sd_imgs[0]), cv2.COLOR_RGB2BGR), sd_settings.prompt, sd_settings.negative_prompt))


                            if not self.is_diffusing:
                                print('stop SD loop')
                                break

                        if sd_imgs != None and self.is_diffusing:
                            self.message_callback(WebUiMessage(WebUiState.IDLE, image = cv2_img(get_depth_img(sd_imgs[0], control_res))))


                self.message_callback(WebUiMessage(WebUiState.IDLE))

            time.sleep(0.01)

# MASK
    def apply_depth_mask(self, input: np.ndarray, mask: np.ndarray) -> np.ndarray:
        _, mask_img = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask_img_float: np.ndarray = mask_img.astype(np.float32)
        mask_img_float /= 255.0
        masked_image_float: np.ndarray = cv2.multiply(input.astype(np.float32), mask_img_float)
        return masked_image_float.astype(np.uint8)

# CALLBACKS
    def message_callback(self, message: WebUiMessage) -> None:
        for c in self.get_message_callbacks(): c(message)

    def add_message_callback(self, callback: MessageCallback) -> None:
        with self._input_mutex:
            self._message_callbacks.append(callback)

    def remove_message_callback(self, callback: MessageCallback) -> None:
        with self._input_mutex:
            if callback in self._message_callbacks:
                self._message_callbacks.remove(callback)

    def clear_message_callbacks(self) -> None:
        with self._input_mutex:
            self._message_callbacks = []

    def get_message_callbacks(self) -> list[MessageCallback]:
        with self._input_mutex:
            return self._message_callbacks.copy()


# PROGRESS
    def progress_callback(self, message: ProgressMessage) -> None:
        if message.image:
            cvimg: np.ndarray = cv2_img(message.image)
            self.message_callback(WebUiMessage(WebUiState.DIFFUSION, message.progress, cvimg, '', ''))


# IMAGE INPUTS
    def set_image(self, image: np.ndarray | None) -> None:
        with self._input_mutex:
            self._image = image
            self._image_consumed = False
    def get_image(self, get_cumsumed_image = False) -> np.ndarray | None:
        with self._input_mutex:
            if not self._image_consumed:
                self._image_consumed = True
                return self._image

            if self._image_consumed and get_cumsumed_image:
                return self._image

            return None

    def set_depth_image(self, image: np.ndarray) -> None:
        with self._input_mutex:
            self._depth_mage = image
    def get_depth_image(self) -> np.ndarray | None:
        with self._input_mutex:
            return self._depth_mage

    def set_portrait_mode(self, value: bool) -> None:
        self._portrait_mode = value


# SD
    def set_sd_model(self, value:str) -> None:
        options: dict = {}
        options['sd_model_checkpoint'] = value
        set_options(options)
    def get_sd_models(self) -> list:
        return self.sd_models

    def set_sd_enabled(self, value: bool) -> None:
        self.sd_enabled = bool(value)

    def set_sd_img2img(self, value: bool) -> None:
        self.sd_img2img = value

    def get_sd_img2img_settings(self) -> img2img_settings:
        with self._input_mutex:
            return deepcopy(self._sd_img2img_settings)

    def get_sd_txt2img_settings(self) -> txt2img_settings:
        with self._input_mutex:
            return deepcopy(self._sd_txt2img_settings)

    def set_sd_sampler(self, value:str) -> None:
        with self._input_mutex:
            self._sd_img2img_settings.sampler_name = str(value)
            self._sd_txt2img_settings.sampler_name = str(value)

    def set_sd_cfg_scale(self, value:float) -> None:
        with self._input_mutex:
            self._sd_img2img_settings.cfg_scale = float(value)
            self._sd_txt2img_settings.cfg_scale = float(value)

    def set_sd_steps(self, value:int) -> None:
        with self._input_mutex:
            self._sd_img2img_settings.steps = int(value)
            self._sd_txt2img_settings.steps = int(value)

    def set_sd_denoising(self, value:float) -> None:
        with self._input_mutex:
            self._sd_img2img_settings.denoising_strength = float(value)
            self._sd_txt2img_settings.denoising_strength = float(value)

    def set_sd_seed(self, value:int) -> None:
        with self._input_mutex:
            self._sd_img2img_settings.seed = int(value)
            self._sd_txt2img_settings.seed = int(value)

    def set_sd_eta(self, value:float) -> None:
        with self._input_mutex:
            self._sd_img2img_settings.eta = float(value)
            self._sd_txt2img_settings.eta = float(value)

    def set_sd_scheduler(self, value:str) -> None:
        with self._input_mutex:
            self._sd_img2img_settings.scheduler = str(value)
            self._sd_txt2img_settings.scheduler = str(value)

    def set_sd_width(self, value:int) -> None:
        with self._input_mutex:
            if self._portrait_mode:
                self._sd_img2img_settings.height = int(value)
                self._sd_txt2img_settings.height = int(value)
            else:
                self._sd_img2img_settings.width = int(value)
                self._sd_txt2img_settings.width = int(value)

    def set_sd_height(self, value:int) -> None:
        with self._input_mutex:
            if self._portrait_mode:
                self._sd_img2img_settings.width = int(value)
                self._sd_txt2img_settings.width = int(value)
            else:
                self._sd_img2img_settings.height = int(value)
                self._sd_txt2img_settings.height = int(value)

    def set_sd_restore_faces(self, value:bool) -> None:
        with self._input_mutex:
            self._sd_img2img_settings.restore_faces = bool(value)
            self._sd_txt2img_settings.restore_faces = bool(value)

    def set_sd_tiling(self, value:bool) -> None:
        with self._input_mutex:
            self._sd_img2img_settings.tiling = bool(value)
            self._sd_txt2img_settings.tiling = bool(value)

    def set_sd_styles(self, value:list[str]) -> None:
        with self._input_mutex:
            self._sd_img2img_settings.styles = value
            self._sd_txt2img_settings.styles = value

# CONTROL NET
    def get_cn_settings(self, index: int) -> control_net_settings:
        with self._input_mutex:
            return deepcopy(self._control_net_settings[index])

    def set_cn_enable(self, index: int, value: bool) -> None:
        with self._input_mutex:
            self._control_net_settings[index].enabled = bool(value)

    def set_cn_model(self, index: int, value: str) -> None:
        with self._input_mutex:
            self._control_net_settings[index].model = value

    def set_cn_module(self, index: int, value: str) -> None:
        with self._input_mutex:
            self._control_net_settings[index].module = value

    def set_cn_thr_max(self, index: int, value: float) -> None:
        with self._input_mutex:
            self._control_net_settings[index].thr_max = float(value)

    def set_cn_thr_min(self, index: int, value: float) -> None:
        with self._input_mutex:
            self._control_net_settings[index].thr_min = float(value)

    def set_cn_weight(self, index: int, value: float) -> None:
        with self._input_mutex:
            self._control_net_settings[index].weight = float(value)

    def set_cn_guide_start(self, index: int, value: float) -> None:
        with self._input_mutex:
            self._control_net_settings[index].guide_start = float(value)

    def set_cn_guide_end(self, index: int, value: float) -> None:
        with self._input_mutex:
            self._control_net_settings[index].guide_end = float(value)

    def set_cn_apply_depth_mask(self, index: int, value: bool) -> None:
        with self._input_mutex:
            self._control_net_settings[index].apply_depth_mask = bool(value)


    def set_repeat_prompt_iter(self, value: int) -> None:
        self.repeat_prompt_iter= int(value)
    def set_repeat_prompt_begin(self, value: float) -> None:
        self.repeat_prompt_begin= value
    def set_repeat_prompt_end(self, value: float) -> None:
        self.repeat_prompt_end= value

    def interrupt(self) -> None:
        if self.is_diffusing:
            self.is_diffusing = False
            interrupt_sd()


    def get_face_info(self, image: np.ndarray) -> dict | None:
        if self.sd_enabled and not self.is_diffusing:
            # get image dimensions
            height, width = image.shape[:2]
            res = max(height, width)
            try:
                string_img: str = b64_img(Image.fromarray(image))
            except:
                return None
            return get_instant_id_info(string_img, width)

        return None