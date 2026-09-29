from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame
import modules.sd.WebUi as SdWebUi
from modules.sd.WebUi import WebUiMessage, WebUiState, interrupt_sd

max_seed = 2**32 - 1 # 4294967295, max unsigned 32-bit integer
max_seed_characters = 10

def str_to_seed(input_str: str) -> int | None:
    input_str = str(input_str)
    if not input_str:
            return None
    input_str = input_str[:max_seed_characters]
    try:
        seed = int(input_str)
        if seed == 0:
            return None
        if seed > max_seed:
            return max_seed
        return seed
    except ValueError:
        # print(f"Error: '{input_str}' is not a valid decimal number")
        return None

def str_float_check(value: str) -> str | None:
    try:
        float_value = float(value)
        return str(float_value)
    except ValueError:
        return None

def str_id_check(value: str) -> str | None:
        try:
            id = int(value)
            id = f"{id:04}"
            return id
        except ValueError:
            return None

class WebUiGui(SdWebUi.WebUi):
    def __init__(self, gui: Gui , webuiPath: str) -> None:
        self.gui: Gui = gui
        super().__init__(webuiPath)

        elem: list = []
        if not self.sd_api_connected:
            elem.append([E(eT.TEXT, 'NO ACCESS TO SD WEBUI')])
            self.diff_frame = Frame('DIFFUSION', elem, 540)
            elem = []
            self.prompt_frame = Frame('PROMPT', elem, 540)
            return

        elem.append([E(eT.CHCK, 'Enable',       self.set_sd_enabled,        False),
                     E(eT.BTTN, 'Interrupt',    self.set_interrupt_gui,     False),
                     E(eT.SLDR, 'Subseed',      self.set_sub_seed_gui, 0,    (0,1),   0.01,      False),
                     E(eT.TEXT, 'Seed'),
                     E(eT.ITXT, 'Seed',         self.set_seed_gui,          '')])
        elem.append([E(eT.TEXT, '         '),
                     E(eT.TEXT, 'Model  '),
                     E(eT.CMBO, 'Model',        self.set_sd_model,          self.sd_models[0],          self.sd_models)])
        elem.append([E(eT.TEXT, '         '),
                     E(eT.TEXT, 'Samp',         None,                       True),
                     E(eT.CMBO, 'Sampler',      self.set_sd_sampler,        self.sd_samplers[0],        self.sd_samplers),
                     E(eT.TEXT, 'Sched',        None,                       True),
                     E(eT.CMBO, 'Scheduler',    self.set_sd_scheduler,      self.sd_schedulers[0],      self.sd_schedulers)])
        elem.append([E(eT.TEXT, '         '),
                     E(eT.TEXT, 'Steps'),
                     E(eT.SLDR, 'SD_Steps',     self.set_sd_steps,          20,     (1,40),     1,      False),
                     E(eT.TEXT, 'Denois'),
                     E(eT.SLDR, 'SD_Denoise',   self.set_sd_denoising,      .5,     ( 0,1),     .05,    False),
                     E(eT.TEXT, 'CFG'),
                     E(eT.SLDR, 'SD_CFGScale',  self.set_sd_cfg_scale,      10,     ( 1,30),    .5,     False)])
        elem.append([E(eT.TEXT, '         '),
                     E(eT.TEXT, 'Width'),
                     E(eT.SLDR, 'SD_Width',     self.set_sd_width,          512,    (0,1920),   4,      False),
                     E(eT.TEXT, 'Height'),
                     E(eT.SLDR, 'SD_Height',    self.set_sd_height,         512,    (0,1920),   4,      False),
                     E(eT.TEXT, 'Eta'),
                     E(eT.SLDR, 'SD_Eta',       self.set_sd_eta,            1,      (0,1),      .05,    False)])
        elem.append([E(eT.SEPR, '')])
        elem.append([E(eT.TEXT, 'CONTROL  '),
                     E(eT.CHCK, '1',            self.set_cn_enable_1,       False),
                     E(eT.CMBO, 'Module_1',     self.set_cn_module_1,       'none', self.controlnetModules),
                     E(eT.CMBO, 'Model_1',      self.set_cn_model_1,        'none', self.controlnetModels)])
        elem.append([E(eT.TEXT, '         '),
                     E(eT.TEXT, 'Threshold'),
                     E(eT.SLDR, 'ThresMin_1',   self.set_cn_thr_min_1,      0,      (0,1),      0.05,   False),
                     E(eT.TEXT, 'Min | Max'),
                     E(eT.SLDR, 'ThresMax_1',   self.set_cn_thr_max_1,      1,      (0,1),      0.05,   False),
                     E(eT.CHCK, 'Mask',         self.set_cn_mask_1,         False)])
        elem.append([E(eT.TEXT, '         '),
                     E(eT.TEXT, 'Weight'),
                     E(eT.SLDR, 'Weight_1',     self.set_cn_weight_1,       1,      (0,2),      .01,     False),
                     E(eT.TEXT, 'Start'),
                     E(eT.SLDR, 'Start_1',      self.set_cn_guide_start_1,  0,      (0,1),      .01,    False),
                     E(eT.TEXT, 'end'),
                     E(eT.SLDR, 'End_1',        self.set_cn_guide_end_1,    1.0,    (0,1),      .01,    False)])
        elem.append([E(eT.SEPR, '')])
        elem.append([E(eT.TEXT, 'CONTROL  '),
                     E(eT.CHCK, '2',            self.set_cn_enable_2,       False),
                     E(eT.CMBO, 'Module_2',     self.set_cn_module_2,       'none', self.controlnetModules),
                     E(eT.CMBO, 'Model_2',      self.set_cn_model_2,        'none', self.controlnetModels)])
        elem.append([E(eT.TEXT, '         '),
                     E(eT.TEXT, 'Weight'),
                     E(eT.SLDR, 'Weight_2',     self.set_cn_weight_2,       1,      (0,2),      .01,     False),
                     E(eT.TEXT, 'Start'),
                     E(eT.SLDR, 'Start_2',      self.set_cn_guide_start_2,  0,      (0,1),      .01,    False),
                     E(eT.TEXT, 'end'),
                     E(eT.SLDR, 'End_2',        self.set_cn_guide_end_2,    1.0,    (0,1),      .01,    False)])
        elem.append([E(eT.SEPR, '')])
        elem.append([E(eT.TEXT, 'CONTROL  '),
                     E(eT.CHCK, '3',            self.set_cn_enable_3,       False),
                     E(eT.CMBO, 'Module_3',     self.set_cn_module_3,       'none', self.controlnetModules),
                     E(eT.CMBO, 'Model_3',      self.set_cn_model_3,        'none', self.controlnetModels)])
        elem.append([E(eT.TEXT, '         '),
                     E(eT.TEXT, 'Weight'),
                     E(eT.SLDR, 'Weight_3',     self.set_cn_weight_3,       1,      (0,2),      .01,     False),
                     E(eT.TEXT, 'Start'),
                     E(eT.SLDR, 'Start_3',      self.set_cn_guide_start_3,  0,      (0,1),      .01,    False),
                     E(eT.TEXT, 'end'),
                     E(eT.SLDR, 'End_3',        self.set_cn_guide_end_3,    1.0,    (0,1),      .01,    False)])
        elem.append([E(eT.SEPR, '')])
        elem.append([E(eT.TEXT, 'NDI      '),
                     E(eT.CHCK, 'Capture',      self.set_capture_ndi,       False),
                     E(eT.CHCK, 'Steps',        self.set_ndi_steps,         True),
                     E(eT.CHCK, 'Last',         self.set_ndi_result,        True),
                     E(eT.CHCK, 'Prompt',       self.set_ndi_prompt,        True)])

        self.diff_frame = Frame('DIFFUSER', elem, 540)


        self.smile_key = 'smile'
        elem= []
        elem.append([E(eT.TEXT, 'SMILE    '),
                     E(eT.SLDR, 'smile_amnt',   self.set_smile_repeat_gui,      10,      (1,50),      1,    True)])
        elem.append([E(eT.TEXT, '         '),
                     E(eT.ITXT, 'mov_id',       self.set_smile_id_gui,          '0',    expand=False),
                     E(eT.CMBO, 'mov_name',     self.set_smile_name_gui,        self.prompter.get_keyword_name(self.smile_key), self.prompter.get_keyword_names(self.smile_key))])
        elem.append([E(eT.TEXT, 'POSITIVE '),
                     E(eT.TEXT, '*'),
                     E(eT.CMBO, 'ins_pos_key',  self.insert_pos_key,            self.prompter.keyword_names[0],                 self.prompter.keyword_names,    expand=False),
                     E(eT.TEXT, '#'),
                     E(eT.CMBO, 'ins_pos_wld',  self.insert_pos_wld,            self.prompter.wildcard_names[0],                self.prompter.wildcard_names,   expand=False),
                     E(eT.TEXT, '+'),
                     E(eT.CMBO, 'ins_pos_ask',  self.insert_pos_ask,            self.prompter.ask_names[0],                     self.prompter.ask_names,        expand=False)])
        elem.append([E(eT.MLTL, 'PosInput',     self.prompter.set_pos_input,    'Positive Prompt Input',    resolution=3)])
        elem.append([E(eT.MLTL, 'RepInput',     self.prompter.set_rep_input,    '',    resolution=12)])
        elem.append([E(eT.TEXT, 'REPLACE  ')  ,
                    #  E(eT.SLDR, 'rp_iter',      self.set_repeat_prompt_iter,    1,   (0, 20), 1),
                     E(eT.SLDR, 'rp_min',       self.set_repeat_prompt_begin,   1.0,   (0.0, 2.0), 0.025),
                     E(eT.SLDR, 'rp_max',       self.set_repeat_prompt_end,     0.5,   (0.0, 2.0), 0.025)])
        elem.append([E(eT.MLTL, 'PosPrompt',    None,                           'Positive Promt Result',    resolution=3)])
        elem.append([E(eT.SEPR, '')])
        elem.append([E(eT.TEXT, 'NEGATIVE ')])
        elem.append([E(eT.MLTL, 'NegInput',     self.prompter.set_neg_input,    'Negative Prompt Input',    resolution=3)])
        elem.append([E(eT.MLTL, 'NegPrompt',    None,                           'Negative Prompt Result',   resolution=4)])
        self.prompt_frame = Frame('PROMPT', elem, 540)



    def message_callback(self, message: WebUiMessage) -> None:
        super().message_callback(message)
        if not self.gui: return
        if not self.gui.isRunning(): return
        if message.prompt != '':
          self.gui.updateElement('PosPrompt', message.prompt)
        if message.negative != '':
          self.gui.updateElement('NegPrompt', message.negative)

        self.update_smile_gui()

    def set_seed_gui(self, seed_string: str) -> None:
        if not self.gui: return
        if not self.gui.isRunning(): return
        seed = str_to_seed(seed_string)
        if seed:
            self.gui.updateElement('Seed', seed)
            self.set_sd_seed(seed)
        else:
            self.gui.updateElement('Seed', '')
            self.set_sd_seed(-1)

    def set_sub_seed_gui(self, value: float) -> None:
        self.sub_seed_strength = value

    def set_interrupt_gui(self) -> None:
        self.interrupt()


    def get_frame_diff(self):
        return self.diff_frame

    def get_frame_prompt(self):
        return self.prompt_frame

    def set_cn_enable_1(self, value: bool) -> None: self.set_cn_enable(0, value)
    def set_cn_enable_2(self, value: bool) -> None: self.set_cn_enable(1, value)
    def set_cn_enable_3(self, value: bool) -> None: self.set_cn_enable(2, value)

    def set_cn_model_1(self, value: str) -> None: self.set_cn_model(0, value)
    def set_cn_model_2(self, value: str) -> None: self.set_cn_model(1, value)
    def set_cn_model_3(self, value: str) -> None: self.set_cn_model(2, value)

    def set_cn_module_1(self, value: str) -> None: self.set_cn_module(0, value)
    def set_cn_module_2(self, value: str) -> None: self.set_cn_module(1, value)
    def set_cn_module_3(self, value: str) -> None: self.set_cn_module(2, value)

    def set_cn_thr_max_1(self, value: float) -> None: self.set_cn_thr_max(0, value)
    def set_cn_thr_min_1(self, value: float) -> None: self.set_cn_thr_min(0, value)
    def set_cn_mask_1(self, value: bool) -> None: self.set_cn_apply_depth_mask(0, value)

    def set_cn_weight_1(self, value: float) -> None: self.set_cn_weight(0, value)
    def set_cn_weight_2(self, value: float) -> None: self.set_cn_weight(1, value)
    def set_cn_weight_3(self, value: float) -> None: self.set_cn_weight(2, value)

    def set_cn_guide_start_1(self, value: float) -> None: self.set_cn_guide_start(0, value)
    def set_cn_guide_start_2(self, value: float) -> None: self.set_cn_guide_start(1, value)
    def set_cn_guide_start_3(self, value: float) -> None: self.set_cn_guide_start(2, value)

    def set_cn_guide_end_1(self, value: float) -> None: self.set_cn_guide_end(0, value)
    def set_cn_guide_end_2(self, value: float) -> None: self.set_cn_guide_end(1, value)
    def set_cn_guide_end_3(self, value: float) -> None: self.set_cn_guide_end(2, value)

    def vote(self) -> None:
        self.prompter.vote()
        self.update_prompt_gui()

    def update_prompt_gui(self) -> None:
        self.update_smile_gui()

    def set_smile_repeat_gui(self, value: int) -> None:
        self.prompter.smile_repeat = int(value)

    def set_smile_name_gui(self, value:str) -> None:
        id: str | None = self.prompter.get_id_from_name(self.smile_key, value)
        if id:
            self.prompter.set_keyword_id(self.smile_key, id)
            self.update_smile_gui()

    def set_smile_id_gui(self, value: str) -> None:
        id: str | None = str_id_check(value)
        if id:
            self.prompter.set_keyword_id(self.smile_key, id)
            self.update_smile_gui()

    def update_smile_gui(self) -> None:
        id: str = self.prompter.get_keyword_id(self.smile_key)
        name: str = self.prompter.get_keyword_name(self.smile_key)
        self.gui.updateElement('mov_id', id)
        self.gui.updateElement('mov_name', name)


    def insert_pos_key(self, value) -> None:
        return
        prompt = self.prompter.get_pos_input()
        symbol = self.prompter.keyword_symbol
        if not prompt:
            return
        prompt += f', {symbol}{value}{symbol}'
        self.gui.updateElement('PosInput', prompt)

    def insert_pos_wld(self, value) -> None:
        return
        prompt = self.prompter.get_pos_input()
        symbol = self.prompter.wildcard_symbol
        if not prompt:
            return
        prompt += f', {symbol}{value}{symbol}'
        self.gui.updateElement('PosInput', prompt)

    def insert_pos_ask(self, value) -> None:
        return
        prompt = self.prompter.get_pos_input()
        symbol = self.prompter.ask_symbol
        if not prompt:
            return
        prompt += f', {symbol}{value}{symbol}'
        self.gui.updateElement('PosInput', prompt)

    def insert_neg_key(self, value) -> None:
        return
        prompt = self.gui.getValue('NegInput')
        symbol = self.prompter.keyword_symbol
        if not prompt:
            return
        prompt += f', {symbol}{value}{symbol}'
        self.gui.updateElement('NegInput', prompt)

    def insert_neg_wld(self, value) -> None:
        return
        prompt = self.gui.getValue('NegInput')
        symbol = self.prompter.wildcard_symbol
        if not prompt:
            return
        prompt += f', {symbol}{value}{symbol}'
        self.gui.updateElement('NegInput', prompt)