from dataclasses import dataclass, field
from enum import Enum
from threading import Thread, Lock, current_thread
from modules.gui.PySimpleGui import PySimpleGui as sg
import os
import glob
from queue import Queue, Empty
from time import sleep

from modules.ConfigBase import ConfigBase

FrameWidget = sg.Frame

BASEHEIGHT = 40
ELEMHEIGHT = 20  # Standard element height
SLIDERHEIGHT = 30  # Slider elements are taller
FRAMEWIDTH = 600

class eType(Enum):
    NONE = 0
    TEXT = 1
    BTTN = 2
    CHCK = 3
    SLDR = 4
    ITXT = 5
    MLTL = 6
    CMBO = 7
    SEPR = 8
    PBAR = 9
    LIST = 10

class qMessageType(Enum):
    NONE = 0
    EVENT = 1
    LOAD = 2
    SAVE = 3
    TOP = 4
    FRONT = 5
    BACK = 6

class qMessage:
    def __init__(self, type: qMessageType, value: str | bool| int | float | None = None, key: str = '', useCallback: bool = False) -> None:
        self.type = type
        self.value = value
        self.key = key
        self.useCallback = useCallback

@dataclass
class GuiConfig(ConfigBase):
    title: str = field(default="Unknown")
    on_top: bool = field(default=True)
    location_x: int = field(default=0)
    location_y: int = field(default=0)
    file_path: str = field(default="")
    default_file: str = field(default="settings")

def Element(type: eType, key: str, callback=None, value: bool| int | float | str = 1, range: tuple[int | float, int |float] | list=(0, 1), resolution: int | float =0.1, expand: bool = True , size=(None, None), text: str | None = None):
    element = None

    disabled = False
    enable_events = True
    if callback is None:
        disabled = True
        enable_events = False

    if type == eType.TEXT :
        element = sg.Text(text = key, auto_size_text=True)

    elif type == eType.SEPR :
        element = sg.HorizontalSeparator(pad = (5, 10)) # type: ignore

    elif type == eType.BTTN :
        element = sg.Button(button_text=key, key = key, expand_x = expand, metadata = callback)

    elif type == eType.CHCK :
        element = sg.Checkbox(text = '', key = key, default = bool(value), metadata = callback, enable_events=enable_events, disabled=disabled)

    elif type == eType.SLDR:
        if size == (None, None): size = (16, 8)
        element = sg.Slider(key = key, default_value = value,  range = range,  resolution = resolution,
                             expand_x=expand, orientation='h', metadata = callback,
                             enable_events=enable_events, disabled=disabled, size = (16,8))
    elif type == eType.ITXT :
        if size == (None, None): size = (10,1)
        # Disable live events - only trigger callback on Return key or focus loss
        element = sg.InputText(key = key, default_text = str(value), size=size, expand_x=expand, metadata = callback,
                               enable_events=False, disabled=disabled)
    elif type == eType.MLTL :
        if resolution > 1:
            element = sg.Multiline(key = key, default_text = str(value), expand_x=True, expand_y=False, size=(1, resolution), metadata = callback,
                                   enable_events=enable_events, disabled=disabled, )
        else:
            element = sg.Multiline(key = key, default_text = str(value), expand_x=True, expand_y=True, metadata = callback,
                                   enable_events=enable_events, disabled=disabled, )
    elif type == eType.CMBO :
        element = sg.Combo(key = key, values = range, default_value = value, metadata = callback, enable_events=enable_events, disabled=disabled, expand_x=expand, size=size)
    elif type == eType.PBAR:
        element = sg.ProgressBar(key = key, max_value=range[1], orientation='horizontal', size=(10,19), expand_x=True)
    elif type == eType.LIST:
        element = sg.Listbox(key = key, values=range, default_values=[], metadata = callback, enable_events=enable_events, disabled=disabled)
    else: print('type unsupported', type, key)

    return element

def Frame(name, elementList, height = 100, width = FRAMEWIDTH) ->sg.Frame:
    frame = sg.Frame(title=name, layout=elementList, size=(width, height), expand_x=False, expand_y=True)
    return frame

def UpdateWindow(window: sg.Window, exitCallback = None) -> bool:
    event, values = window.read(timeout=33) # type: ignore
    if event == 'Escape:27' or event == sg.WIN_CLOSED:
        return False
    elif event == 'Return:16' or event == '\r':
        # Return key pressed - trigger callback for focused InputText element
        focused = window.find_element_with_focus()
        if focused and type(focused) == sg.InputText:
            callback = focused.metadata
            if callback:
                callback(values[focused.Key])
    elif not type(window.find_element(event, True)) == sg.ErrorElement:
        callback = window[event].metadata
        if callback:
            element = window[event]
            if type(element) == sg.Button:
                if element.key == 'Exit':
                    window.close()
                callback()
            else:
                callback(values[event])
    return True

def UpdateEvent(window: sg.Window, message: qMessage) -> None:
    if type(window.find_element(message.key, True)) == sg.ErrorElement:
        # print('key not found', message.key)
        return
    element = window[message.key]
    if type(element) != sg.Button:
        element.update(message.value) # type: ignore
        if message.useCallback:
            callback = element.metadata
            if callback:
                callback(message.value)

def Load(window: sg.Window, settings: sg.UserSettings) -> None :
    if not settings.exists():
        PopUp(window, 'Settings not loaded, no file named ' + str(settings.filename), 0)
        return
    values = settings.load()
    for key in values:
        UpdateEvent(window, qMessage(qMessageType.EVENT, values[key], key, True))
    PopUp(window, 'Settings Loaded from ' + str(settings.filename), 1.5)

def Save(window: sg.Window, settings: sg.UserSettings) -> None :
    event, values = window.read(1) # type: ignore

    delList = []
    for v in values:
        if not window[v].metadata: delList.append(v)
    for d in delList: values.pop(d)

    settings.write_new_dictionary(values)
    settings.save()
    PopUp(window, 'Settings Saved to ' + str(settings.filename), 1.5)

def Top(window: sg.Window, value) -> None:
    if (value): window.keep_on_top_set()
    else: window.keep_on_top_clear()

def PopUp(window: sg.Window, text: str, duration: float = 1) -> None:
    loc = window.current_location(True)
    if not loc: return
    sizX, sizY = window.size
    location = loc[0], loc[1] + int(sizY) # type: ignore
    if duration > 0:
        sg.popup(text, auto_close=True, auto_close_duration=duration, location=location)
    else:
        sg.popup(text, location=location)

class Gui(Thread):
    def __init__(self, settings: GuiConfig, exitCallback = None) -> None:
        super().__init__()
        sg.theme('DarkBlack')
        sg.set_options(font=("consolas", 10))
        self.windowName: str = settings.title + ' GUI'
        self.defaultSettingsName = settings.default_file
        self.settings: sg.UserSettings = sg.UserSettings(path = settings.file_path, filename = self.defaultSettingsName + '.json')
        self.on_top: bool = settings.on_top
        self.location_x: int = settings.location_x
        self.location_y: int = settings.location_y
        self.exit_lock = Lock()
        self.exit_callback = exitCallback
        self.messageQueue: Queue = Queue()
        self.layout: list = []
        self.element_types: dict[str, eType] = {}  # Track element types by key
        self.running: bool = False
        self.defaultFrameWidth: int = 500
        self.allElementsLoaded: bool = False

    def start(self) -> None:
        elem = []
        elem.append([Element(eType.TEXT, '© Matthias Oostrik 2026')])
        autograph = Frame('TITLE', elem, 60)
        elem = []
        elem.append([Element(eType.BTTN, 'Exit', self.call_exit_callback),
                     Element(eType.TEXT, ' '),
                     Element(eType.CMBO, 'SettingsFile', self.set_settings_name, self.defaultSettingsName, self.get_setting_names()),
                     Element(eType.BTTN, 'Save', self.saveSettings),
                     Element(eType.BTTN, 'Load', self.loadSettings)])
        frame = Frame('APP', elem, 60)
        self.addFrame([autograph, frame])

        self.window: sg.Window | None = None
        self.running: bool = True
        self.allElementsLoaded: bool = False
        super().start()
        while not self.allElementsLoaded:
            sleep(0.01)

    def stop(self) -> None:
        if self.running == False:
            return
        with self.exit_lock:
            self.exit_callback = None
        self.running = False

        if current_thread() != self:
            self.join()
            return

    def run(self) -> None:
        self.window = sg.Window(self.windowName, self.layout, keep_on_top=self.on_top, finalize=True,return_keyboard_events=True, location=(self.location_x, self.location_y))
        self.loadSettings()

        while self.running:
            if UpdateWindow(self.window) == False:
                self.call_exit_callback()
                self.stop()
                self.window.close()

            while not self.messageQueue.empty() and self.running:
                m: qMessage = self.messageQueue.get(True)
                if m.type == qMessageType.EVENT: UpdateEvent(self.window, m)
                elif m.type == qMessageType.LOAD:
                    self.settings.filename = self.getStringValue('SettingsFile') + '.json'
                    self.settings.full_filename = None
                    if self.settings.filename == '.json': self.settings.filename = self.defaultSettingsName + '.json'
                    Load(self.window, self.settings)
                elif m.type == qMessageType.SAVE:
                    self.settings.filename = self.getStringValue('SettingsFile') + '.json'
                    self.settings.full_filename = None
                    Save(self.window, self.settings)
                elif m.type == qMessageType.TOP: Top(self.window, m.value)
                elif m.type == qMessageType.FRONT: self.window.BringToFront()
                elif m.type == qMessageType.BACK: self.window.SendToBack()

            self.allElementsLoaded = True

        self.window.close()
        self.window = None

    def addElement(self, type: eType, key: str, callback=None, value=1, size=(10,10), range=(0, 1), resolution=0.1) -> None:
        if self.running:
            print("add elements before start()")
            return
        # Track element type for height calculation
        self.element_types[key] = type
        aElement = Element(type, key, callback, value, range, resolution)
        if type == eType.BTTN or type == eType.CHCK:
            aElement = [aElement]
        self.layout.append(aElement)

    def addFrame(self, frame) -> None:
        if self.running:
            print("add elements before start()")
            return
        self.layout.append(frame)

    def addMessage(self, message: qMessage) -> None:
        if not self.running:
            print("gui has not started")
            return
        self.messageQueue.put(message)

    def updateElement(self, key, value, useCallback = False) -> None:
        self.addMessage(qMessage(qMessageType.EVENT, value, key, useCallback))

    def getValue(self, key:str):
        if not self.window or not self.allElementsLoaded: return None
        element =  self.window.find_element(key, True)
        if not element: return None
        return element.get()

    def getStringValue(self, key:str) -> str:
        result = self.getValue(key)
        if not result: return ''
        return str(result)

    def bringToFront(self) -> None:
        self.addMessage(qMessage(qMessageType.FRONT))

    def sendToBack(self) -> None:
        self.addMessage(qMessage(qMessageType.BACK))

    def set_settings_name(self, value: str) -> None:
        self.settings.filename = value + '.json'
        pass

    def get_setting_names(self) -> list[str]:
        xml_files: list[str] = glob.glob(os.path.join(str(self.settings.path), '*.json'))
        xml_filenames: list[str] = [os.path.splitext(os.path.basename(file))[0] for file in xml_files]
        return xml_filenames

    def saveSettings(self) -> None :
        self.addMessage(qMessage(qMessageType.SAVE))

    def loadSettings(self) -> None :
        self.addMessage(qMessage(qMessageType.LOAD))

    def isRunning(self) -> bool:
        return self.running

    def call_exit_callback(self) -> None:
        with self.exit_lock:
            exit_callback = self.exit_callback
        if exit_callback:
            exit_callback()

    def getDefaultFrameWidth(self) -> int:
        return self.defaultFrameWidth

    def get_element_height(self, elem_type: eType) -> int:
        """Get the height for a specific element type.

        Args:
            elem_type: The element type to get height for

        Returns:
            Height in pixels (50 for sliders, 40 for everything else)
        """
        if elem_type == eType.SLDR:
            return SLIDERHEIGHT
        return ELEMHEIGHT

    def calculate_total_elements_height(self) -> int:
        """Calculate total height of all tracked elements.

        Returns:
            Total height in pixels including base height
        """
        total_height = BASEHEIGHT

        for key, elem_type in self.element_types.items():
            total_height += self.get_element_height(elem_type)

        return total_height

    def get_tracked_element_count(self) -> int:
        """Get the number of tracked elements."""
        return len(self.element_types)

    def calculate_frame_height(self, element_list: list) -> int:
        """Calculate the required height for a frame based on its elements.

        Args:
            element_list: List of element rows (same format passed to Frame())

        Returns:
            Total height in pixels including base height
        """
        total_height = BASEHEIGHT

        for row in element_list:
            if isinstance(row, list):
                # Row of elements - check each element for type
                row_has_slider = False
                for elem in row:
                    # Check if it's a slider element
                    if hasattr(elem, 'Key') and hasattr(elem, 'Type'):
                        # PySimpleGUI element - check metadata or type
                        if 'Slider' in str(type(elem).__name__):
                            row_has_slider = True
                            break

                if row_has_slider:
                    total_height += SLIDERHEIGHT
                else:
                    total_height += ELEMHEIGHT
            else:
                # Single element
                total_height += ELEMHEIGHT

        return total_height
