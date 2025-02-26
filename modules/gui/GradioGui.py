import gradio as gr
import threading
import json
from typing import Callable

class GradioGui:
    def __init__(self, fn, inputs, outputs) -> None:
        self.mainInterface = gr.Blocks()


        self.interface = gr.Interface(fn=fn, inputs=inputs, outputs=outputs)
        self.thread = threading.Thread(target=self.run, args=(), daemon=True)
        self.exit_callback: Callable | None = None
        gr.Textbox

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.thread.join()

    def run(self) -> None:
        self.interface.launch()

