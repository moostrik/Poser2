import streamlit as st
from threading import Thread
import json
from typing import Callable

class StreamlitGui(Thread):
    def __init__(self) -> None:
        super().__init__()
        self.exit_callback: Callable | None = None

        self.slider_value = 50
        self.settings_file = 'streamlit_settings.json'

    def stop(self) -> None:
        self.join()

    def run(self) -> None:
        st.title('Simplified Streamlit App')
        self.slider_value = st.slider('Select a value', 0, 100, self.slider_value)

        if st.button('Save Settings'):
            self.save_settings()

        if st.button('Load Settings'):
            self.load_settings()

    def save_settings(self):
        with open(self.settings_file, 'w') as f:
            json.dump({'slider_value': self.slider_value}, f)
        st.success('Settings saved!')

    def load_settings(self):
        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
                self.slider_value = settings.get('slider_value', 50)
            st.success('Settings loaded!')
        except FileNotFoundError:
            st.error('Settings file not found.')
