"""Root settings tree for SMILE. Each module adds its settings as a Group here."""

from modules.settings import BaseSettings, Group, NiceSettings


class Settings(BaseSettings):
    server: Group[NiceSettings] = Group(NiceSettings)
