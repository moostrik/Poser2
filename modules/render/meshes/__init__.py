from typing import Union


from .AngleMeshes import        AngleMeshes
from .PoseMeshesCapture import  PoseMeshesCapture
from .PoseMeshesRender import   PoseMeshesRender

PoseMesh = Union[PoseMeshesCapture, PoseMeshesRender]