from dataclasses import dataclass, field

from modules.pose.detection.MMDetection import ModelType

@dataclass
class Settings:
    max_poses: int =                field(default=1)
    model_type: ModelType =         field(default=ModelType.SMALL)
    model_path: str =               field(default="models")


    confidence_low: float =         field(default=0.5)
    confidence_high: float =        field(default=0.7)
    verbose: bool =                 field(default=False)
    crop_expansion: float =         field(default=0.0)

    # Segmentation settings
    segmentation_enabled: bool =    field(default=True)
    # segmentation_model_name: str =  field(default="modnet_webcam_portrait_matting.ckpt")
    segmentation_model_name: str =  field(default="rvm_mobilenetv3_fp32.onnx")