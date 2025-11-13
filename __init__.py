from .nodes.image_batch_loader import ImageBatchLoader
from .nodes.image_batch_saver import ImageBatchSaver
from .nodes.text_modify_tool import TextModifyTool
from .nodes.txt_batch_loader import TXTBatchLoader
from .nodes.lora_batch_loader import LoraBatchLoader
from .nodes.video_batch_saver import VideoBatchSaver  # type: ignore

NODE_CLASS_MAPPINGS = {
    "ImageBatchLoader": ImageBatchLoader,
    "ImageBatchSaver": ImageBatchSaver,
    "TextModifyTool": TextModifyTool,
    "TXTBatchLoader": TXTBatchLoader,
    "LoraBatchLoader": LoraBatchLoader,
    "VideoBatchSaver": VideoBatchSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBatchLoader": "Image Batch Loader",
    "ImageBatchSaver": "Image Batch Saver",
    "TextModifyTool": "Text Modify Tool",
    "TXTBatchLoader": "TXT Batch Loader",
    "LoraBatchLoader": "LoRA Batch Loader",
    "VideoBatchSaver": "Video Batch Saver",
}
