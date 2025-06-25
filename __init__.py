from .nodes.image_batch_loader import ImageBatchLoader
from .nodes.image_batch_saver import ImageBatchSaver
from .nodes.simple_image_tagger import SimpleImageTagger
from .nodes.text_modify_tool import TextModifyTool
from .nodes.txt_batch_loader import TXTBatchLoader
from .nodes.lora_batch_loader import LoraBatchLoader

NODE_CLASS_MAPPINGS = {
    "ImageBatchLoader": ImageBatchLoader,
    "ImageBatchSaver": ImageBatchSaver,
    "SimpleImageTagger": SimpleImageTagger,
    "TextModifyTool": TextModifyTool,
    "TXTBatchLoader": TXTBatchLoader,
    "LoraBatchLoader": LoraBatchLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBatchLoader": "Image Batch Loader",
    "ImageBatchSaver": "Image Batch Saver",
    "SimpleImageTagger": "Simple Image Tagger",
    "TextModifyTool": "Text Modify Tool",
    "TXTBatchLoader": "TXT Batch Loader",
    "LoraBatchLoader": "LoRA Batch Loader",
}
