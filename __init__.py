from .nodes.image_batch_loader import ImageBatchLoader
from .nodes.image_batch_saver import ImageBatchSaver
from .nodes.image_list_loader import ImageListLoader
from .nodes.simple_image_tagger import SimpleImageTagger
from .nodes.text_modify_tool import TextModifyTool
from .nodes.txt_batch_loader import TXTBatchLoader

NODE_CLASS_MAPPINGS = {
    "ImageBatchLoader": ImageBatchLoader,
    "ImageBatchSaver": ImageBatchSaver,
    "ImageListLoader": ImageListLoader,
    "SimpleImageTagger": SimpleImageTagger,
    "TextModifyTool": TextModifyTool,
    "TXTBatchLoader": TXTBatchLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBatchLoader": "Image Batch Loader",
    "ImageBatchSaver": "Image Batch Saver",
    "ImageListLoader": "Image List Loader",
    "SimpleImageTagger": "Simple Image Tagger",
    "TextModifyTool": "Text Modify Tool",
    "TXTBatchLoader": "TXT Batch Loader",
}
