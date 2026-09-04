from .nodes.image_batch_process import ImageBatchLoader, ImageBatchSaver
from .nodes.text_batch_process import TXTBatchLoader, TextModifyTool
from .nodes.mask_batch_process import (
    MaskFromBatch,
    MaskRepeatBatch,
    MaskBatchCopy,
    MaskBatchComposite,
)
from .nodes.lora_batch_process import LoraBatchLoader
from .nodes.video_batch_process import VideoBatchSaver  # type: ignore
from .nodes.any_batch_process import AnyBatchGroup, AssetFilter
from .nodes.model3d_batch_process import Model3DBatchSaver, Model3DBatchPreview

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "ImageBatchLoader": ImageBatchLoader,
    "ImageBatchSaver": ImageBatchSaver,
    "TextModifyTool": TextModifyTool,
    "TXTBatchLoader": TXTBatchLoader,
    "LoraBatchLoader": LoraBatchLoader,
    "VideoBatchSaver": VideoBatchSaver,
    "AnyBatchGroup": AnyBatchGroup,
    "AssetFilter": AssetFilter,
    "Model3DBatchSaver": Model3DBatchSaver,
    "Model3DBatchPreview": Model3DBatchPreview,
    "MaskFromBatch": MaskFromBatch,
    "MaskRepeatBatch": MaskRepeatBatch,
    "MaskBatchCopy": MaskBatchCopy,
    "MaskBatchComposite": MaskBatchComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBatchLoader": "Image Batch Loader",
    "ImageBatchSaver": "Image Batch Saver",
    "TextModifyTool": "Text Modify Tool",
    "TXTBatchLoader": "TXT Batch Loader",
    "LoraBatchLoader": "LoRA Batch Loader",
    "VideoBatchSaver": "Video Batch Saver",
    "AnyBatchGroup": "Any Batch Group",
    "AssetFilter": "Asset Filter",
    "Model3DBatchSaver": "3D Model Batch Saver",
    "Model3DBatchPreview": "3D Model Batch Preview",
    "MaskFromBatch": "Mask From Batch",
    "MaskRepeatBatch": "Mask Repeat Batch",
    "MaskBatchCopy": "Mask Batch Copy",
    "MaskBatchComposite": "Mask Batch Composite",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
