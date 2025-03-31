import os
import glob
import hashlib
from PIL import Image, ImageSequence, ImageOps
import torch
import numpy as np
import node_helpers
from server import PromptServer


class ImageListLoader:
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "masks", "filenames")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "load"
    CATEGORY = "Batch Process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING",),
                "recursive": (
                    "BOOLEAN",
                    {
                        "label_on": "yes",
                        "label_off": "no",
                        "default": False,
                        "defaultInput": False,
                    },
                ),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, path: str, recursive: bool = False, **kwargs):
        m = hashlib.sha256(path.encode("utf-8"))
        for image in cls.list_images(path, recursive):
            with open(image, "rb") as f:
                m.update(f.read())
        return m.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, path: str):
        if os.path.exists(path):
            return True
        return f'"{path}" does not exist'

    @classmethod
    def list_images(cls, path: str, recursive: bool = False):
        images = []
        pattern = "**/**" if recursive else "*"

        if os.path.isfile(path):
            files = [path]
        else:
            files = sorted(glob.glob(os.path.join(path, pattern), recursive=recursive))

        valid_extensions = [".jpg", ".jpeg", ".png", ".webp"]

        for filename in files:
            if os.path.isdir(filename):
                continue
            if not any(filename.lower().endswith(ext) for ext in valid_extensions):
                continue
            try:
                with Image.open(filename) as img:
                    img.verify()
                    images.append(filename)
            except (IOError, SyntaxError):
                pass
        return images

    def load(self, path: str, recursive: bool = False, node_id: str = None):
        images = []
        masks = []
        filenames = []
        filepaths = self.list_images(path, recursive)

        for index, image_path in enumerate(filepaths):
            img = node_helpers.pillow(Image.open, image_path)

            output_images = []
            output_masks = []
            w, h = None, None

            excluded_formats = ["MPO"]

            for i in ImageSequence.Iterator(img):
                i = node_helpers.pillow(ImageOps.exif_transpose, i)

                if i.mode == "I":
                    i = i.point(lambda i: i * (1 / 255))
                image = i.convert("RGB")

                if len(output_images) == 0:
                    w = image.size[0]
                    h = image.size[1]

                if image.size[0] != w or image.size[1] != h:
                    continue

                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None, ...]
                if "A" in i.getbands():
                    mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                    mask = 1.0 - torch.from_numpy(mask)
                else:
                    mask = torch.zeros(
                        (image.shape[1], image.shape[2]), dtype=torch.float32
                    )
                output_images.append(image)
                output_masks.append(mask.unsqueeze(0))

            if len(output_images) > 1 and img.format not in excluded_formats:
                output_image = torch.cat(output_images, dim=0)
                output_mask = torch.cat(output_masks, dim=0)
            else:
                output_image = output_images[0]
                output_mask = output_masks[0]

            images.append(output_image)
            masks.append(output_mask)
            filenames.append(os.path.splitext(os.path.basename(image_path))[0])

            PromptServer.instance.send_sync(
                "progress", {"node": node_id, "max": len(filepaths), "value": index}
            )

        return images, masks, filenames
