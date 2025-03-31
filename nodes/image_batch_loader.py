import os
import random
import numpy as np
import torch
from PIL import Image, ImageOps
import hashlib
import re


class ImageBatchLoader:
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "filename")
    FUNCTION = "load_batch_images"
    CATEGORY = "Batch Process"

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

    def __init__(self):
        self.image_states = {}
        self.current_directory = ""
        self.images = []
        self.search_states = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING",),
                "search_title": ("STRING", {"default": ""}),
                "delimiter": ("STRING", {"default": ""}),
                "mode": (
                    ["single_image", "incremental_image", "random"],
                    {"default": "incremental_image"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "filename_option": (
                    [
                        "filename",
                        "prefix",
                        "suffix",
                        "prefix & suffix",
                        "prefix nor suffix",
                    ],
                ),
            },
        }

    def set_directory(
        self, directory, filename_option="filename", search_title="", delimiter=""
    ):
        if (
            directory != self.current_directory
            or filename_option
            or search_title
            or delimiter
        ):
            if not os.path.isdir(directory):
                raise ValueError(
                    f"The provided path '{directory}' is not a valid directory."
                )

            all_images = [
                f
                for f in os.listdir(directory)
                if any(f.endswith(ext) for ext in self.SUPPORTED_EXTENSIONS)
            ]
            filtered_images = self.filter_images(
                directory, all_images, filename_option, search_title, delimiter
            )

            self.images = sorted([os.path.join(directory, f) for f in filtered_images])
            self.current_directory = directory

            search_key = (directory, filename_option, search_title, delimiter)
            if search_key not in self.search_states:
                self.search_states[search_key] = 0

            if not self.images:
                print("No matching image files found in the provided directory.")
            else:
                pass

    def load_images(self, directory):
        if not os.path.isdir(directory):
            raise ValueError(f"Invalid directory: {directory}")

        all_images = [
            f
            for f in os.listdir(directory)
            if any(f.endswith(ext) for ext in self.SUPPORTED_EXTENSIONS)
        ]
        return sorted([os.path.join(directory, f) for f in all_images])

    def filter_images(self, directory, files, filename_option, search_title, delimiter):
        def get_prefix(filename):
            if delimiter:
                return filename.split(delimiter)[0]
            else:
                return re.split(r"[^a-zA-Z0-9]", filename)[0]

        def get_suffix(filename):
            name_without_ext = os.path.splitext(filename)[0]
            if delimiter:
                return name_without_ext.split(delimiter)[-1]
            else:
                return re.split(r"[^a-zA-Z0-9]", name_without_ext)[-1]

        filtered_files = files

        if search_title:
            if filename_option == "filename":
                filtered_files = [f for f in filtered_files if search_title in f]
            elif filename_option == "prefix":
                search_prefix = get_prefix(search_title)
                filtered_files = [
                    f for f in filtered_files if get_prefix(f) == search_prefix
                ]
            elif filename_option == "suffix":
                search_suffix = get_suffix(search_title)
                filtered_files = [
                    f for f in filtered_files if get_suffix(f) == search_suffix
                ]
            elif filename_option == "prefix & suffix":
                search_prefix = get_prefix(search_title)
                search_suffix = get_suffix(search_title)
                filtered_files = [
                    f
                    for f in filtered_files
                    if get_prefix(f) == search_prefix or get_suffix(f) == search_suffix
                ]
            elif filename_option == "prefix nor suffix":
                search_prefix = get_prefix(search_title)
                search_suffix = get_suffix(search_title)
                filtered_files = [
                    f
                    for f in filtered_files
                    if get_prefix(f) != search_prefix and get_suffix(f) != search_suffix
                ]

        return filtered_files

    def load_batch_images(
        self,
        directory,
        search_title="",
        delimiter="",
        mode="incremental_image",
        seed=0,
        filename_option="filename",
    ):
        self.set_directory(directory, filename_option, search_title, delimiter)

        if not self.images:
            return (torch.zeros(1, 64, 64, 3)), "no_images_found"

        search_key = (directory, filename_option, search_title, delimiter)

        if mode == "single_image":
            return self.load_image_by_index(search_key)
        elif mode == "incremental_image":
            return self.load_image_by_index(search_key)
        elif mode == "random":
            random.seed(seed)
            rnd_index = random.randint(0, len(self.images) - 1)
            return self.load_image_by_path(self.images[rnd_index])
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def load_image_by_index(self, search_key):
        if not self.images:
            print("No images loaded.")
            return None, None

        current_index = self.search_states[search_key]
        if current_index >= len(self.images):
            current_index = 0

        file_path = self.images[current_index]
        self.search_states[search_key] = (current_index + 1) % len(self.images)

        return self.load_image_by_path(file_path)

    def load_image_by_path(self, path):
        try:
            image = Image.open(path)
            image = ImageOps.exif_transpose(image).convert("RGB")
            filename = os.path.basename(path)
            # 去除文件扩展名
            filename = os.path.splitext(filename)[0]
            return self.pil2tensor(image), filename
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return (torch.zeros(1, 64, 64, 3)), "error"

    def pil2tensor(self, image):
        image_np = np.array(image).astype(np.float32) / 255.0
        if len(image_np.shape) == 2:
            image_np = np.expand_dims(image_np, axis=-1)
        image_np = np.expand_dims(image_np, axis=0)
        return torch.from_numpy(image_np)

    @classmethod
    def IS_CHANGED(cls, directory, **kwargs):
        if not os.path.exists(directory):
            return ""
        try:
            loader = cls()
            paths = loader.load_images(directory)
            return hashlib.sha256(",".join(paths).encode()).hexdigest()
        except Exception as e:
            print(f"Error checking for changes: {str(e)}")
            return ""
