import os
import random
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence
import hashlib
import re
import glob
import node_helpers
from server import PromptServer


class ImageBatchLoader:
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("image", "filename", "image_count", "image_list")
    OUTPUT_IS_LIST = (False, False, False, True)
    FUNCTION = "load_batch_images"
    CATEGORY = "Batch Process"

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

    def __init__(self):
        self.image_states = {}
        self.current_directory = ""
        self.images = []
        self.search_states = {}
        self._last_scan_key = None
        self._all_image_paths = []

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
                "image_list": (
                    "BOOLEAN",
                    {
                        "label_on": "yes",
                        "label_off": "no",
                        "default": False,
                        "defaultInput": False,
                    },
                ),
                "subfolder": (
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

    def set_directory(
        self,
        directory,
        filename_option="filename",
        search_title="",
        delimiter="",
        subfolder=False,
    ):
        scan_key = (directory, filename_option, search_title, delimiter, subfolder)
        if scan_key != self._last_scan_key:
            if not os.path.isdir(directory):
                raise ValueError(
                    f"The provided path '{directory}' is not a valid directory."
                )

            # Fast list of all images (no PIL verification)
            all_image_paths = self.list_images(directory, subfolder)
            self._all_image_paths = all_image_paths

            # Extract just the filenames for filtering
            all_images = [os.path.basename(path) for path in all_image_paths]

            filtered_images = self.filter_images(
                directory, all_images, filename_option, search_title, delimiter
            )

            # Filter full paths in O(n) by basename membership
            allowed_names = set(filtered_images)
            self.images = [
                p for p in all_image_paths if os.path.basename(p) in allowed_names
            ]
            self.images = sorted(self.images)

            self.current_directory = directory
            self._last_scan_key = scan_key

            if scan_key not in self.search_states:
                self.search_states[scan_key] = 0

            if not self.images:
                print("No matching image files found in the provided directory.")

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

    @classmethod
    def list_images(cls, path: str, subfolder: bool = False, verify: bool = False):
        images = []

        if os.path.isfile(path):
            files = [path]
        else:
            if subfolder:
                files = []
                for root, _, filenames in os.walk(path):
                    for name in filenames:
                        files.append(os.path.join(root, name))
            else:
                try:
                    files = [
                        entry.path for entry in os.scandir(path) if entry.is_file()
                    ]
                except FileNotFoundError:
                    files = []

        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

        candidate_files = [
            f for f in files if os.path.splitext(f)[1].lower() in valid_extensions
        ]

        if not verify:
            return sorted(candidate_files)

        for filename in candidate_files:
            try:
                with Image.open(filename) as img:
                    img.verify()
                images.append(filename)
            except Exception:
                continue

        return sorted(images)

    def load_batch_images(
        self,
        directory,
        search_title="",
        delimiter="",
        mode="incremental_image",
        seed=0,
        filename_option="filename",
        image_list=False,
        subfolder=False,
        node_id=None,
    ):
        # Ensure directory is scanned once and cached
        self.set_directory(
            directory, filename_option, search_title, delimiter, subfolder
        )

        # Use cached listing for count and names
        all_images_in_dir = self._all_image_paths
        image_count = str(len(all_images_in_dir))

        # Only load all images if image_list is True
        if image_list:
            all_loaded_images = self.load_all_images(
                path=directory,
                subfolder=subfolder,
                node_id=node_id,
                filepaths=all_images_in_dir,
            )
            if all_loaded_images:
                return (
                    all_loaded_images[0],
                    os.path.splitext(os.path.basename(all_images_in_dir[0]))[0],
                    image_count,
                    all_loaded_images,
                )
            else:
                return (torch.zeros(1, 64, 64, 3)), "no_images_found", image_count, []
        else:
            # For regular mode, return empty list for image_list output (fast)
            empty_list = []

        if not self.images:
            return (
                (torch.zeros(1, 64, 64, 3)),
                "no_images_found",
                image_count,
                empty_list,
            )

        search_key = (directory, filename_option, search_title, delimiter, subfolder)

        if mode == "single_image":
            image, filename = self.load_image_by_index(search_key)
            return image, filename, image_count, empty_list
        elif mode == "incremental_image":
            image, filename = self.load_image_by_index(search_key)
            return image, filename, image_count, empty_list
        elif mode == "random":
            random.seed(seed)
            rnd_index = random.randint(0, len(self.images) - 1)
            image, filename = self.load_image_by_path(self.images[rnd_index])
            return image, filename, image_count, empty_list
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def load_all_images(
        self,
        path: str = None,
        subfolder: bool = False,
        node_id: str = None,
        filepaths: list = None,
    ):
        """Load all images for the image_list output"""
        images = []
        if filepaths is None:
            # Fallback to listing if not provided
            filepaths = self.list_images(path, subfolder)

        for index, image_path in enumerate(filepaths):
            try:
                img = node_helpers.pillow(Image.open, image_path)
                img = node_helpers.pillow(ImageOps.exif_transpose, img)

                if img.mode == "I":
                    img = img.point(lambda i: i * (1 / 255))
                img = img.convert("RGB")

                image_np = np.array(img).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None, ...]
                images.append(image_tensor)

                if node_id:
                    PromptServer.instance.send_sync(
                        "progress",
                        {"node": node_id, "max": len(filepaths), "value": index},
                    )
            except Exception as e:
                print(f"Error loading image {image_path}: {str(e)}")
                continue

        return images

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
