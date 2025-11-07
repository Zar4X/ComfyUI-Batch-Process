import os
import random
import hashlib
import re
import sys
import io
import logging
import warnings
from contextlib import contextmanager
import folder_paths
import comfy.utils
import comfy.lora


@contextmanager
def suppress_comfy_logs():
    """Temporarily suppress stdout, stderr, logging, and warnings to hide verbose ComfyUI LoRA loading messages"""
    # Save original states
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_logging_level = logging.root.level

    # Also save file descriptors for low-level redirection
    old_stdout_fd = None
    old_stderr_fd = None
    devnull_fd = None

    try:
        # Suppress Python-level output
        logging.root.setLevel(logging.CRITICAL + 1)
        warnings.filterwarnings("ignore")

        # Create a devnull-like object for Python-level redirection
        devnull = io.StringIO()
        sys.stdout = devnull
        sys.stderr = devnull

        # Also redirect OS-level file descriptors (for C extensions, PyTorch, etc.)
        # This catches output that bypasses Python's sys.stdout/stderr
        try:
            sys.stdout.flush()
            sys.stderr.flush()

            # Duplicate the current stdout/stderr file descriptors
            old_stdout_fd = os.dup(1)
            old_stderr_fd = os.dup(2)

            # Open devnull and redirect stdout/stderr to it
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull_fd, 1)  # Redirect stdout (fd 1)
            os.dup2(devnull_fd, 2)  # Redirect stderr (fd 2)
        except (OSError, AttributeError):
            # If OS-level redirection fails, continue with Python-level only
            pass

        yield

    finally:
        # Restore Python-level output
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logging.root.setLevel(old_logging_level)
        warnings.filterwarnings("default")

        # Restore OS-level file descriptors
        if old_stdout_fd is not None:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(old_stdout_fd, 1)
                os.dup2(old_stderr_fd, 2)
                os.close(old_stdout_fd)
                os.close(old_stderr_fd)
                if devnull_fd is not None:
                    os.close(devnull_fd)
            except (OSError, AttributeError):
                pass


class LoraBatchLoader:
    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "filename")
    FUNCTION = "load_batch_loras"
    CATEGORY = "Batch Process"

    SUPPORTED_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".bin"}

    def __init__(self):
        self.lora_states = {}
        self.current_directory = ""
        self.loras = []
        self.search_states = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "directory": ("STRING",),
                "search_title": ("STRING", {"default": ""}),
                "delimiter": ("STRING", {"default": ""}),
                "mode": (
                    ["incremental", "random"],
                    {"default": "incremental"},
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
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
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

            all_loras = [
                f
                for f in os.listdir(directory)
                if any(f.lower().endswith(ext) for ext in self.SUPPORTED_EXTENSIONS)
            ]
            filtered_loras = self.filter_loras(
                directory, all_loras, filename_option, search_title, delimiter
            )

            self.loras = sorted([os.path.join(directory, f) for f in filtered_loras])
            self.current_directory = directory

            search_key = (directory, filename_option, search_title, delimiter)
            if search_key not in self.search_states:
                self.search_states[search_key] = 0

            if not self.loras:
                print("No matching LoRA files found in the provided directory.")
            else:
                print(f"Found {len(self.loras)} LoRA files in directory.")

    def load_loras(self, directory):
        if not os.path.isdir(directory):
            raise ValueError(f"Invalid directory: {directory}")

        all_loras = [
            f
            for f in os.listdir(directory)
            if any(f.lower().endswith(ext) for ext in self.SUPPORTED_EXTENSIONS)
        ]
        return sorted([os.path.join(directory, f) for f in all_loras])

    def filter_loras(self, directory, files, filename_option, search_title, delimiter):
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

    def load_batch_loras(
        self,
        model,
        clip,
        directory,
        search_title="",
        delimiter="",
        mode="incremental",
        seed=0,
        filename_option="filename",
        strength_model=1.0,
        strength_clip=1.0,
    ):
        self.set_directory(directory, filename_option, search_title, delimiter)

        if not self.loras:
            print("No LoRA files found, returning original model and clip.")
            return (model, clip, "no_loras_found")

        search_key = (directory, filename_option, search_title, delimiter)

        if mode == "incremental":
            return self.load_lora_by_index(
                model, clip, search_key, strength_model, strength_clip
            )
        elif mode == "random":
            random.seed(seed)
            rnd_index = random.randint(0, len(self.loras) - 1)
            print(
                f"[LoRA Batch Loader] Random mode - Index: {rnd_index + 1}/{len(self.loras)}"
            )
            return self.load_lora_by_path(
                model, clip, self.loras[rnd_index], strength_model, strength_clip
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def load_lora_by_index(
        self, model, clip, search_key, strength_model, strength_clip
    ):
        if not self.loras:
            print("No LoRAs loaded.")
            return model, clip, "no_loras"

        current_index = self.search_states[search_key]
        if current_index >= len(self.loras):
            current_index = 0

        file_path = self.loras[current_index]
        self.search_states[search_key] = (current_index + 1) % len(self.loras)

        # Print index info for better tracking
        print(f"[LoRA Batch Loader] Index: {current_index + 1}/{len(self.loras)}")
        return self.load_lora_by_path(
            model, clip, file_path, strength_model, strength_clip
        )

    def load_lora_by_path(self, model, clip, path, strength_model, strength_clip):
        try:
            filename = os.path.basename(path)
            # Remove file extension for cleaner filename output
            filename_clean = os.path.splitext(filename)[0]

            # Print LoRA name every time it's loaded
            print(f"[LoRA Batch Loader] Loading LoRA: {filename_clean}")

            # Suppress verbose ComfyUI LoRA loading messages
            with suppress_comfy_logs():
                # Load the LoRA using ComfyUI's built-in LoRA loading functionality
                lora = comfy.utils.load_torch_file(path, safe_load=True)

                # Create key mapping for the LoRA
                model_lora_keys = comfy.lora.model_lora_keys_unet(model.model)
                clip_lora_keys = comfy.lora.model_lora_keys_clip(clip.cond_stage_model)

                # Combine the key mappings
                key_map = {}
                key_map.update(model_lora_keys)
                key_map.update(clip_lora_keys)

                # Load the LoRA patches (this prints all the verbose messages)
                loaded = comfy.lora.load_lora(lora, key_map)

                # Apply LoRA to model
                new_modelpatcher = model.clone()
                k = {}
                for x in loaded:
                    k[x] = loaded[x]
                new_modelpatcher.add_patches(k, strength_model)

                # Apply LoRA to clip
                new_clip = clip.clone()
                k = {}
                for x in loaded:
                    k[x] = loaded[x]
                new_clip.add_patches(k, strength_clip)

            return (new_modelpatcher, new_clip, filename_clean)

        except Exception as e:
            print(f"Error loading LoRA {path}: {str(e)}")
            return (model, clip, "error_loading_lora")

    @classmethod
    def IS_CHANGED(cls, directory, **kwargs):
        if not os.path.exists(directory):
            return ""
        try:
            loader = cls()
            paths = loader.load_loras(directory)
            return hashlib.sha256(",".join(paths).encode()).hexdigest()
        except Exception as e:
            print(f"Error checking for changes: {str(e)}")
            return ""
