import os
import torch
import numpy as np
import json
from PIL import Image, ImageSequence
from PIL.PngImagePlugin import PngInfo
import folder_paths
from server import PromptServer
import re


class ImageBatchSaver:
    INPUT_IS_LIST = True
    FUNCTION = "save"
    CATEGORY = "Batch Process"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True

    ALLOWED_EXT = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"]

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always execute - return a unique value each time
        import time

        return time.time()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "images": ("IMAGE",),
                "contents": ("STRING", {"forceInput": True}),
                "output_path": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "IMG"}),
                "filename_delimiter": ("STRING", {"default": "_"}),
                "filename_suffix": ("STRING", {"default": ""}),
                "extension": (
                    ["png", "jpg", "jpeg", "webp", "bmp", "tiff", "gif"],
                    {"default": "png"},
                ),
                "filename_number_padding": (
                    "INT",
                    {"default": 4, "min": 1, "max": 9, "step": 1},
                ),
                "filename_number": (
                    ["off", "start", "end"],
                    {"default": "end"},
                ),
                "embeded_workflow": ("BOOLEAN", {"default": True}),
                "append_frames": (
                    "BOOLEAN",
                    {
                        "label_on": "true",
                        "label_off": "false",
                        "default": False,
                        "defaultInput": False,
                        "tooltip": "Append new frames to existing file with same base name. Useful for accumulating frames across loop iterations to reduce memory usage.",
                    },
                ),
                "start_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999999,
                        "step": 1,
                        "tooltip": "Start index (1-based) for frames to save. Use 0 to save all frames, or set 1 for first frame, 2 for second frame, etc.",
                    },
                ),
                "end_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999999,
                        "step": 1,
                        "tooltip": "End index (inclusive, 1-based) for frames to save. Use 0 to save all remaining frames, or set >0 to limit range.",
                    },
                ),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def save(
        self,
        images=None,
        contents=None,
        output_path="",
        filename_prefix="IMG",
        filename_delimiter="_",
        filename_suffix="",
        extension="png",
        filename_number_padding=4,
        filename_number="end",
        append_frames=False,
        start_index=0,
        end_index=0,
        embeded_workflow=True,
        node_id=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        # 输入验证和处理
        extension = self._get_first_or_default(extension, "png")
        filename_prefix = self._get_first_or_default(filename_prefix, "IMG")
        output_path = self._get_first_or_default(output_path, "")
        filename_number_padding = self._get_first_or_default(filename_number_padding, 4)
        filename_delimiter = self._get_first_or_default(filename_delimiter, "_")
        filename_suffix = self._get_first_or_default(filename_suffix, "")
        filename_suffix = filename_suffix.strip("'[]")
        filename_number = self._get_first_or_default(filename_number, "end")
        append_frames = self._get_first_or_default(append_frames, False)
        start_index = self._get_first_or_default(start_index, 0)
        end_index = self._get_first_or_default(end_index, 0)
        embeded_workflow = self._get_first_or_default(embeded_workflow, True)

        # 解析数字位置选项
        counter_start = filename_number in ["start"]
        counter_end = filename_number in ["end"]

        if extension not in ["png", "jpg", "jpeg", "webp", "bmp", "tiff", "gif"]:
            raise ValueError(f"Invalid extension: {extension}")

        if filename_number_padding < 1:
            raise ValueError(
                f"filename_number_padding must be at least 1, got {filename_number_padding}"
            )

        # 处理 images 输入
        if images is not None:
            processed_images = []
            for img in images if isinstance(images, list) else [images]:
                if (
                    isinstance(img, torch.Tensor)
                    and img.dim() == 4
                    and img.shape[0] > 1
                ):
                    batch_size = img.shape[0]
                    for i in range(batch_size):
                        processed_images.append(img[i].unsqueeze(0))
                else:
                    processed_images.append(img)
            images = processed_images
            image_count = len(images)
        else:
            images = []
            image_count = len(contents) if contents is not None else 1

        # 处理 output_path
        output_path = self._normalize_input(output_path, image_count)

        # 处理 filename_prefix
        filename_prefix = self._normalize_input(filename_prefix, image_count)

        # 处理文本输入
        contents = (
            self._normalize_input(contents, image_count)
            if contents is not None
            else None
        )

        # 准备输出目录
        output_dir = folder_paths.get_output_directory()

        # 批量保存
        saved_files = []
        for idx, (prefix, path) in enumerate(zip(filename_prefix, output_path)):
            try:
                final_output_path = self._get_output_path(output_dir, path)
                os.makedirs(final_output_path, exist_ok=True)

                original_filename = os.path.splitext(os.path.basename(prefix))[0]
                base_filename = self._generate_filename(
                    prefix=original_filename,
                    suffix=filename_suffix,
                    padding=filename_number_padding,
                    counter_start=counter_start,
                    counter_end=counter_end,
                    delimiter=filename_delimiter,
                    final_output_path=final_output_path,
                )

                # 保存图片
                if images and idx < len(images):
                    image_to_save = images[idx]

                    # Apply start_index and end_index filtering (1-based)
                    if (
                        isinstance(image_to_save, torch.Tensor)
                        and image_to_save.dim() == 4
                    ):
                        # Multi-frame tensor: (batch, height, width, channels)
                        num_frames = image_to_save.shape[0]
                        actual_start = (start_index - 1) if start_index > 0 else 0
                        actual_end = (
                            (end_index - 1) if end_index > 0 else (num_frames - 1)
                        )

                        # Validate and clamp indices
                        actual_start = max(0, min(actual_start, num_frames - 1))
                        actual_end = max(actual_start, min(actual_end, num_frames - 1))

                        if actual_start <= actual_end and num_frames > 0:
                            image_to_save = image_to_save[actual_start : actual_end + 1]
                        else:
                            image_to_save = image_to_save[0:1]  # At least one frame

                    # Handle append mode
                    if append_frames:
                        # Auto-convert to GIF if current format doesn't support multiple frames
                        original_extension = extension
                        if extension.lower() not in ["gif", "tiff", "webp"]:
                            # Automatically convert to GIF for append_frames (best for animations)
                            extension = "gif"
                            print(
                                f"Info: {original_extension} format doesn't support multiple frames. "
                                f"Auto-converting to GIF format for append_frames feature."
                            )

                        # Find existing file with same base name (try both original and converted extensions)
                        base_name_no_counter = self._get_base_name_without_counter(
                            original_filename, filename_suffix, filename_delimiter
                        )
                        existing_path = self._find_existing_file(
                            final_output_path,
                            base_name_no_counter,
                            extension,
                            counter_start,
                            counter_end,
                            filename_delimiter,
                        )

                        # If not found with converted extension, try original extension
                        if not existing_path and original_extension != extension:
                            existing_path = self._find_existing_file(
                                final_output_path,
                                base_name_no_counter,
                                original_extension,
                                counter_start,
                                counter_end,
                                filename_delimiter,
                            )
                            # If found with original extension, we'll convert it to new format
                            if existing_path:
                                print(
                                    f"Info: Found existing file with {original_extension} extension. "
                                    f"Will convert to {extension} format for multi-frame support."
                                )

                        if existing_path and os.path.exists(existing_path):
                            # Load existing frames and append new ones
                            existing_frames = self._load_image_frames(existing_path)
                            if existing_frames is not None:
                                # Concatenate: existing frames + new frames
                                if image_to_save.dim() == 3:
                                    image_to_save = image_to_save.unsqueeze(0)
                                if existing_frames.dim() == 3:
                                    existing_frames = existing_frames.unsqueeze(0)
                                image_to_save = torch.cat(
                                    [existing_frames, image_to_save], dim=0
                                )

                                # Update path extension if format was converted
                                if original_extension != extension:
                                    # Change extension in path
                                    base_path = os.path.splitext(existing_path)[0]
                                    full_path = f"{base_path}.{extension}"
                                else:
                                    full_path = existing_path

                                num_new_frames = (
                                    image_to_save.shape[0] - existing_frames.shape[0]
                                )
                                print(
                                    f"Appending {num_new_frames} frame(s) to file: {full_path} "
                                    f"(total: {image_to_save.shape[0]} frames)"
                                )
                            else:
                                # Failed to load, create new file with correct extension
                                full_path = os.path.join(
                                    final_output_path, f"{base_filename}.{extension}"
                                )
                        else:
                            # No existing file, create new one with correct extension
                            full_path = os.path.join(
                                final_output_path, f"{base_filename}.{extension}"
                            )
                    else:
                        # Normal mode: always create new file
                        full_path = os.path.join(
                            final_output_path, f"{base_filename}.{extension}"
                        )

                    self._save_image_tensor(
                        image_to_save,
                        full_path,
                        embed_workflow=embeded_workflow,
                        prompt=prompt,
                        extra_pnginfo=extra_pnginfo,
                        extension=extension,
                    )
                    saved_files.append(full_path)
                    print(f"Saved image: {full_path}")
                elif not images:
                    # 如果没有图像，仍然生成文件路径（可能用于其他用途）
                    full_path = os.path.join(
                        final_output_path, f"{base_filename}.{extension}"
                    )
                    saved_files.append(full_path)
                    print(f"Generated path: {full_path}")

                # 保存文本
                if contents and idx < len(contents):
                    content_path = os.path.join(
                        final_output_path, f"{base_filename}.txt"
                    )
                    with open(content_path, "w", encoding="utf-8") as f:
                        f.write(str(contents[idx]).strip())
                    saved_files.append(content_path)
                    print(f"Saved content: {content_path}")

            except Exception as e:
                print(f"Error saving file {idx+1}: {str(e)}")

            self._update_progress(node_id, idx + 1, image_count)

        return ()

    def _save_image_tensor(
        self, tensor, path, embed_workflow, prompt, extra_pnginfo, extension
    ):
        try:
            # Ensure tensor is on CPU
            if tensor.is_cuda:
                tensor = tensor.cpu()

            # Handle multi-frame tensors (4D: batch, height, width, channels)
            if tensor.dim() == 4 and tensor.shape[0] > 1:
                self._save_multi_frame_tensor(
                    tensor, path, embed_workflow, prompt, extra_pnginfo, extension
                )
                return

            # Handle single frame - normalize to 3D (H, W, C)
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)

            # Convert to numpy array
            if tensor.dtype == torch.uint8:
                # Already in uint8 format
                img_array = tensor.numpy()
            else:
                # Convert from float [0,1] to uint8 [0,255]
                img_array = np.clip(255.0 * tensor.numpy(), 0, 255).astype(np.uint8)

            # Handle different tensor shapes
            if img_array.ndim == 3:
                # 3D array: could be (C, H, W) or (H, W, C)
                if img_array.shape[0] == 3 or img_array.shape[0] == 1:
                    # (C, H, W) format - transpose to (H, W, C)
                    img_array = np.transpose(img_array, (1, 2, 0))
                elif img_array.shape[2] == 3 or img_array.shape[2] == 1:
                    # Already in (H, W, C) format
                    pass
                else:
                    # Unusual shape - try to reshape
                    # If it's a very wide/narrow image, handle it
                    if img_array.shape[0] == 1 and img_array.shape[1] == 1:
                        # Shape like (1, 1, width) - reshape to (1, width, 1) then add channel
                        img_array = img_array.reshape(img_array.shape[2], 1, 1)
                        img_array = np.transpose(img_array, (1, 0, 2))
                    elif img_array.shape[1] == 1:
                        # Shape like (height, 1, width) - transpose
                        img_array = np.transpose(img_array, (0, 2, 1))

            # Ensure we have valid image dimensions
            if img_array.ndim == 2:
                # Grayscale 2D image - add channel dimension
                img_array = np.expand_dims(img_array, axis=2)
            elif img_array.ndim == 1:
                # 1D array - try to reshape to 2D
                # Assume it's a square or try to infer dimensions
                size = int(np.sqrt(len(img_array)))
                if size * size == len(img_array):
                    img_array = img_array.reshape(size, size, 1)
                else:
                    raise ValueError(
                        f"Cannot reshape 1D array of length {len(img_array)} to image"
                    )

            # Ensure valid channel count (1 for grayscale, 3 for RGB)
            if img_array.shape[2] == 1:
                # Grayscale - keep as is, PIL will handle it
                img_array = img_array.squeeze(
                    2
                )  # Remove channel dimension for grayscale
            elif img_array.shape[2] > 3:
                # More than 3 channels - take first 3
                img_array = img_array[:, :, :3]
            elif img_array.shape[2] == 2:
                # 2 channels - duplicate to make 3
                img_array = np.concatenate([img_array, img_array[:, :, :1]], axis=2)

            # Create PIL Image
            if img_array.ndim == 2:
                # Grayscale
                img = Image.fromarray(img_array, mode="L")
            elif img_array.ndim == 3 and img_array.shape[2] == 3:
                # RGB
                img = Image.fromarray(img_array, mode="RGB")
            else:
                # Fallback - convert to RGB
                if img_array.ndim == 2:
                    img = Image.fromarray(img_array, mode="L").convert("RGB")
                else:
                    img = Image.fromarray(img_array[:, :, :3], mode="RGB")

        except Exception as e:
            print(f"Error processing tensor with shape {tensor.shape}: {str(e)}")
            # Fallback: try basic conversion
            try:
                tensor_np = tensor.cpu().numpy()
                if tensor_np.dtype != np.uint8:
                    tensor_np = np.clip(255.0 * tensor_np, 0, 255).astype(np.uint8)

                # Try to reshape to valid image
                while tensor_np.ndim < 2:
                    tensor_np = np.expand_dims(tensor_np, axis=0)
                while tensor_np.ndim > 3:
                    tensor_np = tensor_np.squeeze(0)

                if tensor_np.ndim == 2:
                    img = Image.fromarray(tensor_np, mode="L").convert("RGB")
                elif tensor_np.ndim == 3:
                    if tensor_np.shape[0] == 3 or tensor_np.shape[0] == 1:
                        tensor_np = np.transpose(tensor_np, (1, 2, 0))
                    if tensor_np.shape[2] == 1:
                        tensor_np = tensor_np.squeeze(2)
                        img = Image.fromarray(tensor_np, mode="L").convert("RGB")
                    else:
                        img = Image.fromarray(tensor_np[:, :, :3], mode="RGB")
                else:
                    raise ValueError(f"Cannot convert tensor with shape {tensor.shape}")
            except Exception as e2:
                raise ValueError(
                    f"Failed to save image: {str(e)}, fallback also failed: {str(e2)}"
                )

        if embed_workflow:
            if extension.lower() == "webp":
                exif_data = img.getexif()
                if prompt is not None:
                    prompt_str = json.dumps(prompt)
                    exif_data[0x010F] = "Prompt:" + prompt_str
                if extra_pnginfo is not None:
                    if isinstance(extra_pnginfo, list):
                        merged_extra = {}
                        for item in extra_pnginfo:
                            if isinstance(item, dict):
                                merged_extra.update(item)
                        extra_pnginfo = merged_extra
                    if isinstance(extra_pnginfo, dict):
                        workflow_metadata = json.dumps(extra_pnginfo)
                        exif_data[0x010E] = "Workflow:" + workflow_metadata
                exif_bytes = exif_data.tobytes()
                img.save(path, exif=exif_bytes)
            else:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    if isinstance(extra_pnginfo, list):
                        merged_extra = {}
                        for item in extra_pnginfo:
                            if isinstance(item, dict):
                                merged_extra.update(item)
                        extra_pnginfo = merged_extra
                    if isinstance(extra_pnginfo, dict):
                        for key in extra_pnginfo:
                            metadata.add_text(key, json.dumps(extra_pnginfo[key]))
                img.save(path, pnginfo=metadata)
        else:
            img.save(path)

    def _save_multi_frame_tensor(
        self, tensor, path, embed_workflow, prompt, extra_pnginfo, extension
    ):
        """Save a 4D tensor (batch, height, width, channels) as multi-frame image"""
        frames = []
        for i in range(tensor.shape[0]):
            frame = tensor[i]

            # Ensure frame is on CPU
            if frame.is_cuda:
                frame = frame.cpu()

            # Convert to numpy
            if frame.dtype == torch.uint8:
                frame_np = frame.numpy()
            else:
                frame_np = np.clip(255.0 * frame.numpy(), 0, 255).astype(np.uint8)

            # Handle different frame shapes
            if frame_np.ndim == 3:
                # 3D: could be (C, H, W) or (H, W, C)
                if frame_np.shape[0] == 3 or frame_np.shape[0] == 1:
                    # (C, H, W) format - transpose to (H, W, C)
                    frame_np = np.transpose(frame_np, (1, 2, 0))
                elif frame_np.shape[2] == 3 or frame_np.shape[2] == 1:
                    # Already in (H, W, C) format
                    pass
                else:
                    # Try to fix unusual shapes
                    if frame_np.shape[0] == 1:
                        frame_np = np.transpose(frame_np, (1, 2, 0))
            elif frame_np.ndim == 2:
                # 2D grayscale - add channel dimension
                frame_np = np.expand_dims(frame_np, axis=2)

            # Ensure valid format for PIL
            if frame_np.ndim == 2:
                # Grayscale
                pil_img = Image.fromarray(frame_np, mode="L")
            elif frame_np.ndim == 3:
                if frame_np.shape[2] == 1:
                    # Single channel - convert to grayscale
                    pil_img = Image.fromarray(frame_np.squeeze(2), mode="L")
                elif frame_np.shape[2] == 3:
                    # RGB
                    pil_img = Image.fromarray(frame_np, mode="RGB")
                else:
                    # More channels - take first 3
                    pil_img = Image.fromarray(frame_np[:, :, :3], mode="RGB")
            else:
                raise ValueError(
                    f"Unsupported frame shape: {frame.shape} -> {frame_np.shape}"
                )

            frames.append(pil_img)

        # Save metadata only on first frame
        first_frame = frames[0]
        if embed_workflow:
            if extension.lower() == "webp":
                exif_data = first_frame.getexif()
                if prompt is not None:
                    prompt_str = json.dumps(prompt)
                    exif_data[0x010F] = "Prompt:" + prompt_str
                if extra_pnginfo is not None:
                    if isinstance(extra_pnginfo, list):
                        merged_extra = {}
                        for item in extra_pnginfo:
                            if isinstance(item, dict):
                                merged_extra.update(item)
                        extra_pnginfo = merged_extra
                    if isinstance(extra_pnginfo, dict):
                        workflow_metadata = json.dumps(extra_pnginfo)
                        exif_data[0x010E] = "Workflow:" + workflow_metadata
                exif_bytes = exif_data.tobytes()
                first_frame.save(
                    path,
                    save_all=True,
                    append_images=frames[1:] if len(frames) > 1 else [],
                    duration=100,  # 100ms per frame
                    loop=0,
                    exif=exif_bytes,
                )
            elif extension.lower() == "gif":
                first_frame.save(
                    path,
                    save_all=True,
                    append_images=frames[1:] if len(frames) > 1 else [],
                    duration=100,
                    loop=0,
                )
            elif extension.lower() == "tiff":
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    if isinstance(extra_pnginfo, list):
                        merged_extra = {}
                        for item in extra_pnginfo:
                            if isinstance(item, dict):
                                merged_extra.update(item)
                        extra_pnginfo = merged_extra
                    if isinstance(extra_pnginfo, dict):
                        for key in extra_pnginfo:
                            metadata.add_text(key, json.dumps(extra_pnginfo[key]))
                first_frame.save(
                    path,
                    save_all=True,
                    append_images=frames[1:] if len(frames) > 1 else [],
                    pnginfo=metadata,
                )
            else:
                # For formats that don't support multiple frames, save only first frame
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    if isinstance(extra_pnginfo, list):
                        merged_extra = {}
                        for item in extra_pnginfo:
                            if isinstance(item, dict):
                                merged_extra.update(item)
                        extra_pnginfo = merged_extra
                    if isinstance(extra_pnginfo, dict):
                        for key in extra_pnginfo:
                            metadata.add_text(key, json.dumps(extra_pnginfo[key]))
                first_frame.save(path, pnginfo=metadata)
                print(
                    f"Warning: {extension} doesn't support multiple frames. Only first frame saved."
                )
        else:
            if extension.lower() in ["gif", "webp"]:
                first_frame.save(
                    path,
                    save_all=True,
                    append_images=frames[1:] if len(frames) > 1 else [],
                    duration=100,
                    loop=0,
                )
            elif extension.lower() == "tiff":
                first_frame.save(
                    path,
                    save_all=True,
                    append_images=frames[1:] if len(frames) > 1 else [],
                )
            else:
                first_frame.save(path)
                print(
                    f"Warning: {extension} doesn't support multiple frames. Only first frame saved."
                )

    def _get_base_name_without_counter(self, prefix, suffix, delimiter):
        """Generate base filename without counter"""
        parts = [str(prefix)]
        if suffix and suffix.strip():
            parts.append(str(suffix))
        return delimiter.join(parts)

    def _find_existing_file(
        self, directory, base_name, extension, counter_start, counter_end, delimiter
    ):
        """Find existing file with same base name (ignoring counter)"""
        if not os.path.exists(directory):
            return None

        existing_files = os.listdir(directory)
        base_pattern = re.escape(base_name)

        # Build patterns to match files with this base name
        if counter_start and counter_end:
            # Counter can be at start or end
            pattern_start = re.compile(
                rf"^(\d+){re.escape(delimiter)}{base_pattern}\.{re.escape(extension)}$"
            )
            pattern_end = re.compile(
                rf"^{base_pattern}{re.escape(delimiter)}(\d+)\.{re.escape(extension)}$"
            )
        elif counter_start:
            pattern_start = re.compile(
                rf"^(\d+){re.escape(delimiter)}{base_pattern}\.{re.escape(extension)}$"
            )
            pattern_end = None
        elif counter_end:
            pattern_start = None
            pattern_end = re.compile(
                rf"^{base_pattern}{re.escape(delimiter)}(\d+)\.{re.escape(extension)}$"
            )
        else:
            # No counter, exact match
            exact_name = f"{base_name}.{extension}"
            exact_path = os.path.join(directory, exact_name)
            if os.path.exists(exact_path):
                return exact_path
            return None

        # Find matching files
        matching_files = []
        for file in existing_files:
            if not file.endswith(f".{extension}"):
                continue
            file_base = os.path.splitext(file)[0]

            match = None
            if pattern_start:
                match = pattern_start.match(file_base)
            if not match and pattern_end:
                match = pattern_end.match(file_base)

            if match:
                matching_files.append((file, int(match.group(1))))

        if matching_files:
            # Return the file with the highest counter (most recent)
            matching_files.sort(key=lambda x: x[1], reverse=True)
            return os.path.join(directory, matching_files[0][0])

        return None

    def _load_image_frames(self, path):
        """Load an image file and return all frames as a tensor"""
        try:
            img = Image.open(path)
            frames = []

            # Handle multi-frame images (GIF, TIFF, WebP)
            try:
                for frame in ImageSequence.Iterator(img):
                    frame = frame.convert("RGB")
                    img_array = np.array(frame).astype(np.float32) / 255.0
                    frames.append(torch.from_numpy(img_array))
            except Exception:
                # Single frame image
                img = img.convert("RGB")
                img_array = np.array(img).astype(np.float32) / 255.0
                frames.append(torch.from_numpy(img_array))

            if len(frames) == 0:
                return None

            if len(frames) == 1:
                return frames[0]
            else:
                # Stack frames along batch dimension: (num_frames, H, W, C)
                return torch.stack(frames, dim=0)
        except Exception as e:
            print(f"Error loading image frames from {path}: {str(e)}")
            return None

    def _get_first_or_default(self, value, default):
        if isinstance(value, list):
            for v in value:
                if isinstance(v, type(default)):
                    return v
            return default
        return value if value is not None else default

    def _normalize_input(self, input_data, count):
        if input_data is None:
            return [None] * count
        if not isinstance(input_data, list):
            return [input_data] * count
        return input_data + [input_data[-1]] * (count - len(input_data))

    def _get_output_path(self, base_dir, user_path):
        if not user_path or str(user_path).lower() in ["none", "."]:
            return base_dir
        return (
            os.path.join(base_dir, user_path)
            if not os.path.isabs(str(user_path))
            else user_path
        )

    def _generate_filename(
        self,
        prefix,
        suffix,
        padding,
        counter_start,
        counter_end,
        delimiter,
        final_output_path,
    ):
        existing_files = os.listdir(final_output_path)

        if suffix.strip():
            pattern_start = re.compile(
                rf"^(\d+){re.escape(delimiter)}{re.escape(prefix)}{re.escape(delimiter)}{re.escape(suffix)}$"
            )
            pattern_end = re.compile(
                rf"^{re.escape(prefix)}{re.escape(delimiter)}{re.escape(suffix)}{re.escape(delimiter)}(\d+)$"
            )
        else:
            pattern_start = re.compile(
                rf"^(\d+){re.escape(delimiter)}{re.escape(prefix)}$"
            )
            pattern_end = re.compile(
                rf"^{re.escape(prefix)}{re.escape(delimiter)}(\d+)$"
            )

        numbers = []
        for file in existing_files:
            file_base = os.path.splitext(file)[0]
            if counter_start and not counter_end:
                match = pattern_start.match(file_base)
            elif counter_end and not counter_start:
                match = pattern_end.match(file_base)
            elif counter_start and counter_end:
                match_start = pattern_start.match(file_base)
                match_end = pattern_end.match(file_base)
                match = match_start or match_end
                if match_start and match_end:
                    numbers.extend([int(match_start.group(1)), int(match_end.group(1))])
                    continue
            else:
                match = None
            if match:
                numbers.append(int(match.group(1)))

        counter = max(numbers) + 1 if numbers else 1
        parts = []
        if counter_start:
            parts.append(f"{counter:0{padding}d}")
        parts.append(str(prefix))
        if suffix.strip():
            parts.append(str(suffix))
        if counter_end:
            parts.append(f"{counter:0{padding}d}")

        return delimiter.join(parts)

    def _update_progress(self, node_id, current, total):
        if node_id:
            PromptServer.instance.send_sync(
                "progress", {"node": node_id, "value": current, "max": total}
            )
