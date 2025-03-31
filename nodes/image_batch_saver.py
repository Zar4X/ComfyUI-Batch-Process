import os
import torch
import numpy as np
import json
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
from server import PromptServer
import re


class ImageBatchSaver:
    INPUT_IS_LIST = True
    FUNCTION = "save"
    CATEGORY = "Batch Process"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "file_paths")
    OUTPUT_NODE = True

    ALLOWED_EXT = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]

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
                    ["png", "jpg", "jpeg", "webp", "bmp", "tiff"],
                    {"default": "png"},
                ),
                "filename_number_padding": (
                    "INT",
                    {"default": 4, "min": 1, "max": 9, "step": 1},
                ),
                "filename_number": (
                    ["off", "start", "end", "start & end"],
                    {"default": "end"},
                ),
                "embeded_workflow": ("BOOLEAN", {"default": True}),
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
        embeded_workflow=True,  # 默认值改为布尔类型
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
        embeded_workflow = self._get_first_or_default(
            embeded_workflow, True
        )  # 改为布尔处理

        # 解析数字位置选项
        counter_start = filename_number in ["start", "start & end"]
        counter_end = filename_number in ["end", "start & end"]

        if extension not in ["png", "jpg", "jpeg", "webp", "bmp", "tiff"]:
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
            image_count = len(contents) if contents is not None else 0

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
                    full_path = os.path.join(
                        final_output_path, f"{base_filename}.{extension}"
                    )
                    self._save_image_tensor(
                        images[idx],
                        full_path,
                        embed_workflow=embeded_workflow,
                        prompt=prompt,
                        extra_pnginfo=extra_pnginfo,
                        extension=extension,
                    )
                    saved_files.append(full_path)
                    print(f"Saved image: {full_path}")

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

        return (images, saved_files)

    def _save_image_tensor(
        self, tensor, path, embed_workflow, prompt, extra_pnginfo, extension
    ):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3 and tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        img_array = np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)

        if embed_workflow:  # 直接使用布尔值判断
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
        pattern_start = re.compile(
            rf"^(\d{{{padding}}}){delimiter}{re.escape(prefix)}{re.escape(suffix)}.*$"
        )
        pattern_end = re.compile(
            rf"^{re.escape(prefix)}{delimiter}(\d{{{padding}}}){re.escape(suffix)}.*$"
        )

        numbers = []
        for file in existing_files:
            file_base = os.path.splitext(file)[0]
            if counter_start and not counter_end:
                match = pattern_start.match(file_base)
            elif counter_end and not counter_start:
                match = pattern_end.match(file_base)
            elif counter_start and counter_end:
                match = pattern_start.match(file_base) or pattern_end.match(file_base)
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
